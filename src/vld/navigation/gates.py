r"""Phase 4 — Pre-Execution Verification Gates: Interval Abstract Interpretation & Strong Bisimulation (`vld.navigation.gates`).

Governing equations:
1. **Exact Extended-Rational Interval Abstract Interpretation (`IntervalBound`, `analyze_dag_intervals`):**
   Over closed extended rational intervals $[a, b] \subseteq \mathbb{Q} \cup \{-\infty, +\infty\}$ ($a \le b$),
   abstract operators $\hat{f}$ satisfy Galois soundness $\forall x \in \gamma([a, b]),\; f(x) \in \gamma(\hat{f}([a, b]))$.
   Against a finite node safety box $S_v = [L_v, U_v]$, the Kleene $\mathcal{K}_3$ safety pre-gate evaluates:
     $$\operatorname{Safe}([a, b], [L_v, U_v]) = \begin{cases}
       \text{TRUE} & \text{if } a \neq -\infty \land b \neq +\infty \land L_v \le a \land b \le U_v, \\
       \text{FALSE} & \text{if } (b \neq +\infty \land b < L_v) \lor (a \neq -\infty \land a > U_v), \\
       \text{INDETERMINATE} & \text{otherwise.}
     \end{cases}$$
2. **Strong Bisimulation Equivalence (`check_strong_bisimulation`):**
   Given two finite labeled transition systems $\mathcal{T}_i = (\Sigma_i, A, \Delta_i, O_i)$, relation
   $R \subseteq (\Sigma_1 \sqcup \Sigma_2)^2$ is a strong bisimulation iff $(p, q) \in R \implies O(p) = O(q)$
   and for every $a \in A$, $\Delta_a(p)$ and $\Delta_a(q)$ hit identical $R$-equivalence blocks.
   Partition refinement computes the coarsest strong bisimulation $\sim_{\text{bisim}}$ and extracts a
   shortest distinguishing action trace via BFS whenever $s_1 \not\sim_{\text{bisim}} s_2$.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest
from vld.navigation.dag import ComputationDAGError, NavigableComputationDAG
from vld.triad import TruthDisposition


def _ext_le(a: Fraction | None, b: Fraction | None, *, a_is_neg_inf: bool, b_is_pos_inf: bool) -> bool:
    if a is None and a_is_neg_inf:
        return True
    if b is None and b_is_pos_inf:
        return True
    if a is None or b is None:
        return False
    return a <= b


@dataclass(frozen=True, slots=True)
class IntervalBound:
    r"""Exact extended rational interval $[l, u] \subseteq \mathbb{Q} \cup \{-\infty, +\infty\}$ (`None` = $\mp\infty$)."""

    lo: Fraction | None
    hi: Fraction | None

    def __post_init__(self) -> None:
        if isinstance(self.lo, (bool, float)) or isinstance(self.hi, (bool, float)):
            raise ValueError("IntervalBound endpoints must be exact int/Fraction/str or None (+/-inf), not bool/float.")
        lo_f = None if self.lo is None else Fraction(self.lo)
        hi_f = None if self.hi is None else Fraction(self.hi)
        if lo_f is not None and hi_f is not None and lo_f > hi_f:
            raise ValueError(f"Inverted IntervalBound: {lo_f} > {hi_f}.")
        object.__setattr__(self, "lo", lo_f)
        object.__setattr__(self, "hi", hi_f)

    @classmethod
    def unbounded(cls) -> IntervalBound:
        return cls(None, None)

    @property
    def is_finite(self) -> bool:
        return self.lo is not None and self.hi is not None

    def __add__(self, other: IntervalBound) -> IntervalBound:
        lo = None if (self.lo is None or other.lo is None) else (self.lo + other.lo)
        hi = None if (self.hi is None or other.hi is None) else (self.hi + other.hi)
        return IntervalBound(lo, hi)

    def __sub__(self, other: IntervalBound) -> IntervalBound:
        lo = None if (self.lo is None or other.hi is None) else (self.lo - other.hi)
        hi = None if (self.hi is None or other.lo is None) else (self.hi - other.lo)
        return IntervalBound(lo, hi)

    def __mul__(self, other: IntervalBound) -> IntervalBound:
        if self.lo == 0 and self.hi == 0:
            return IntervalBound(Fraction(0), Fraction(0))
        if other.lo == 0 and other.hi == 0:
            return IntervalBound(Fraction(0), Fraction(0))
        if not self.is_finite or not other.is_finite:
            # Exact sign-aware extended multiplication
            self_nonneg = self.lo is not None and self.lo >= 0
            other_nonneg = other.lo is not None and other.lo >= 0
            if self_nonneg and other_nonneg:
                assert self.lo is not None and other.lo is not None
                hi = None if (self.hi is None or other.hi is None) else (self.hi * other.hi)
                return IntervalBound(self.lo * other.lo, hi)
            return IntervalBound.unbounded()
        assert self.lo is not None and self.hi is not None and other.lo is not None and other.hi is not None
        prods = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return IntervalBound(min(prods), max(prods))

    def truediv_exact(self, other: IntervalBound) -> IntervalBound:
        if other.lo is None or other.hi is None:
            if other.lo is not None and other.lo > 0:
                inv = IntervalBound(Fraction(0), Fraction(1, 1) / other.lo)
                return self * inv
            if other.hi is not None and other.hi < 0:
                inv = IntervalBound(Fraction(1, 1) / other.hi, Fraction(0))
                return self * inv
            raise ZeroDivisionError("Interval divisor straddles zero.")
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Interval divisor straddles or equals zero.")
        inv = IntervalBound(Fraction(1, 1) / other.hi, Fraction(1, 1) / other.lo)
        return self * inv

    def relu(self) -> IntervalBound:
        lo = Fraction(0) if self.lo is None else max(Fraction(0), self.lo)
        hi = None if self.hi is None else max(Fraction(0), self.hi)
        return IntervalBound(lo, hi)

    def affine(self, scale: int | Fraction | str, bias: int | Fraction | str) -> IntervalBound:
        s = Fraction(scale)
        b = Fraction(bias)
        if s == 0:
            return IntervalBound(b, b)
        if s > 0:
            lo = None if self.lo is None else (s * self.lo + b)
            hi = None if self.hi is None else (s * self.hi + b)
            return IntervalBound(lo, hi)
        lo = None if self.hi is None else (s * self.hi + b)
        hi = None if self.lo is None else (s * self.lo + b)
        return IntervalBound(lo, hi)

    def clamp(self, min_v: int | Fraction | str, max_v: int | Fraction | str) -> IntervalBound:
        mn, mx = Fraction(min_v), Fraction(max_v)
        if mn > mx:
            raise ValueError("clamp min_v > max_v.")
        lo = mn if self.lo is None else min(mx, max(mn, self.lo))
        hi = mx if self.hi is None else max(mn, min(mx, self.hi))
        return IntervalBound(lo, hi)

    def check_within(self, allowed: IntervalBound) -> TruthDisposition:
        r"""Evaluate Kleene $\mathcal{K}_3$ containment $\text{self} \subseteq \text{allowed}$."""
        # Disjoint -> definitely FALSE
        if self.hi is not None and allowed.lo is not None and self.hi < allowed.lo:
            return TruthDisposition.FALSE
        if self.lo is not None and allowed.hi is not None and self.lo > allowed.hi:
            return TruthDisposition.FALSE
        # Contained -> definitely TRUE
        lo_ok = _ext_le(allowed.lo, self.lo, a_is_neg_inf=True, b_is_pos_inf=False)
        hi_ok = _ext_le(self.hi, allowed.hi, a_is_neg_inf=False, b_is_pos_inf=True)
        if lo_ok and hi_ok:
            return TruthDisposition.TRUE
        return TruthDisposition.INDETERMINATE

    def to_dict(self) -> dict[str, str]:
        return {
            "hi": "+inf" if self.hi is None else str(self.hi),
            "lo": "-inf" if self.lo is None else str(self.lo),
        }


@dataclass(frozen=True, slots=True)
class IntervalGateReport:
    """Result of static interval abstract interpretation over a `NavigableComputationDAG`."""

    bounds: dict[str, IntervalBound]
    dispositions: dict[str, TruthDisposition]
    overall_disposition: TruthDisposition
    report_digest: str


def analyze_dag_intervals(
    dag: NavigableComputationDAG,
    *,
    input_bounds: Mapping[str, IntervalBound] = (),
    safety_contracts: Mapping[str, IntervalBound] = (),
) -> IntervalGateReport:
    r"""Propagate Galois-sound extended rational intervals across DAG operations before execution.

    Supported node `operation` types:
      - `"const"` (`payload={"value": "c"}`) or node ID present in `input_bounds`
      - `"add"`, `"sub"`, `"mul"`, `"div"` (binary over `dependencies=(u, v)`)
      - `"relu"` (unary over `dependencies=(u,)`)
      - `"affine"` (`dependencies=(u,)`, `payload={"scale": s, "bias": b}`)
      - `"clamp"` (`dependencies=(u,)`, `payload={"min": lo, "max": hi}`)
      - `"mod_residue"` (`dependencies=(u,)`, `payload={"modulus": m}` -> sound enclosure over $\mathbb{Q}$)
    """
    topo = dag.topological_order()
    in_map = dict(input_bounds) if isinstance(input_bounds, Mapping) else {}
    contracts = dict(safety_contracts) if isinstance(safety_contracts, Mapping) else {}

    bounds: dict[str, IntervalBound] = {}
    dispositions: dict[str, TruthDisposition] = {}
    dead_nodes: set[str] = set()
    overall = TruthDisposition.TRUE

    for nid in topo:
        node = dag.nodes[nid]
        op = node.operation
        p = node.payload
        deps = node.dependencies

        if any(d in dead_nodes for d in deps):
            dead_nodes.add(nid)
            bounds[nid] = IntervalBound.unbounded()
            dispositions[nid] = TruthDisposition.FALSE
            overall = overall & TruthDisposition.FALSE
            continue

        div_hazard: TruthDisposition | None = None
        if nid in in_map:
            b = in_map[nid]
        elif op == "const":
            val = Fraction(str(p["value"]))
            b = IntervalBound(val, val)
        elif op == "add" and len(deps) == 2:
            b = bounds[deps[0]] + bounds[deps[1]]
        elif op == "sub" and len(deps) == 2:
            b = bounds[deps[0]] - bounds[deps[1]]
        elif op == "mul" and len(deps) == 2:
            b = bounds[deps[0]] * bounds[deps[1]]
        elif op == "div" and len(deps) == 2:
            denom = bounds[deps[1]]
            if denom.lo == 0 and denom.hi == 0:
                dead_nodes.add(nid)
                dispositions[nid] = TruthDisposition.FALSE
                overall = overall & TruthDisposition.FALSE
                bounds[nid] = IntervalBound.unbounded()
                continue
            straddles_zero = (
                (denom.lo is None and denom.hi is None)
                or (denom.lo is None and denom.hi is not None and denom.hi >= 0)
                or (denom.hi is None and denom.lo is not None and denom.lo <= 0)
                or (denom.lo is not None and denom.hi is not None and denom.lo <= 0 <= denom.hi)
            )
            if straddles_zero:
                # Galois-sound enclosure over y != 0 when y straddles 0 is (-inf, +inf)!
                b = IntervalBound.unbounded()
                div_hazard = TruthDisposition.INDETERMINATE
            else:
                b = bounds[deps[0]].truediv_exact(denom)
        elif op == "relu" and len(deps) == 1:
            b = bounds[deps[0]].relu()
        elif op == "affine" and len(deps) == 1:
            b = bounds[deps[0]].affine(str(p["scale"]), str(p["bias"]))
        elif op == "clamp" and len(deps) == 1:
            b = bounds[deps[0]].clamp(str(p["min"]), str(p["max"]))
        elif op == "mod_residue" and len(deps) == 1:
            m = int(p["modulus"])
            if isinstance(p["modulus"], bool) or m < 2:
                raise ComputationDAGError("mod_residue modulus must be an integer >= 2.")
            u_b = bounds[deps[0]]
            if u_b.lo is not None and u_b.hi is not None and u_b.lo == u_b.hi:
                rem = u_b.lo % m
                b = IntervalBound(rem, rem)
            elif u_b.lo is not None and u_b.hi is not None and u_b.lo >= 0 and u_b.hi < m:
                b = u_b
            else:
                # Over Q, x - m * floor(x / m) lies in [0, m), whose closed rational hull is [0, m].
                b = IntervalBound(Fraction(0), Fraction(m))
        else:
            raise ComputationDAGError(f"Unsupported abstract-interpretation operation {op!r} on node {nid!r}.")

        bounds[nid] = b
        node_disp = TruthDisposition.TRUE
        if div_hazard is not None:
            node_disp = node_disp & div_hazard
        if nid in contracts:
            node_disp = node_disp & b.check_within(contracts[nid])
        dispositions[nid] = node_disp
        overall = overall & node_disp

    digest = domain_digest(
        "vld.interval-gate-report.v1",
        {
            "bounds": {k: v.to_dict() for k, v in sorted(bounds.items())},
            "dispositions": {k: v.value for k, v in sorted(dispositions.items())},
            "overall": overall.value,
        },
    )
    return IntervalGateReport(
        bounds=bounds,
        dispositions=dispositions,
        overall_disposition=overall,
        report_digest=digest,
    )


@dataclass(frozen=True, slots=True)
class BisimulationReport:
    """Result of coarsest strong bisimulation check between two labeled transition / computation graphs."""

    equivalent: bool
    block_count: int
    distinguishing_trace: tuple[str, ...] | None
    certificate_digest: str


def check_strong_bisimulation(
    *,
    actions: Sequence[str],
    left_states: Sequence[str],
    left_transitions: Mapping[tuple[str, str], Sequence[str]],
    left_observations: Mapping[str, Any],
    left_initial: str,
    right_states: Sequence[str],
    right_transitions: Mapping[tuple[str, str], Sequence[str]],
    right_observations: Mapping[str, Any],
    right_initial: str,
) -> BisimulationReport:
    r"""Check strong bisimilarity $s_1 \sim_{\text{bisim}} s_2$ over disjoint union $\Sigma_1 \sqcup \Sigma_2$."""
    act_tuple = tuple(sorted(set(actions)))
    left_set = set(left_states)
    right_set = set(right_states)
    if left_initial not in left_set or right_initial not in right_set:
        raise ValueError("Initial state missing from declared state set.")

    tagged_states: list[tuple[int, str]] = [
        *((0, s) for s in left_states),
        *((1, s) for s in right_states),
    ]
    obs_digest_map: dict[tuple[int, str], str] = {}
    trans_map: dict[tuple[tuple[int, str], str], tuple[tuple[int, str], ...]] = {}

    for s in left_states:
        if s not in left_observations:
            raise ValueError(f"Missing observation for left state {s!r}.")
        obs_digest_map[(0, s)] = domain_digest("vld.bisim-obs.v1", left_observations[s])
        for a in act_tuple:
            raw_targets = left_transitions.get((s, a), ())
            for t in raw_targets:
                if t not in left_set:
                    raise ValueError(f"Left transition {(s, a)!r} targets undeclared state {t!r}.")
            trans_map[((0, s), a)] = tuple(sorted({(0, t) for t in raw_targets}))

    for s in right_states:
        if s not in right_observations:
            raise ValueError(f"Missing observation for right state {s!r}.")
        obs_digest_map[(1, s)] = domain_digest("vld.bisim-obs.v1", right_observations[s])
        for a in act_tuple:
            raw_targets = right_transitions.get((s, a), ())
            for t in raw_targets:
                if t not in right_set:
                    raise ValueError(f"Right transition {(s, a)!r} targets undeclared state {t!r}.")
            trans_map[((1, s), a)] = tuple(sorted({(1, t) for t in raw_targets}))

    # Initial partition by canonical observation digest
    obs_groups: dict[str, int] = {}
    block_of: dict[tuple[int, str], int] = {}
    for st in tagged_states:
        od = obs_digest_map[st]
        if od not in obs_groups:
            obs_groups[od] = len(obs_groups)
        block_of[st] = obs_groups[od]

    # Iterative relational partition refinement until fixed point
    while True:
        sig_to_block: dict[tuple[Any, ...], int] = {}
        next_block: dict[tuple[int, str], int] = {}
        for st in tagged_states:
            sig = (
                block_of[st],
                tuple(tuple(sorted({block_of[t] for t in trans_map[(st, a)]})) for a in act_tuple),
            )
            if sig not in sig_to_block:
                sig_to_block[sig] = len(sig_to_block)
            next_block[st] = sig_to_block[sig]
        if len(sig_to_block) == len(set(block_of.values())):
            block_of = next_block
            break
        block_of = next_block

    l_root = (0, left_initial)
    r_root = (1, right_initial)
    eq = block_of[l_root] == block_of[r_root]

    trace: tuple[str, ...] | None = None
    if not eq:
        if obs_digest_map[l_root] != obs_digest_map[r_root]:
            trace = ()
        else:
            visited: set[tuple[tuple[int, str], tuple[int, str]]] = {(l_root, r_root)}
            q: deque[tuple[tuple[int, str], tuple[int, str], tuple[str, ...]]] = deque([(l_root, r_root, ())])
            while q and trace is None:
                p_st, q_st, path = q.popleft()
                for a in act_tuple:
                    p_succ = trans_map[(p_st, a)]
                    q_succ = trans_map[(q_st, a)]
                    p_blocks = {block_of[x] for x in p_succ}
                    q_blocks = {block_of[y] for y in q_succ}
                    if p_blocks != q_blocks:
                        if (p_blocks - q_blocks) and p_succ:
                            unmatched_p = next(x for x in p_succ if block_of[x] not in q_blocks)
                            if not q_succ or all(obs_digest_map[unmatched_p] != obs_digest_map[cq] for cq in q_succ):
                                trace = (*path, a)
                                break
                            for cand_q in q_succ:
                                if (unmatched_p, cand_q) not in visited:
                                    visited.add((unmatched_p, cand_q))
                                    q.append((unmatched_p, cand_q, (*path, a)))
                        elif (q_blocks - p_blocks) and q_succ:
                            unmatched_q = next(y for y in q_succ if block_of[y] not in p_blocks)
                            if not p_succ or all(obs_digest_map[cp] != obs_digest_map[unmatched_q] for cp in p_succ):
                                trace = (*path, a)
                                break
                            for cand_p in p_succ:
                                if (cand_p, unmatched_q) not in visited:
                                    visited.add((cand_p, unmatched_q))
                                    q.append((cand_p, unmatched_q, (*path, a)))
                    else:
                        for x in p_succ:
                            for y in q_succ:
                                if block_of[x] != block_of[y] and (x, y) not in visited:
                                    visited.add((x, y))
                                    q.append((x, y, (*path, a)))
            if trace is None:
                trace = ("<branching-divergence>",)

    cert_digest = domain_digest(
        "vld.bisim-certificate.v1",
        {
            "block_count": len(set(block_of.values())),
            "distinguishing_trace": None if trace is None else list(trace),
            "equivalent": eq,
            "left_initial": left_initial,
            "right_initial": right_initial,
        },
    )
    return BisimulationReport(
        equivalent=eq,
        block_count=len(set(block_of.values())),
        distinguishing_trace=trace,
        certificate_digest=cert_digest,
    )


__all__ = [
    "BisimulationReport",
    "IntervalBound",
    "IntervalGateReport",
    "analyze_dag_intervals",
    "check_strong_bisimulation",
]
