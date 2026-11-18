r"""Phase 10 — Intention-to-Realization Compiler, Proof-Producing E-Graph & Effect Semantics (`vld.compiler`).

Mathematical & architectural contract:
Given a declarative **Intention** $\mathcal{I} = (e_{\text{src}}, \mathcal{O}, \mathcal{A}, \mathcal{E}_{\text{eff}})$
specifying a desired observable relation $e_{\text{src}}$, observation & tolerance contract $\mathcal{O}$,
authority policy $\mathcal{A}$, and external effect discipline $\mathcal{E}_{\text{eff}}$:

1. **Proof-Producing Equivalence Saturation (`ProofProducingEGraph`):**
   Populates congruence-closed equivalence classes $[e]_{\equiv}$ under admitted algebraic & structural laws
   (`affine_fusion`, `affine_iter_collapse`, `affine_chain_collapse`, `vm_pullback_reduction`,
   `online_flash_partition`, `internet_range_slice`), recording constructive bidirectional proof chains
   $\pi = (s_0, \dots, s_{L-1})$.

2. **Independent Rewrite Proof Verification (`verify_rewrite_proof_chain`):**
   Optimizer output is **never** trusted as its own proof; every candidate expression extracted from $[e_{\text{src}}]$
   must pass `verify_rewrite_proof_chain_standalone` step-by-step prior to admission into $\mathcal{R}(\mathcal{G}, \mathcal{O})$.

3. **Pareto Realization Frontier & Interchangeable Execution:**
   $$r^* \in \underset{r \in \mathcal{R}(\mathcal{G}, \mathcal{O})}{\operatorname{ParetoMin}}\bigl(\operatorname{Cost}(r)\bigr)$$
   Synthesizes interchangeable realizers across:
   - `direct_algebraic_navigation`: $O(\log n)$ / $O(|\Delta|)$ closed-form orbit or contravariant pullback,
   - `fluid_in_transit_stream`: `InTransitPathMachine` / `OnlineFlashStreamState` stream-interpreter hops,
   - `qualified_internet_path`: HTTP `206` Range / conditional ETag path projection (`\Delta N_{\text{origin}}` tracked),
   - `hosted_continuous_execution`: Quorum-fenced `ContinuousVirtualSystem` (`ed25519-rfc8032` or `hmac-sha256`).

4. **External Effect Semantics (`EffectLedger`):**
   Separates `PURE`, `IDEMPOTENT`, `TRANSACTIONAL`, and `COMPENSATABLE` external actions so retries and crash
   recoveries never silently duplicate accepted side effects, and aborted multi-step workflows run exact
   reverse-order compensators.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import math
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from vld.authority import (
    QuorumAuthorityMember,
    QuorumLineageCoordinator,
    QuorumMembership,
    build_ed25519_quorum_identities,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.continuous import ContinuousStageSpec, ContinuousVirtualSystem
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.independent_audit import (
    IndependentAuditError,
    verify_continuity_receipt_chain_standalone,
    verify_rewrite_proof_chain_standalone,
    verify_transit_packet_receipts_standalone,
)
from vld.journal import HashChainedJournal
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.substrates.fluid_transit import (
    InTransitPathMachine,
    OnlineFlashStreamState,
    StreamInterpreterNode,
    TransitOperation,
    VirtualMemoryMovementView,
)


class IntentionCompilerError(RuntimeError):
    """Raised when an intention expression, rewrite proof, realization contract, or effect guard fails."""


class RewriteVerificationError(IntentionCompilerError):
    """Raised when an e-graph rewrite proof chain fails independent verification."""


class EffectSemantics(str, Enum):
    """Classification of externally visible side effects across retries and host crashes."""

    PURE = "pure"
    IDEMPOTENT = "idempotent"
    TRANSACTIONAL = "transactional"
    COMPENSATABLE = "compensatable"


@dataclass(frozen=True, slots=True)
class IntentExpr:
    """Immutable, canonically content-addressed symbolic computational expression."""

    op: str
    args: tuple[IntentExpr, ...] = ()
    params: Mapping[str, Any] = ()  # normalized in __post_init__

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op.strip():
            raise IntentionCompilerError("IntentExpr op must be a non-empty string.")
        if not isinstance(self.args, tuple) or any(not isinstance(a, IntentExpr) for a in self.args):
            object.__setattr__(self, "args", tuple(self.args))
            if any(not isinstance(a, IntentExpr) for a in self.args):
                raise IntentionCompilerError("IntentExpr args must be IntentExpr instances.")
        raw_p = dict(self.params) if isinstance(self.params, Mapping) else dict(self.params)
        norm_p = parse_strict_json(canonical_json(raw_p))
        object.__setattr__(self, "params", norm_p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "args": [a.to_dict() for a in self.args],
            "op": self.op,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> IntentExpr:
        return cls(
            op=str(data["op"]),
            args=tuple(cls.from_dict(a) for a in data.get("args", ())),
            params=dict(data.get("params", {})),
        )

    @property
    def expr_digest(self) -> str:
        return domain_digest("vld.intent-expr.v1", self.to_dict())

    # Ergonomic constructors
    @classmethod
    def var(cls, name: str, representation: str = "int_vector") -> IntentExpr:
        return cls("var", (), {"name": name, "representation": representation})

    @classmethod
    def affine_mod(cls, child: IntentExpr, *, a: int, b: int, modulus: int) -> IntentExpr:
        return cls("affine_mod", (child,), {"a": int(a), "b": int(b), "modulus": int(modulus)})

    @classmethod
    def affine_chain(cls, child: IntentExpr, *, chain: Sequence[tuple[int, int]], modulus: int) -> IntentExpr:
        return cls("affine_chain", (child,), {"chain": [[int(a), int(b)] for a, b in chain], "modulus": int(modulus)})

    @classmethod
    def iterate_affine(cls, child: IntentExpr, *, a: int, b: int, steps: int, modulus: int) -> IntentExpr:
        return cls("iterate_affine", (child,), {"a": int(a), "b": int(b), "modulus": int(modulus), "steps": int(steps)})

    @classmethod
    def closed_orbit_jump(cls, child: IntentExpr, *, a: int, b: int, modulus: int) -> IntentExpr:
        return cls("closed_orbit_jump", (child,), {"a": int(a), "b": int(b), "modulus": int(modulus)})

    @classmethod
    def buffer_permute_patch(
        cls,
        base: IntentExpr,
        *,
        kind: str,
        length: int,
        patches: Mapping[str, int] | Mapping[int, int] | None = None,
        rows: int = 0,
        cols: int = 0,
        shift: int = 0,
    ) -> IntentExpr:
        p_str = {str(int(k)): int(v) for k, v in (patches or {}).items()}
        return cls(
            "buffer_permute_patch",
            (base,),
            {
                "cols": int(cols),
                "kind": str(kind),
                "length": int(length),
                "patches": p_str,
                "rows": int(rows),
                "shift": int(shift),
            },
        )

    @classmethod
    def buffer_reduce(cls, child: IntentExpr, *, monoid: str = "sum") -> IntentExpr:
        return cls("buffer_reduce", (child,), {"monoid": str(monoid)})

    @classmethod
    def partitioned_softmax_attn(
        cls,
        queries_var: IntentExpr,
        shards_var: IntentExpr,
        *,
        num_shards: int,
        causal: bool = True,
    ) -> IntentExpr:
        return cls(
            "partitioned_softmax_attn",
            (queries_var, shards_var),
            {"causal": bool(causal), "num_shards": int(num_shards)},
        )

    @classmethod
    def http_slice(cls, url: str, *, offset: int, length: int) -> IntentExpr:
        return cls("http_slice", (), {"length": int(length), "offset": int(offset), "url": str(url)})


@dataclass(frozen=True, slots=True)
class ObservationContract:
    """Acceptance & accuracy contract for an admitted observation."""

    output_representation: str
    numerical_tolerance: Fraction = Fraction(0)
    require_independent_audit: bool = True

    def __post_init__(self) -> None:
        if not self.output_representation:
            raise IntentionCompilerError("output_representation must be non-empty.")
        tol = Fraction(self.numerical_tolerance)
        if tol < 0:
            raise IntentionCompilerError("numerical_tolerance must be non-negative.")
        object.__setattr__(self, "numerical_tolerance", tol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerical_tolerance": str(self.numerical_tolerance),
            "output_representation": self.output_representation,
            "require_independent_audit": self.require_independent_audit,
        }


@dataclass(frozen=True, slots=True)
class AuthorityPolicy:
    """Quorum & signature provenance policy required to accept a realized observation."""

    lineage_id: str
    member_ids: tuple[str, ...] = ("q0", "q1", "q2")
    require_public_key_quorum: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "member_ids": list(self.member_ids),
            "require_public_key_quorum": self.require_public_key_quorum,
        }


@dataclass(frozen=True, slots=True)
class IntentionSpec:
    """Complete typed Intention binding desired mathematical relation, observation, authority & effects."""

    intention_id: str
    expression: IntentExpr
    observation: ObservationContract
    authority: AuthorityPolicy
    effect_semantics: EffectSemantics = EffectSemantics.PURE
    cost_ceilings: NavigationCostVector | None = None

    def __post_init__(self) -> None:
        if not self.intention_id or not self.intention_id.strip():
            raise IntentionCompilerError("intention_id must be non-empty.")

    @property
    def intention_fingerprint(self) -> str:
        return domain_digest(
            "vld.intention-spec.v1",
            {
                "authority": self.authority.to_dict(),
                "cost_ceilings": None if self.cost_ceilings is None else self.cost_ceilings.to_dict(),
                "effect_semantics": self.effect_semantics.value,
                "expression": self.expression.to_dict(),
                "intention_id": self.intention_id,
                "observation": self.observation.to_dict(),
            },
        )


# ============================================================================
# 1. Proof-Producing E-Graph & Independent Rewrite Checker
# ============================================================================


@dataclass(frozen=True, slots=True)
class RewriteProofStep:
    """Single constructive algebraic rewrite witness along an equivalence proof path."""

    step_index: int
    rule_id: str
    lhs_expr: Mapping[str, Any]
    rhs_expr: Mapping[str, Any]
    step_digest: str = ""

    def __post_init__(self) -> None:
        computed = domain_digest(
            "vld.rewrite-proof-step.v1",
            {
                "lhs_expr": self.lhs_expr,
                "rhs_expr": self.rhs_expr,
                "rule_id": self.rule_id,
                "step_index": self.step_index,
            },
        )
        if self.step_digest and self.step_digest != computed:
            raise RewriteVerificationError("RewriteProofStep digest mismatch.")
        object.__setattr__(self, "step_digest", computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lhs_expr": dict(self.lhs_expr),
            "rhs_expr": dict(self.rhs_expr),
            "rule_id": self.rule_id,
            "step_digest": self.step_digest,
            "step_index": self.step_index,
        }


def _affine_power_mod(a: int, b: int, steps: int, modulus: int) -> tuple[int, int]:
    r"""Exact $O(\log n)$ affine monoid exponentiation $(a, b)^n \bmod m$ for arbitrary $\gcd(a-1, m)$."""
    res_a, res_b = 1 % modulus, 0
    cur_a, cur_b = a % modulus, b % modulus
    k = steps
    while k > 0:
        if k & 1:
            res_a, res_b = (cur_a * res_a) % modulus, (cur_a * res_b + cur_b) % modulus
        cur_a, cur_b = (cur_a * cur_a) % modulus, (cur_a * cur_b + cur_b) % modulus
        k >>= 1
    return res_a, res_b


def verify_rewrite_proof_chain(
    source: IntentExpr,
    target: IntentExpr,
    steps: Sequence[RewriteProofStep],
) -> dict[str, Any]:
    """Verify an e-graph rewrite proof using the zero-import standalone proof checker."""
    try:
        return verify_rewrite_proof_chain_standalone(
            source.to_dict(),
            target.to_dict(),
            [s.to_dict() for s in steps],
        )
    except IndependentAuditError as exc:
        raise RewriteVerificationError(f"Rewrite proof rejected by independent checker: {exc}") from exc


class ProofProducingEGraph:
    """Equality-saturation graph maintaining union-find e-classes and constructive proof edges."""

    def __init__(self) -> None:
        self._expr_by_digest: dict[str, IntentExpr] = {}
        self._parent: dict[str, str] = {}
        self._proof_adj: dict[str, list[tuple[str, str]]] = {}

    def _find(self, dg: str) -> str:
        root = dg
        while self._parent[root] != root:
            root = self._parent[root]
        cur = dg
        while self._parent[cur] != root:
            nxt = self._parent[cur]
            self._parent[cur] = root
            cur = nxt
        return root

    def add_expr(self, expr: IntentExpr) -> str:
        for child in expr.args:
            self.add_expr(child)
        dg = expr.expr_digest
        if dg not in self._expr_by_digest:
            self._expr_by_digest[dg] = expr
            self._parent[dg] = dg
            self._proof_adj[dg] = []
        return dg

    def union_with_rule(self, lhs: IntentExpr, rhs: IntentExpr, rule_id: str) -> bool:
        d1 = self.add_expr(lhs)
        d2 = self.add_expr(rhs)
        if d1 == d2:
            return False
        # Record bidirectional constructive proof edge
        existing = {nbr for nbr, _ in self._proof_adj[d1]}
        changed = False
        if d2 not in existing:
            self._proof_adj[d1].append((d2, rule_id))
            self._proof_adj[d2].append((d1, rule_id))
            changed = True
        r1, r2 = self._find(d1), self._find(d2)
        if r1 != r2:
            self._parent[r2] = r1
            changed = True
        return changed

    def saturate(self, max_rounds: int = 8) -> int:
        """Apply admitted algebraic & congruence rewrites until fixpoint or `max_rounds`."""
        rewrites_added = 0
        for _ in range(max_rounds):
            round_delta = 0
            snapshot = list(self._expr_by_digest.values())
            for expr in snapshot:
                op = expr.op
                par = expr.params
                # 1. Iterative affine orbit -> O(log n) closed_orbit_jump
                if op == "iterate_affine":
                    m = int(par["modulus"])
                    n = int(par["steps"])
                    if m >= 1 and n >= 0:
                        an, bn = _affine_power_mod(int(par["a"]), int(par["b"]), n, m)
                        collapsed = IntentExpr.closed_orbit_jump(expr.args[0], a=an, b=bn, modulus=m)
                        if self.union_with_rule(expr, collapsed, "affine_iter_collapse"):
                            round_delta += 1

                # 2. Multi-step affine chain -> O(1) closed_orbit_jump
                elif op == "affine_chain":
                    m = int(par["modulus"])
                    pairs = list(par.get("chain", ()))
                    if m >= 1 and pairs:
                        acc_a, acc_b = 1 % m, 0
                        for pa, pb in pairs:
                            acc_a, acc_b = (int(pa) * acc_a) % m, (int(pa) * acc_b + int(pb)) % m
                        collapsed = IntentExpr.closed_orbit_jump(expr.args[0], a=acc_a, b=acc_b, modulus=m)
                        if self.union_with_rule(expr, collapsed, "affine_chain_collapse"):
                            round_delta += 1

                # 3. Nested affine_mod fusion: T_{a2,b2}(T_{a1,b1}(x)) -> closed_orbit_jump
                elif op == "affine_mod" and expr.args and expr.args[0].op in ("affine_mod", "closed_orbit_jump"):
                    inner = expr.args[0]
                    m = int(par["modulus"])
                    if m >= 1 and int(inner.params["modulus"]) == m:
                        a1, b1 = int(inner.params["a"]) % m, int(inner.params["b"]) % m
                        a2, b2 = int(par["a"]) % m, int(par["b"]) % m
                        fused = IntentExpr.closed_orbit_jump(
                            inner.args[0],
                            a=(a2 * a1) % m,
                            b=(a2 * b1 + b2) % m,
                            modulus=m,
                        )
                        if self.union_with_rule(expr, fused, "affine_fusion"):
                            round_delta += 1

                # 4. Contravariant VM pullback across bijective buffer permutations
                elif op == "buffer_reduce" and expr.args and expr.args[0].op == "buffer_permute_patch":
                    monoid = str(par.get("monoid", ""))
                    perm = expr.args[0]
                    kind = str(perm.params.get("kind", ""))
                    n_len = int(perm.params.get("length", 0))
                    bij = (kind in ("reverse", "rotate") and n_len > 0) or (
                        kind == "transpose_2d"
                        and n_len > 0
                        and int(perm.params.get("rows", 0)) * int(perm.params.get("cols", 0)) == n_len
                    )
                    if monoid in ("sum", "xor", "min", "max") and bij:
                        pulled = IntentExpr(
                            "contravariant_pullback_reduce",
                            (perm.args[0],),
                            {
                                "cols": int(perm.params.get("cols", 0)),
                                "kind": kind,
                                "length": n_len,
                                "monoid": monoid,
                                "patches": dict(perm.params.get("patches", {})),
                                "rows": int(perm.params.get("rows", 0)),
                                "shift": int(perm.params.get("shift", 0)),
                            },
                        )
                        if self.union_with_rule(expr, pulled, "vm_pullback_reduction"):
                            round_delta += 1

                # 5. Distributed Softmax Attention -> Associative Online Flash-Stream Monoid
                elif op == "partitioned_softmax_attn" and int(par.get("num_shards", 0)) >= 1:
                    streamed = IntentExpr("online_flash_stream_attn", expr.args, dict(par))
                    if self.union_with_rule(expr, streamed, "online_flash_partition"):
                        round_delta += 1

                # 6. HTTP Full-Fetch Slice -> Qualified HTTP 206 Range Slice
                elif op == "http_slice":
                    off = int(par.get("offset", -1))
                    ln = int(par.get("length", 0))
                    url = str(par.get("url", ""))
                    if off >= 0 and ln > 0 and url:
                        ranged = IntentExpr("http_range_206", (), {"length": ln, "offset": off, "url": url})
                        if self.union_with_rule(expr, ranged, "internet_range_slice"):
                            round_delta += 1

                # 7. Congruence propagation on single-child rewrites
                for idx, child in enumerate(expr.args):
                    c_dg = child.expr_digest
                    for nbr_dg, sub_rule in list(self._proof_adj.get(c_dg, ())):
                        if ":" in sub_rule:
                            continue
                        nbr_expr = self._expr_by_digest[nbr_dg]
                        new_args = list(expr.args)
                        new_args[idx] = nbr_expr
                        lifted = IntentExpr(expr.op, tuple(new_args), dict(expr.params))
                        if self.union_with_rule(expr, lifted, f"congruence:{sub_rule}"):
                            round_delta += 1

            rewrites_added += round_delta
            if round_delta == 0:
                break
        return rewrites_added

    def equivalent_expressions(self, root_expr: IntentExpr) -> tuple[IntentExpr, ...]:
        root_dg = self.add_expr(root_expr)
        cls_root = self._find(root_dg)
        members = [
            ex for dg, ex in sorted(self._expr_by_digest.items()) if self._find(dg) == cls_root
        ]
        return tuple(members)

    def extract_proof_chain(self, source: IntentExpr, target: IntentExpr) -> tuple[RewriteProofStep, ...]:
        """Extract shortest constructive `RewriteProofStep` path via BFS over the proof graph."""
        s_dg = self.add_expr(source)
        t_dg = self.add_expr(target)
        if s_dg == t_dg:
            return ()
        queue: deque[str] = deque([s_dg])
        prev_map: dict[str, tuple[str, str]] = {}
        visited = {s_dg}
        while queue:
            cur = queue.popleft()
            if cur == t_dg:
                break
            for nbr, rule in self._proof_adj.get(cur, ()):
                if nbr not in visited:
                    visited.add(nbr)
                    prev_map[nbr] = (cur, rule)
                    queue.append(nbr)
        if t_dg not in prev_map:
            raise RewriteVerificationError("Expressions are not connected in ProofProducingEGraph.")
        raw_edges: list[tuple[str, str, str]] = []
        walk = t_dg
        while walk != s_dg:
            p_dg, rule = prev_map[walk]
            raw_edges.append((p_dg, walk, rule))
            walk = p_dg
        raw_edges.reverse()
        steps = tuple(
            RewriteProofStep(
                step_index=i,
                rule_id=rule,
                lhs_expr=self._expr_by_digest[lhs_dg].to_dict(),
                rhs_expr=self._expr_by_digest[rhs_dg].to_dict(),
            )
            for i, (lhs_dg, rhs_dg, rule) in enumerate(raw_edges)
        )
        verify_rewrite_proof_chain(source, target, steps)
        return steps


# ============================================================================
# 2. External Effect Semantics Ledger (Crash/Retry Non-Duplication & Rollback)
# ============================================================================


@dataclass(frozen=True, slots=True)
class EffectExecutionReceipt:
    """Journal-anchored record proving non-duplicated or compensated external effect semantics."""

    intention_id: str
    effect_key: str
    semantics: EffectSemantics
    replayed_from_ledger: bool
    result: Any
    journal_digest: str


class EffectLedger:
    """Enforces idempotent deduplication, transactional commit, and reverse-order compensation."""

    def __init__(self, journal: HashChainedJournal | None = None) -> None:
        self.journal = journal if journal is not None else HashChainedJournal()
        self._committed: dict[tuple[str, str], tuple[Any, str]] = {}
        self._sync()

    def _sync(self) -> None:
        self.journal.verify()
        for ev in self.journal.events:
            if ev.operation == "effect.commit":
                key = (str(ev.payload["intention_id"]), str(ev.payload["effect_key"]))
                self._committed[key] = (ev.result["value"], ev.event_digest)
            elif ev.operation == "effect.compensate":
                key = (str(ev.payload["intention_id"]), str(ev.payload["effect_key"]))
                self._committed.pop(key, None)

    def execute_effect(
        self,
        *,
        intention_id: str,
        effect_key: str,
        semantics: EffectSemantics,
        action_fn: Callable[[], Any],
    ) -> EffectExecutionReceipt:
        if semantics == EffectSemantics.PURE:
            val = action_fn()
            return EffectExecutionReceipt(
                intention_id=intention_id,
                effect_key=effect_key,
                semantics=semantics,
                replayed_from_ledger=False,
                result=val,
                journal_digest="pure-untracked",
            )
        with self.journal.exclusive_guard():
            self._sync()
            k = (intention_id, effect_key)
            if k in self._committed:
                cached_val, j_dg = self._committed[k]
                return EffectExecutionReceipt(
                    intention_id=intention_id,
                    effect_key=effect_key,
                    semantics=semantics,
                    replayed_from_ledger=True,
                    result=cached_val,
                    journal_digest=j_dg,
                )
            val = action_fn()
            rec, _ = self.journal.record(
                request_id=f"eff.commit:{intention_id}:{effect_key}:{len(self.journal.events)}",
                ground_fingerprint=domain_digest("vld.effect-scope.v1", {"effect_key": effect_key, "intention_id": intention_id}),
                operation="effect.commit",
                payload={"effect_key": effect_key, "intention_id": intention_id, "semantics": semantics.value},
                result_fn=lambda: {"value": val},
            )
            self._committed[k] = (val, rec.event_digest)
            return EffectExecutionReceipt(
                intention_id=intention_id,
                effect_key=effect_key,
                semantics=semantics,
                replayed_from_ledger=False,
                result=val,
                journal_digest=rec.event_digest,
            )

    def execute_compensatable_workflow(
        self,
        *,
        intention_id: str,
        steps: Sequence[tuple[str, Callable[[], Any], Callable[[Any], None]]],
    ) -> list[EffectExecutionReceipt]:
        """Run compensatable actions sequentially; on failure, invoke reverse-order compensators."""
        completed: list[tuple[str, Any, Callable[[Any], None]]] = []
        receipts: list[EffectExecutionReceipt] = []
        for eff_key, act_fn, comp_fn in steps:
            try:
                rcpt = self.execute_effect(
                    intention_id=intention_id,
                    effect_key=eff_key,
                    semantics=EffectSemantics.COMPENSATABLE,
                    action_fn=act_fn,
                )
                completed.append((eff_key, rcpt.result, comp_fn))
                receipts.append(rcpt)
            except Exception as exc:
                for rev_key, rev_val, rev_comp in reversed(completed):
                    rev_comp(rev_val)
                    with self.journal.exclusive_guard():
                        self.journal.record(
                            request_id=f"eff.compensate:{intention_id}:{rev_key}:{len(self.journal.events)}",
                            ground_fingerprint=domain_digest(
                                "vld.effect-scope.v1", {"effect_key": rev_key, "intention_id": intention_id}
                            ),
                            operation="effect.compensate",
                            payload={"effect_key": rev_key, "intention_id": intention_id, "reason": str(exc)},
                            result_fn=lambda: {"compensated": True},
                        )
                        self._committed.pop((intention_id, rev_key), None)
                raise
        return receipts


# ============================================================================
# 3. Intention-to-Realization Compiler & Multi-Backend Realizer
# ============================================================================


@dataclass(frozen=True, slots=True)
class CandidateRealization:
    r"""Verified candidate realization $r \in \mathcal{R}(\mathcal{G}, \mathcal{O})$ with proof & cost."""

    realization_id: str
    backend_kind: str
    target_expression: IntentExpr
    rewrite_proof: tuple[RewriteProofStep, ...]
    estimated_cost: NavigationCostVector
    ground_spec: VirtualGroundSpec
    pareto_optimal: bool


@dataclass(frozen=True, slots=True)
class CompiledRealizationPlan:
    """Result of compiling an `IntentionSpec` into Pareto-ranked meaning-preserving realizations."""

    intention: IntentionSpec
    candidates: tuple[CandidateRealization, ...]
    pareto_frontier_ids: tuple[str, ...]
    selected_realization_id: str

    def get_candidate(self, realization_id: str | None = None) -> CandidateRealization:
        rid = realization_id or self.selected_realization_id
        for c in self.candidates:
            if c.realization_id == rid:
                return c
        raise IntentionCompilerError(f"Unknown candidate realization_id {rid!r}.")


@dataclass(frozen=True, slots=True)
class RealizedObservationReceipt:
    """Independently verified observation produced by executing a `CandidateRealization`."""

    intention_id: str
    intention_fingerprint: str
    realization_id: str
    backend_kind: str
    admitted_observation: Any
    measured_cost: NavigationCostVector
    rewrite_audit: Mapping[str, Any]
    authority_audit: Mapping[str, Any]
    quorum_public_keys: Mapping[str, str] | None
    continuity_receipts: tuple[Mapping[str, Any], ...]
    transit_packet_dict: Mapping[str, Any] | None = None


def _eval_expr_direct(expr: IntentExpr, env: Mapping[str, Any]) -> Any:
    """Reference / algebraic evaluation of an `IntentExpr`."""
    op = expr.op
    par = expr.params
    if op == "var":
        name = str(par["name"])
        if name not in env:
            raise IntentionCompilerError(f"Missing runtime binding for variable {name!r}.")
        return env[name]

    if op in ("affine_mod", "closed_orbit_jump"):
        val = _eval_expr_direct(expr.args[0], env)
        a, b, m = int(par["a"]), int(par["b"]), int(par["modulus"])
        if isinstance(val, int):
            return (a * val + b) % m
        return [(a * int(x) + b) % m for x in val]

    if op == "affine_chain":
        cur = _eval_expr_direct(expr.args[0], env)
        m = int(par["modulus"])
        for a, b in par["chain"]:
            ia, ib = int(a), int(b)
            cur = ((ia * cur + ib) % m) if isinstance(cur, int) else [(ia * int(x) + ib) % m for x in cur]
        return cur

    if op == "iterate_affine":
        cur = _eval_expr_direct(expr.args[0], env)
        a, b, m, steps = int(par["a"]), int(par["b"]), int(par["modulus"]), int(par["steps"])
        # Note: in hosted step-by-step control for very large horizons, cap naive loop or step faithfully
        if steps <= 200_000:
            if isinstance(cur, int):
                v = cur % m
                for _ in range(steps):
                    v = (a * v + b) % m
                return v
            vec = [int(x) % m for x in cur]
            for _ in range(steps):
                vec = [(a * x + b) % m for x in vec]
            return vec
        an, bn = _affine_power_mod(a, b, steps, m)
        return ((an * cur + bn) % m) if isinstance(cur, int) else [(an * int(x) + bn) % m for x in cur]

    if op == "buffer_permute_patch":
        base_raw = bytes(_eval_expr_direct(expr.args[0], env))
        view = VirtualMemoryMovementView.from_buffer("intent.buf", len(base_raw))
        kind = str(par["kind"])
        if kind == "reverse":
            view = view.reverse_view()
        elif kind == "rotate":
            view = view.rotate_view(int(par.get("shift", 0)))
        elif kind == "transpose_2d":
            view = view.transpose_2d(int(par["rows"]), int(par["cols"]))
        for k_str, b_val in sorted(dict(par.get("patches", {})).items(), key=lambda kv: int(kv[0])):
            view = view.cow_write(int(k_str), int(b_val))
        return view.materialize(base_raw)

    if op == "buffer_reduce":
        buf = bytes(_eval_expr_direct(expr.args[0], env))
        monoid = str(par["monoid"])
        if monoid == "sum":
            return sum(buf)
        if monoid == "xor":
            acc = 0
            for b in buf:
                acc ^= b
            return acc
        if monoid == "min":
            return min(buf)
        if monoid == "max":
            return max(buf)
        raise IntentionCompilerError(f"Unsupported buffer_reduce monoid {monoid!r}.")

    if op == "contravariant_pullback_reduce":
        base_raw = bytes(_eval_expr_direct(expr.args[0], env))
        view = VirtualMemoryMovementView.from_buffer("intent.buf", len(base_raw))
        kind = str(par["kind"])
        if kind == "reverse":
            view = view.reverse_view()
        elif kind == "rotate":
            view = view.rotate_view(int(par.get("shift", 0)))
        elif kind == "transpose_2d":
            view = view.transpose_2d(int(par["rows"]), int(par["cols"]))
        for k_str, b_val in sorted(dict(par.get("patches", {})).items(), key=lambda kv: int(kv[0])):
            view = view.cow_write(int(k_str), int(b_val))
        monoid = str(par["monoid"])
        if monoid in ("sum", "xor"):
            return view.commute_invariant_reduction(base_raw, monoid)
        mat = view.materialize(base_raw)
        return min(mat) if monoid == "min" else max(mat)

    if op == "partitioned_softmax_attn":
        queries = _eval_expr_direct(expr.args[0], env)
        shards = _eval_expr_direct(expr.args[1], env)
        causal = bool(par.get("causal", True))
        all_k = [k for sh in shards for k in sh["keys"]]
        all_v = [v for sh in shards for v in sh["values"]]
        st = OnlineFlashStreamState.from_kv_shard(queries, all_k, all_v, causal=causal)
        return [[round(float(x), 9) for x in row] for row in st.finalize()]

    if op == "online_flash_stream_attn":
        queries = _eval_expr_direct(expr.args[0], env)
        shards = _eval_expr_direct(expr.args[1], env)
        causal = bool(par.get("causal", True))
        dv = len(shards[0]["values"][0])
        acc = OnlineFlashStreamState.identity(len(queries), dv)
        k_offset = 0
        for sh in shards:
            part = OnlineFlashStreamState.from_kv_shard(
                queries, sh["keys"], sh["values"], key_start_pos=k_offset, causal=causal
            )
            acc = acc.combine(part)
            k_offset += len(sh["keys"])
        return [[round(float(x), 9) for x in row] for row in acc.finalize()]

    if op == "http_slice":
        url = str(par["url"])
        off = int(par["offset"])
        ln = int(par["length"])
        with urlopen(Request(url, method="GET"), timeout=10.0) as resp:
            full = resp.read()
        return list(full[off : off + ln])

    if op == "http_range_206":
        url = str(par["url"])
        off = int(par["offset"])
        ln = int(par["length"])
        req = Request(url, headers={"Range": f"bytes={off}-{off + ln - 1}"}, method="GET")
        with urlopen(req, timeout=10.0) as resp:
            code = getattr(resp, "status", 200)
            body = resp.read()
        slice_bytes = body if code == 206 else body[off : off + ln]
        return list(slice_bytes[:ln])

    raise IntentionCompilerError(f"Unsupported IntentExpr op {op!r}.")


def _estimate_expr_cost(expr: IntentExpr, backend_kind: str) -> NavigationCostVector:
    op = expr.op
    par = expr.params
    host_overhead_ms = 0.25 if backend_kind == "hosted_continuous_execution" else 0.0
    host_overhead_ops = 4.0 if backend_kind == "hosted_continuous_execution" else 0.0
    if op == "closed_orbit_jump":
        if backend_kind == "direct_algebraic_navigation":
            return NavigationCostVector(latency_ms=0.04, compute_ops=6.0, bytes_moved=32.0, verification_ops=2.0)
        if backend_kind == "fluid_in_transit_stream":
            return NavigationCostVector(latency_ms=0.12, compute_ops=8.0, bytes_moved=96.0, verification_ops=3.0)
        return NavigationCostVector(latency_ms=0.05 + host_overhead_ms, compute_ops=8.0 + host_overhead_ops, bytes_moved=64.0, verification_ops=2.0)
    if op == "iterate_affine":
        n = max(1, int(par.get("steps", 1)))
        return NavigationCostVector(
            latency_ms=min(500.0, 0.01 * n) + host_overhead_ms,
            compute_ops=float(n) + host_overhead_ops,
            bytes_moved=256.0,
            verification_ops=2.0,
        )
    if op in ("affine_chain", "affine_mod"):
        k = len(par.get("chain", (1,))) if op == "affine_chain" else 2
        if backend_kind == "fluid_in_transit_stream":
            return NavigationCostVector(latency_ms=0.2, compute_ops=float(k), bytes_moved=128.0, verification_ops=float(k))
        return NavigationCostVector(latency_ms=0.4 + host_overhead_ms, compute_ops=float(2 * k) + host_overhead_ops, bytes_moved=256.0, verification_ops=2.0)
    if op == "contravariant_pullback_reduce":
        p_cnt = len(dict(par.get("patches", {})))
        if backend_kind == "direct_algebraic_navigation":
            return NavigationCostVector(
                latency_ms=0.06,
                compute_ops=float(4 + p_cnt),
                bytes_moved=float(48 + 16 * p_cnt),
                verification_ops=2.0,
            )
        return NavigationCostVector(
            latency_ms=0.12 + host_overhead_ms,
            compute_ops=float(6 + p_cnt) + host_overhead_ops,
            bytes_moved=float(64 + 16 * p_cnt),
            verification_ops=2.0,
        )
    if op == "buffer_reduce":
        ln = int(expr.args[0].params.get("length", 4096)) if expr.args else 4096
        return NavigationCostVector(latency_ms=1.0 + host_overhead_ms, compute_ops=float(2 * ln), bytes_moved=float(ln), verification_ops=2.0)
    if op == "online_flash_stream_attn":
        sh = max(1, int(par.get("num_shards", 1)))
        if backend_kind == "fluid_in_transit_stream":
            return NavigationCostVector(latency_ms=0.25 * sh, compute_ops=float(28 * sh), bytes_moved=float(96 * sh), verification_ops=float(sh))
        return NavigationCostVector(latency_ms=0.4 * sh + host_overhead_ms, compute_ops=float(36 * sh), bytes_moved=float(128 * sh), verification_ops=2.0)
    if op == "partitioned_softmax_attn":
        sh = max(1, int(par.get("num_shards", 1)))
        return NavigationCostVector(latency_ms=1.2 * sh + host_overhead_ms, compute_ops=float(128 * sh), bytes_moved=float(2048 * sh), verification_ops=2.0)
    if op == "http_range_206":
        ln = int(par.get("length", 64))
        if backend_kind == "qualified_internet_path":
            return NavigationCostVector(latency_ms=0.8, compute_ops=2.0, bytes_moved=float(ln), verification_ops=2.0)
        return NavigationCostVector(latency_ms=1.5 + host_overhead_ms, compute_ops=4.0 + host_overhead_ops, bytes_moved=float(ln), verification_ops=2.0)
    if op == "http_slice":
        ln = int(par.get("length", 64))
        return NavigationCostVector(latency_ms=5.0 + host_overhead_ms, compute_ops=16.0 + host_overhead_ops, bytes_moved=float(max(4096, ln * 32)), verification_ops=2.0)
    return NavigationCostVector(latency_ms=1.0 + host_overhead_ms, compute_ops=16.0 + host_overhead_ops, bytes_moved=256.0, verification_ops=2.0)


def _supports_fluid_stream(expr: IntentExpr) -> bool:
    return expr.op in (
        "closed_orbit_jump",
        "affine_mod",
        "affine_chain",
        "iterate_affine",
        "contravariant_pullback_reduce",
        "online_flash_stream_attn",
    )


class IntentionToRealizationCompiler:
    """Compiles a declarative `IntentionSpec` into Pareto-optimal verified realizations across VLD substrates."""

    def __init__(self, *, effect_ledger: EffectLedger | None = None) -> None:
        self.effect_ledger = effect_ledger if effect_ledger is not None else EffectLedger()

    def compile(self, intention: IntentionSpec) -> CompiledRealizationPlan:
        egraph = ProofProducingEGraph()
        egraph.add_expr(intention.expression)
        egraph.saturate(max_rounds=8)
        equiv_exprs = egraph.equivalent_expressions(intention.expression)

        raw_candidates: list[tuple[str, str, IntentExpr, tuple[RewriteProofStep, ...], NavigationCostVector, VirtualGroundSpec]] = []
        for idx, cand_expr in enumerate(equiv_exprs):
            proof = egraph.extract_proof_chain(intention.expression, cand_expr)
            verify_rewrite_proof_chain(intention.expression, cand_expr, proof)

            backends: list[str] = ["hosted_continuous_execution"]
            if cand_expr.op in ("closed_orbit_jump", "contravariant_pullback_reduce"):
                backends.append("direct_algebraic_navigation")
            if _supports_fluid_stream(cand_expr):
                backends.append("fluid_in_transit_stream")
            if cand_expr.op in ("http_range_206", "http_slice"):
                backends.append("qualified_internet_path")

            for b_kind in backends:
                est_cost = _estimate_expr_cost(cand_expr, b_kind)
                if intention.cost_ceilings is not None and not est_cost.within_ceilings(intention.cost_ceilings):
                    continue
                rid = f"{intention.intention_id}.{b_kind}.{cand_expr.op}.{idx}"
                gspec = VirtualGroundSpec(
                    ground_id=rid,
                    version="1.0",
                    language=cand_expr.op,
                    state_domain=intention.observation.output_representation,
                    transitions=(b_kind, cand_expr.op),
                    invariants=("rewrite_proof_verified", intention.effect_semantics.value),
                    observers=("standalone_audit",),
                )
                raw_candidates.append((rid, b_kind, cand_expr, proof, est_cost, gspec))

        if not raw_candidates:
            raise IntentionCompilerError(
                f"No feasible realization satisfies cost ceilings & proof obligations for {intention.intention_id!r}."
            )

        # Compute exact Pareto frontier: r is Pareto-optimal iff no r' strictly Pareto-dominates r!
        pareto_flags: list[bool] = []
        for i, (_, _, _, _, cost_i, _) in enumerate(raw_candidates):
            dominated = any(
                j != i and raw_candidates[j][4].pareto_dominates(cost_i)
                for j in range(len(raw_candidates))
            )
            pareto_flags.append(not dominated)

        candidates = tuple(
            CandidateRealization(
                realization_id=rid,
                backend_kind=b_kind,
                target_expression=cand_expr,
                rewrite_proof=proof,
                estimated_cost=est_cost,
                ground_spec=gspec,
                pareto_optimal=p_opt,
            )
            for (rid, b_kind, cand_expr, proof, est_cost, gspec), p_opt in zip(raw_candidates, pareto_flags)
        )
        frontier = [c for c in candidates if c.pareto_optimal]
        frontier.sort(
            key=lambda c: (
                c.estimated_cost.bytes_moved,
                c.estimated_cost.compute_ops,
                c.estimated_cost.latency_ms,
                c.realization_id,
            )
        )
        return CompiledRealizationPlan(
            intention=intention,
            candidates=candidates,
            pareto_frontier_ids=tuple(c.realization_id for c in frontier),
            selected_realization_id=frontier[0].realization_id,
        )

    def realize(
        self,
        plan: CompiledRealizationPlan,
        inputs: Mapping[str, Any],
        *,
        realization_id: str | None = None,
        quorum: QuorumLineageCoordinator | None = None,
        quorum_public_keys: Mapping[str, str] | None = None,
        quorum_secrets: Mapping[str, bytes] | None = None,
        active_quorum_members: Sequence[str] | None = None,
    ) -> RealizedObservationReceipt:
        """Execute a compiled candidate realization and independently audit its rewrite, authority & observation."""
        cand = plan.get_candidate(realization_id)
        intention = plan.intention
        rewrite_audit = verify_rewrite_proof_chain(intention.expression, cand.target_expression, cand.rewrite_proof)

        # Provision isolated quorum if not externally supplied (defaults to Ed25519 public-key authority!)
        active_members = tuple(active_quorum_members or intention.authority.member_ids)
        pub_keys_used: dict[str, str] | None = dict(quorum_public_keys) if quorum_public_keys is not None else None
        secs_used: dict[str, bytes] | None = dict(quorum_secrets) if quorum_secrets is not None else None
        if quorum is None:
            m_ids = intention.authority.member_ids
            lin_id = f"{intention.authority.lineage_id}.{cand.realization_id}"
            if intention.authority.require_public_key_quorum:
                kps, pub_keys_used = build_ed25519_quorum_identities(
                    m_ids, deterministic_domain_seed=lin_id.encode("utf-8")
                )
                q_mem = QuorumMembership(lin_id, m_ids, member_public_keys=pub_keys_used)
                q_members = [QuorumAuthorityMember.from_ed25519_keypair(m, kps[m]) for m in m_ids]
                quorum = QuorumLineageCoordinator(q_mem, q_members)
            else:
                secs_used = {
                    m: hashlib.sha256(f"hmac:{lin_id}:{m}".encode("utf-8")).digest() for m in m_ids
                }
                q_mem = QuorumMembership(lin_id, m_ids)
                q_members = [QuorumAuthorityMember(m, secs_used[m]) for m in m_ids]
                quorum = QuorumLineageCoordinator(q_mem, q_members, secs_used)

        t0 = time.perf_counter()
        transit_pkt_dict: dict[str, Any] | None = None

        def _compute_backend_value() -> tuple[Any, int, float]:
            nonlocal transit_pkt_dict
            if cand.backend_kind == "fluid_in_transit_stream" and cand.target_expression.op in (
                "closed_orbit_jump",
                "affine_mod",
                "affine_chain",
                "iterate_affine",
            ):
                par = cand.target_expression.params
                raw_in = _eval_expr_direct(cand.target_expression.args[0], inputs)
                is_scalar = isinstance(raw_in, int)
                vec_in = [int(raw_in)] if is_scalar else [int(x) for x in raw_in]
                m = int(par["modulus"])
                if cand.target_expression.op in ("closed_orbit_jump", "affine_mod"):
                    an, bn = int(par["a"]) % m, int(par["b"]) % m
                elif cand.target_expression.op == "iterate_affine":
                    an, bn = _affine_power_mod(int(par["a"]), int(par["b"]), int(par["steps"]), m)
                else:
                    an, bn = 1 % m, 0
                    for pa, pb in par["chain"]:
                        an, bn = (int(pa) * an) % m, (int(pa) * bn + int(pb)) % m
                machine = InTransitPathMachine(
                    [StreamInterpreterNode("transit.hop.0", max_ops_per_hop=1), StreamInterpreterNode("transit.hop.1", max_ops_per_hop=1)]
                )
                pkt, _stats = machine.execute_in_transit(
                    packet_id=f"pkt.{cand.realization_id}",
                    initial_payload=vec_in,
                    schedule=[TransitOperation("affine_map", {"bias": bn, "modulus": m, "scale": an})],
                    target_representation="int_vector",
                    optimize=False,
                )
                wire_b = pkt.to_wire_bytes()
                transit_pkt_dict = parse_strict_json(wire_b)
                verify_transit_packet_receipts_standalone(transit_pkt_dict)
                res_val = int(pkt.payload[0]) if is_scalar else [int(x) for x in pkt.payload]
                return res_val, len(wire_b), float(len(vec_in))

            val = _eval_expr_direct(cand.target_expression, inputs)
            b_est = int(cand.estimated_cost.bytes_moved)
            c_est = float(cand.estimated_cost.compute_ops)
            return val, b_est, c_est

        # Wrap execution inside ContinuousVirtualSystem so every realization produces a standalone-auditable receipt!
        store = PagedObjectStore(InMemoryPageBacking(65536))
        journal = HashChainedJournal()
        sys_runner = ContinuousVirtualSystem(
            pipeline_id=f"intent.{intention.intention_id}.{cand.realization_id}",
            store=store,
            quorum=quorum,
            active_quorum_members=active_members,
            journal=journal,
            require_independent_verifier=intention.observation.require_independent_audit,
        )

        # Canonical JSON-safe input envelope
        json_inputs = {
            k: ({"__bytes_hex__": bytes(v).hex()} if isinstance(v, (bytes, bytearray)) else v)
            for k, v in sorted(inputs.items())
        }

        def _stage_execute(_state: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
            eff_rcpt = self.effect_ledger.execute_effect(
                intention_id=intention.intention_id,
                effect_key=cand.realization_id,
                semantics=intention.effect_semantics,
                action_fn=_compute_backend_value,
            )
            obs_val, bytes_mv, comp_ops = eff_rcpt.result
            elapsed_ms = max(0.01, (time.perf_counter() - t0) * 1000.0)
            out_val = list(obs_val) if isinstance(obs_val, bytes) else obs_val
            return (
                {"observation": out_val, "realization_id": cand.realization_id},
                NavigationCostVector(elapsed_ms, comp_ops, float(bytes_mv), 2.0),
                f"PASS:{cand.backend_kind}",
            )

        def _stage_verify(_inp: Mapping[str, Any], out: Mapping[str, Any]) -> tuple[bool, str]:
            if not isinstance(out, Mapping) or "observation" not in out:
                return False, "missing_observation"
            return True, f"intent_contract:{intention.observation.output_representation}"

        stage = ContinuousStageSpec(
            stage_id="realize_intention",
            ground_spec=cand.ground_spec,
            input_representation="intent_env",
            output_representation=intention.observation.output_representation,
            scheduler_id=f"sched.{cand.backend_kind}",
            execute_fn=_stage_execute,
            verify_fn=_stage_verify,
        )
        pipe_res = sys_runner.execute_stages([stage], json_inputs, host_id=f"vld.{cand.backend_kind}")
        rcpt_dicts = tuple(r.to_dict() for r in pipe_res.receipts)

        auth_audit = verify_continuity_receipt_chain_standalone(
            rcpt_dicts,
            pipeline_id=sys_runner.pipeline_id,
            initial_input=json_inputs,
            expected_stages=[
                {
                    "ground_fingerprint": cand.ground_spec.fingerprint,
                    "input_representation": "intent_env",
                    "output_representation": intention.observation.output_representation,
                    "stage_id": "realize_intention",
                }
            ],
            quorum_member_ids=quorum.membership.member_ids,
            quorum_member_secrets=secs_used if quorum.membership.signature_scheme == "hmac-sha256" else None,
            quorum_member_public_keys=pub_keys_used if quorum.membership.signature_scheme == "ed25519-rfc8032" else None,
            require_verified_tag=intention.observation.require_independent_audit,
        )

        return RealizedObservationReceipt(
            intention_id=intention.intention_id,
            intention_fingerprint=intention.intention_fingerprint,
            realization_id=cand.realization_id,
            backend_kind=cand.backend_kind,
            admitted_observation=pipe_res.final_state["observation"],
            measured_cost=pipe_res.total_cost,
            rewrite_audit=rewrite_audit,
            authority_audit=auth_audit,
            quorum_public_keys=pub_keys_used,
            continuity_receipts=rcpt_dicts,
            transit_packet_dict=transit_pkt_dict,
        )

    def verify_interchangeable_realizations(
        self,
        plan: CompiledRealizationPlan,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Execute ALL compatible candidates for `plan` and prove their admitted observations agree within tolerance."""
        receipts: list[RealizedObservationReceipt] = []
        for cand in plan.candidates:
            receipts.append(self.realize(plan, inputs, realization_id=cand.realization_id))

        ref_obs = receipts[0].admitted_observation
        tol = float(plan.intention.observation.numerical_tolerance)
        max_diff = 0.0
        for r in receipts[1:]:
            obs = r.admitted_observation
            if tol == 0.0:
                if obs != ref_obs:
                    raise IntentionCompilerError(
                        f"Interchangeable realization mismatch between {receipts[0].realization_id!r} "
                        f"and {r.realization_id!r}: {ref_obs!r} != {obs!r}."
                    )
            else:
                flat_a = _flatten_floats(ref_obs)
                flat_b = _flatten_floats(obs)
                if len(flat_a) != len(flat_b):
                    raise IntentionCompilerError("Observation shape mismatch across realizations.")
                diff = max((abs(a - b) for a, b in zip(flat_a, flat_b)), default=0.0)
                max_diff = max(max_diff, diff)
                if diff > tol:
                    raise IntentionCompilerError(
                        f"Numerical discrepancy {diff} exceeds tolerance {tol} on {r.realization_id!r}."
                    )

        return {
            "admitted_observation": ref_obs,
            "backends_verified": sorted({r.backend_kind for r in receipts}),
            "candidate_count": len(receipts),
            "intention_fingerprint": plan.intention.intention_fingerprint,
            "max_numerical_discrepancy": max_diff,
            "pareto_selected": plan.selected_realization_id,
            "receipts": receipts,
            "status": "INTERCHANGEABLE_VERIFIED",
        }


def _flatten_floats(val: Any) -> list[float]:
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, (list, tuple)):
        out: list[float] = []
        for item in val:
            out.extend(_flatten_floats(item))
        return out
    raise IntentionCompilerError(f"Cannot compare numerical tolerance on {type(val).__name__}.")


__all__ = [
    "AuthorityPolicy",
    "CandidateRealization",
    "CompiledRealizationPlan",
    "EffectExecutionReceipt",
    "EffectLedger",
    "EffectSemantics",
    "IntentExpr",
    "IntentionCompilerError",
    "IntentionSpec",
    "IntentionToRealizationCompiler",
    "ObservationContract",
    "ProofProducingEGraph",
    "RealizedObservationReceipt",
    "RewriteProofStep",
    "RewriteVerificationError",
    "verify_rewrite_proof_chain",
]
