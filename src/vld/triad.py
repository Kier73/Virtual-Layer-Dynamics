r"""Three-valued guard semantics, epistemic classification, and state-space triad analysis.

Mathematical model:
Given a finite Virtual Ground transition relation $\Delta \subseteq \Sigma \times A \times \Sigma$
and an observer projection $P: \Sigma \to Y$:

1. **Three-valued Kleene guard logic ($\mathcal{K}_3$):**
   For phase or metric distance $d \in [0, \pi]$ with conservative uncertainty bound $e \ge 0$,
   let $[L, U] = [\max(0, d - e), \min(\pi, d + e)]$. Relative to tolerance $\tau \ge 0$:
     - $U \le \tau \implies \text{TRUE}$
     - $L > \tau   \implies \text{FALSE}$
     - $L \le \tau < U \implies \text{INDETERMINATE}$ (atomic trap; never guess a branch).

2. **Predictability & Information Triad over $(\Sigma, A, \Delta, P)$:**
   - `PREDICTABLE_PRESERVING`: $|\Delta(s, a)| = 1$ and action map $\delta_a$ is injective on $\Sigma$
     ($H_{\text{erasure}}(a) = 0$ bits; invertible / group-like orbit).
   - `PREDICTABLE_CONTRACTING`: $|\Delta(s, a)| = 1$ and $|\delta_a^{-1}(\delta_a(s))| > 1$
     (deterministic many-to-one step that removes historical distinctions, enabling compact
     sufficient-state checkpoints; step collision erasure under uniform prior is
     $H_{\text{erasure}}(a) = \sum_{s' \in \operatorname{im}(\delta_a)} \frac{|\delta_a^{-1}(s')|}{|\Sigma|} \log_2 |\delta_a^{-1}(s')|$ bits).
   - `INFORMATION_PRODUCING`: environment/sensor-coupled transition declared in `external_couplings`
     whose realization injects external mutual information $I(S'; \text{Env} \mid S=s, A=a) > 0$.
   - `AMBIGUOUS`: $|\Delta(s, a)| > 1$ without an admitted external-coupling model (conflicting hypotheses).
   - `UNKNOWN`: $|\Delta(s, a)| = 0$ (unobserved transition cell).

3. **Predictive Quotient (Moore / Myhill-Nerode Partition Refinement):**
   $s_1 \sim_P s_2$ iff $P(s_1) = P(s_2)$ and for every complete deterministic action $a$,
   $\delta_a(s_1) \sim_P \delta_a(s_2)$ (and identical unresolved/multi-target signatures otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping


class TruthDisposition(str, Enum):
    """Three-valued guard disposition under Kleene strong logic K_3."""

    TRUE = "true"
    FALSE = "false"
    INDETERMINATE = "indeterminate"

    def __invert__(self) -> TruthDisposition:
        if self is TruthDisposition.TRUE:
            return TruthDisposition.FALSE
        if self is TruthDisposition.FALSE:
            return TruthDisposition.TRUE
        return TruthDisposition.INDETERMINATE

    def __and__(self, other: TruthDisposition) -> TruthDisposition:
        if self is TruthDisposition.FALSE or other is TruthDisposition.FALSE:
            return TruthDisposition.FALSE
        if self is TruthDisposition.TRUE and other is TruthDisposition.TRUE:
            return TruthDisposition.TRUE
        return TruthDisposition.INDETERMINATE

    def __or__(self, other: TruthDisposition) -> TruthDisposition:
        if self is TruthDisposition.TRUE or other is TruthDisposition.TRUE:
            return TruthDisposition.TRUE
        if self is TruthDisposition.FALSE and other is TruthDisposition.FALSE:
            return TruthDisposition.FALSE
        return TruthDisposition.INDETERMINATE


class IndeterminateGuardTrap(RuntimeError):
    """Raised when an execution guard evaluates to INDETERMINATE and refuses to guess."""


class EpistemicStatus(str, Enum):
    """Distinct evidential categories (not a linear truth ladder)."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    HYPOTHESIS = "hypothesis"
    CERTIFIED = "certified"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class TransitionClass(str, Enum):
    """Information and predictability signature of a (state, action) transition."""

    PREDICTABLE_PRESERVING = "predictable_preserving"
    PREDICTABLE_CONTRACTING = "predictable_contracting"
    INFORMATION_PRODUCING = "information_producing"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


def evaluate_interval_guard(distance: float, error_bound: float, tolerance: float) -> TruthDisposition:
    r"""Evaluate conservative interval equality guard $d \pm e \le \tau$ on $[0, \pi]$."""
    if not (math.isfinite(distance) and math.isfinite(error_bound) and math.isfinite(tolerance)):
        raise ValueError("Guard parameters must be finite.")
    if distance < 0.0 or error_bound < 0.0 or tolerance < 0.0:
        raise ValueError("Distance, error_bound, and tolerance must be non-negative.")
    if distance > math.pi or tolerance > math.pi:
        raise ValueError("Phase distance and tolerance must lie in [0, pi].")

    bounded_error = min(math.pi, error_bound)
    lower = max(0.0, distance - bounded_error)
    upper = min(math.pi, distance + bounded_error)
    if upper <= tolerance:
        return TruthDisposition.TRUE
    if lower > tolerance:
        return TruthDisposition.FALSE
    return TruthDisposition.INDETERMINATE


def require_determinate_branch(disposition: TruthDisposition, context: str = "branch") -> bool:
    """Convert TRUE/FALSE to Python bool, trapping atomically on INDETERMINATE."""
    if disposition is TruthDisposition.TRUE:
        return True
    if disposition is TruthDisposition.FALSE:
        return False
    raise IndeterminateGuardTrap(f"Indeterminate guard refused to guess at {context}.")


class IncompleteGroundError(ValueError):
    """Raised when compilation requires a complete deterministic Ground."""


@dataclass(frozen=True)
class ActionInformationSummary:
    """Information-flow metrics for a single action across the finite state space."""

    action: str
    is_total_deterministic: bool
    image_cardinality: int
    erasure_bits: float
    unknown_cells: int
    ambiguous_cells: int
    information_producing_cells: int


@dataclass(frozen=True)
class GroundAnalysisReport:
    """Structural, predictive, and information-theoretic analysis of a finite Ground."""

    state_count: int
    action_count: int
    reachable_count: int
    quotient_class_count: int
    quotient_classes: tuple[tuple[str, ...], ...]
    cell_classes: dict[tuple[str, str], TransitionClass]
    action_summaries: dict[str, ActionInformationSummary]

    @property
    def compression_ratio(self) -> float:
        r"""Ratio $|\Sigma / {\sim_P}| / |\Sigma|$ in $(0, 1]$. Smaller means stronger predictive collapse."""
        return self.quotient_class_count / self.state_count

    @property
    def is_executable_deterministic(self) -> bool:
        """True iff every (state, action) cell is deterministic (preserving or contracting)."""
        return all(
            cls in (TransitionClass.PREDICTABLE_PRESERVING, TransitionClass.PREDICTABLE_CONTRACTING)
            for cls in self.cell_classes.values()
        )

    def require_executable_deterministic(self) -> None:
        """Refuse lowering when any cell is UNKNOWN, AMBIGUOUS, or externally ungrounded."""
        bad = [
            f"({s}, {a})={cls.value}"
            for (s, a), cls in sorted(self.cell_classes.items())
            if cls not in (TransitionClass.PREDICTABLE_PRESERVING, TransitionClass.PREDICTABLE_CONTRACTING)
        ]
        if bad:
            preview = ", ".join(bad[:5])
            raise IncompleteGroundError(
                f"Ground contains {len(bad)} non-deterministic or unresolved cell(s): {preview}"
            )


def analyze_finite_ground(
    states: Iterable[str],
    actions: Iterable[str],
    transitions: Mapping[tuple[str, str], Iterable[str]],
    observer: Mapping[str, str],
    *,
    initial_states: Iterable[str] | None = None,
    external_couplings: Iterable[tuple[str, str]] = (),
) -> GroundAnalysisReport:
    r"""Analyze capacity, predictive quotient, and predictability/information triad of a Ground."""
    state_tuple = tuple(sorted(set(states)))
    action_tuple = tuple(sorted(set(actions)))
    if not state_tuple:
        raise ValueError("Ground must declare at least one state.")
    if not action_tuple:
        raise ValueError("Ground must declare at least one action.")

    state_set = set(state_tuple)
    action_set = set(action_tuple)
    for s in state_tuple:
        if s not in observer:
            raise ValueError(f"Observer projection missing for state {s!r}.")

    coupled_set = set(external_couplings)
    for s, a in coupled_set:
        if s not in state_set or a not in action_set:
            raise ValueError(f"External coupling {(s, a)!r} references undeclared state or action.")

    normalized: dict[tuple[str, str], tuple[str, ...]] = {}
    for (s, a), targets in transitions.items():
        if s not in state_set or a not in action_set:
            raise ValueError(f"Transition key {(s, a)!r} references undeclared state or action.")
        target_tuple = tuple(sorted(set(targets)))
        for t in target_tuple:
            if t not in state_set:
                raise ValueError(f"Transition target {t!r} from {(s, a)!r} is not in declared states.")
        normalized[(s, a)] = target_tuple

    for s in state_tuple:
        for a in action_tuple:
            normalized.setdefault((s, a), ())

    # 1. Reachable state capacity via BFS from initial_states (defaults to all states if unspecified).
    seeds = tuple(sorted(set(initial_states))) if initial_states is not None else state_tuple
    if not seeds:
        raise ValueError("initial_states cannot be empty when explicitly supplied.")
    for s in seeds:
        if s not in state_set:
            raise ValueError(f"Initial state {s!r} is not in declared states.")

    visited = set(seeds)
    frontier = list(seeds)
    while frontier:
        curr = frontier.pop()
        for a in action_tuple:
            for nxt in normalized[(curr, a)]:
                if nxt not in visited:
                    visited.add(nxt)
                    frontier.append(nxt)

    # 2. Classify each (state, action) cell and compute per-action erasure entropy.
    cell_classes: dict[tuple[str, str], TransitionClass] = {}
    action_summaries: dict[str, ActionInformationSummary] = {}
    n_states = len(state_tuple)

    for a in action_tuple:
        preimages: dict[str, list[str]] = {}
        unknown_cnt = 0
        ambiguous_cnt = 0
        producing_cnt = 0
        det_count = 0

        for s in state_tuple:
            targets = normalized[(s, a)]
            if (s, a) in coupled_set and len(targets) >= 1:
                cell_classes[(s, a)] = TransitionClass.INFORMATION_PRODUCING
                producing_cnt += 1
            elif len(targets) == 0:
                cell_classes[(s, a)] = TransitionClass.UNKNOWN
                unknown_cnt += 1
            elif len(targets) > 1:
                cell_classes[(s, a)] = TransitionClass.AMBIGUOUS
                ambiguous_cnt += 1
            else:
                det_count += 1
                preimages.setdefault(targets[0], []).append(s)

        # Refine deterministic cells into preserving (|preimage| == 1) vs contracting (|preimage| > 1).
        for target, sources in preimages.items():
            cls = (
                TransitionClass.PREDICTABLE_PRESERVING
                if len(sources) == 1
                else TransitionClass.PREDICTABLE_CONTRACTING
            )
            for src in sources:
                cell_classes[(src, a)] = cls

        is_total_det = det_count == n_states
        erasure_bits = 0.0
        if is_total_det:
            # H(S | delta_a(S)) under uniform prior over Sigma:
            # sum_{s'} (|preimage(s')| / |Sigma|) * log2(|preimage(s')|)
            for sources in preimages.values():
                k = len(sources)
                if k > 1:
                    erasure_bits += (k / n_states) * math.log2(k)

        action_summaries[a] = ActionInformationSummary(
            action=a,
            is_total_deterministic=is_total_det,
            image_cardinality=len(preimages),
            erasure_bits=erasure_bits,
            unknown_cells=unknown_cnt,
            ambiguous_cells=ambiguous_cnt,
            information_producing_cells=producing_cnt,
        )

    # 3. Moore / Myhill-Nerode partition refinement for predictive equivalence ~_P.
    # Initial partition groups states solely by immediate observation P(s).
    obs_groups: dict[str, int] = {}
    block_of: dict[str, int] = {}
    for s in state_tuple:
        obs = observer[s]
        if obs not in obs_groups:
            obs_groups[obs] = len(obs_groups)
        block_of[s] = obs_groups[obs]

    while True:
        sig_to_block: dict[tuple[object, ...], int] = {}
        new_block_of: dict[str, int] = {}
        for s in state_tuple:
            action_sigs: list[object] = [block_of[s]]
            for a in action_tuple:
                cls = cell_classes[(s, a)]
                targets = normalized[(s, a)]
                target_blocks = tuple(sorted(block_of[t] for t in targets))
                action_sigs.append((cls.value, target_blocks))
            sig = tuple(action_sigs)
            if sig not in sig_to_block:
                sig_to_block[sig] = len(sig_to_block)
            new_block_of[s] = sig_to_block[sig]

        if len(sig_to_block) == len(set(block_of.values())):
            block_of = new_block_of
            break
        block_of = new_block_of

    blocks: dict[int, list[str]] = {}
    for s in state_tuple:
        blocks.setdefault(block_of[s], []).append(s)
    quotient_classes = tuple(sorted(tuple(sorted(members)) for members in blocks.values()))

    return GroundAnalysisReport(
        state_count=n_states,
        action_count=len(action_tuple),
        reachable_count=len(visited),
        quotient_class_count=len(quotient_classes),
        quotient_classes=quotient_classes,
        cell_classes=cell_classes,
        action_summaries=action_summaries,
    )
