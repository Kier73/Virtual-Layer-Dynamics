r"""Active perception and finite distinguishing-probe planners (`greedy`, `beam`, `minimax`).

Mathematical model:
Over a partial predictive Ground $\mathcal{T} = (\Sigma, A, s_0, C, \text{obs})$ and horizon $H \ge 1$,
each state pair $(s_1, s_2)$ ($s_1 < s_2$) is classified across bounded action words $|w| \le H$:
  - `DISTINGUISHED`: $\exists w$ with $|w| \le H$ along which both trajectories traverse exclusively
    `known` singleton transitions and emit distinct channel vectors.
  - `EQUIVALENT`: All transitions traversed from $s_1, s_2$ up to depth $H$ are `known` singletons
    and emit identical channel vectors for every $|w| \le H$.
  - `UNRESOLVED`: At least one reachable cell is `unknown` or `ambiguous` and no known word yet separates $(s_1, s_2)$.

Planners select probes $p = (s, a, \{(\omega_j, w_j)\}, c_p \in \mathbb{Q}_{>0})$:
  1. `minimax`: Exact memoized policy tree minimizing lexicographic
     $(\text{worst\_unresolved\_pairs}, \text{worst\_unresolved\_cells}, \text{worst\_cost}, \text{max\_probes})$.
  2. `greedy`: Myopic value-of-information step maximizing expected uncertainty reduction per unit `Fraction` cost.
  3. `beam`: Width-$B$ multi-step beam search over counterfactual probe trajectories.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from vld.canonical import domain_digest
from vld.signal.transitions import GroundTransitionSystem, TransitionOutcome


class ProbePlanningError(ValueError):
    """Raised when a probe specification, scenario set, or search bound is invalid."""


def _exact_fraction(value: int | Fraction, name: str, *, allow_zero: bool = False) -> Fraction:
    if isinstance(value, (bool, float)):
        raise ProbePlanningError(f"{name} must be an exact int or Fraction, not {type(value).__name__}.")
    frac = Fraction(value)
    if frac < 0 or (frac == 0 and not allow_zero):
        raise ProbePlanningError(f"{name} must be {'non-negative' if allow_zero else 'strictly positive'}.")
    return frac


@dataclass(frozen=True, slots=True)
class ProbeScenario:
    """Counterfactual outcome branch for a partial-Ground probe."""

    outcome: TransitionOutcome
    weight: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, TransitionOutcome):
            raise ProbePlanningError("ProbeScenario requires a TransitionOutcome.")
        object.__setattr__(self, "weight", _exact_fraction(self.weight, "weight", allow_zero=False))


@dataclass(frozen=True, slots=True)
class TransitionProbe:
    """Candidate active-perception measurement on a `(state, action)` cell."""

    state: str
    action: str
    scenarios: tuple[ProbeScenario, ...]
    cost: Fraction = Fraction(1)
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, str) or not self.state:
            raise ProbePlanningError("probe state must be a non-empty string.")
        if not isinstance(self.action, str) or not self.action:
            raise ProbePlanningError("probe action must be a non-empty string.")
        scens = tuple(self.scenarios)
        if not scens or any(not isinstance(s, ProbeScenario) for s in scens):
            raise ProbePlanningError("probe requires at least one ProbeScenario.")
        seen_outcomes = {s.outcome for s in scens}
        if len(seen_outcomes) != len(scens):
            raise ProbePlanningError("Duplicate TransitionOutcome in probe scenarios.")
        c = _exact_fraction(self.cost, "cost", allow_zero=False)
        lbl = self.label or f"probe:{self.state}:{self.action}"
        object.__setattr__(self, "scenarios", scens)
        object.__setattr__(self, "cost", c)
        object.__setattr__(self, "label", lbl)


@dataclass(frozen=True, slots=True)
class GroundUncertaintyMetrics:
    """Exact counts of unresolved state-equivalence pairs and unresolved transition cells."""

    distinguished_pairs: int
    equivalent_pairs: int
    unresolved_pairs: int
    unresolved_cells: int


def evaluate_ground_uncertainty(system: GroundTransitionSystem, horizon: int = 2) -> GroundUncertaintyMetrics:
    r"""Classify all unordered state pairs $(s_1, s_2)$ within bounded horizon $H$."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ProbePlanningError("horizon must be a positive integer.")

    states = system.states
    actions = system.actions
    unresolved_cells = len(system.unknown_pairs) + len(system.ambiguous_pairs)

    distinguished = 0
    equivalent = 0
    unresolved = 0

    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            s1, s2 = states[i], states[j]
            queue: deque[tuple[str, str, int]] = deque([(s1, s2, 0)])
            seen: set[tuple[str, str, int]] = {(s1, s2, 0)}
            found_diff = False
            saw_unknown_or_ambig = False

            while queue and not found_diff:
                u1, u2, depth = queue.popleft()
                if depth >= horizon:
                    continue
                for act in actions:
                    obs1 = system.get_observation(u1, act)
                    obs2 = system.get_observation(u2, act)
                    if obs1 is None or obs2 is None or obs1.status != "known" or obs2.status != "known":
                        saw_unknown_or_ambig = True
                        continue
                    o1 = obs1.outcomes[0]
                    o2 = obs2.outcomes[0]
                    if o1.output != o2.output:
                        found_diff = True
                        break
                    nxt = (o1.target, o2.target, depth + 1)
                    if o1.target != o2.target and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)

            if found_diff:
                distinguished += 1
            elif saw_unknown_or_ambig:
                unresolved += 1
            else:
                equivalent += 1

    return GroundUncertaintyMetrics(
        distinguished_pairs=distinguished,
        equivalent_pairs=equivalent,
        unresolved_pairs=unresolved,
        unresolved_cells=unresolved_cells,
    )


def apply_counterfactual_probe(
    system: GroundTransitionSystem,
    probe: TransitionProbe,
    outcome: TransitionOutcome,
) -> GroundTransitionSystem:
    """Validate probe compatibility with existing Ground observations and return updated Ground."""
    obs = system.get_observation(probe.state, probe.action)
    if obs is not None and obs.status == "known" and obs.outcomes[0] != outcome:
        raise ProbePlanningError(
            f"Probe scenario outcome {outcome} contradicts already-known cell {(probe.state, probe.action)}."
        )
    if obs is not None and obs.status == "ambiguous" and outcome not in obs.outcomes:
        raise ProbePlanningError(
            f"Probe scenario outcome {outcome} is not among declared ambiguous alternatives for {(probe.state, probe.action)}."
        )
    return system.with_resolved_cell(probe.state, probe.action, outcome, evidence_tag=f"counterfactual:{probe.label}")


@dataclass(frozen=True, slots=True)
class PolicyBranch:
    outcome: TransitionOutcome
    weight: Fraction
    child: PolicyNode


@dataclass(frozen=True, slots=True)
class PolicyNode:
    ground_fingerprint: str
    unresolved_pairs_before: int
    unresolved_cells_before: int
    worst_case_unresolved_pairs: int
    worst_case_unresolved_cells: int
    worst_case_cost: Fraction
    maximum_probes: int
    probe: TransitionProbe | None = None
    branches: tuple[PolicyBranch, ...] = ()
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class MinimaxProbePlan:
    system_fingerprint: str
    horizon: int
    budget: Fraction | None
    max_probes: int
    root: PolicyNode
    expanded_states: int

    @property
    def first_probe(self) -> TransitionProbe | None:
        return self.root.probe

    @property
    def policy_digest(self) -> str:
        return domain_digest(
            "vld.minimax-probe-plan.v1",
            {
                "expanded_states": self.expanded_states,
                "first_probe": None if self.first_probe is None else self.first_probe.label,
                "horizon": self.horizon,
                "max_probes": self.max_probes,
                "system_fingerprint": self.system_fingerprint,
                "worst_case_cost": [self.root.worst_case_cost.numerator, self.root.worst_case_cost.denominator],
                "worst_case_unresolved_cells": self.root.worst_case_unresolved_cells,
                "worst_case_unresolved_pairs": self.root.worst_case_unresolved_pairs,
            },
        )


def plan_probes_minimax(
    system: GroundTransitionSystem,
    candidates: Sequence[TransitionProbe],
    *,
    horizon: int = 2,
    budget: int | Fraction | None = None,
    max_probes: int = 3,
    max_search_states: int = 2048,
) -> MinimaxProbePlan:
    """Compute exact worst-case optimal probe policy tree over finite counterfactual scenarios."""
    if isinstance(max_probes, bool) or not isinstance(max_probes, int) or max_probes < 1:
        raise ProbePlanningError("max_probes must be a positive integer.")
    frac_budget = None if budget is None else _exact_fraction(budget, "budget", allow_zero=True)
    probe_tuple = tuple(candidates)
    memo: dict[tuple[Any, ...], PolicyNode] = {}
    expanded = 0

    def solve(
        cur_sys: GroundTransitionSystem,
        used: frozenset[tuple[str, str]],
        rem_budget: Fraction | None,
    ) -> PolicyNode:
        nonlocal expanded
        key = (cur_sys.fingerprint, tuple(sorted(used)), rem_budget)
        if key in memo:
            return memo[key]
        if expanded >= max_search_states:
            raise ProbePlanningError(f"Minimax search exceeded max_search_states={max_search_states}.")
        expanded += 1

        metrics = evaluate_ground_uncertainty(cur_sys, horizon=horizon)
        avail = [
            p
            for p in probe_tuple
            if (p.state, p.action) not in used and (rem_budget is None or p.cost <= rem_budget)
        ]
        avail.sort(key=lambda p: (p.state, p.action, p.label, p.cost))

        if metrics.unresolved_pairs == 0 and metrics.unresolved_cells == 0:
            reason = "fully_resolved"
        elif len(used) >= max_probes:
            reason = "probe_limit_reached"
        elif not avail:
            reason = "budget_or_candidates_exhausted"
        else:
            reason = "stop_is_optimal"

        best = PolicyNode(
            ground_fingerprint=cur_sys.fingerprint,
            unresolved_pairs_before=metrics.unresolved_pairs,
            unresolved_cells_before=metrics.unresolved_cells,
            worst_case_unresolved_pairs=metrics.unresolved_pairs,
            worst_case_unresolved_cells=metrics.unresolved_cells,
            worst_case_cost=Fraction(0),
            maximum_probes=0,
            terminal_reason=reason,
        )
        best_score = (metrics.unresolved_pairs, metrics.unresolved_cells, Fraction(0), 0, "")

        if (metrics.unresolved_pairs == 0 and metrics.unresolved_cells == 0) or len(used) >= max_probes:
            memo[key] = best
            return best

        for p in avail:
            nxt_budget = None if rem_budget is None else (rem_budget - p.cost)
            cell = (p.state, p.action)
            branches: list[PolicyBranch] = []
            for sc in sorted(p.scenarios, key=lambda s: (s.outcome.target, s.outcome.output)):
                nxt_sys = apply_counterfactual_probe(cur_sys, p, sc.outcome)
                child = solve(nxt_sys, used | frozenset({cell}), nxt_budget)
                branches.append(PolicyBranch(sc.outcome, sc.weight, child))

            worst_pairs = max(b.child.worst_case_unresolved_pairs for b in branches)
            worst_cells = max(b.child.worst_case_unresolved_cells for b in branches)
            worst_cost = p.cost + max(b.child.worst_case_cost for b in branches)
            max_depth = 1 + max(b.child.maximum_probes for b in branches)
            cand_score = (worst_pairs, worst_cells, worst_cost, max_depth, p.label)

            if cand_score < best_score:
                best_score = cand_score
                best = PolicyNode(
                    ground_fingerprint=cur_sys.fingerprint,
                    unresolved_pairs_before=metrics.unresolved_pairs,
                    unresolved_cells_before=metrics.unresolved_cells,
                    worst_case_unresolved_pairs=worst_pairs,
                    worst_case_unresolved_cells=worst_cells,
                    worst_case_cost=worst_cost,
                    maximum_probes=max_depth,
                    probe=p,
                    branches=tuple(branches),
                    terminal_reason=None,
                )

        memo[key] = best
        return best

    root = solve(system, frozenset(), frac_budget)
    return MinimaxProbePlan(
        system_fingerprint=system.fingerprint,
        horizon=horizon,
        budget=frac_budget,
        max_probes=max_probes,
        root=root,
        expanded_states=expanded,
    )


def _expected_uncertainty_reduction(
    system: GroundTransitionSystem,
    probe: TransitionProbe,
    *,
    horizon: int,
) -> Fraction:
    base = evaluate_ground_uncertainty(system, horizon=horizon)
    base_score = Fraction(base.unresolved_pairs * 100 + base.unresolved_cells)
    total_w = sum((s.weight for s in probe.scenarios), start=Fraction(0))
    exp_after = Fraction(0)
    for sc in probe.scenarios:
        nxt = apply_counterfactual_probe(system, probe, sc.outcome)
        m = evaluate_ground_uncertainty(nxt, horizon=horizon)
        after_score = Fraction(m.unresolved_pairs * 100 + m.unresolved_cells)
        exp_after += (sc.weight / total_w) * after_score
    return (base_score - exp_after) / probe.cost


def plan_probes_greedy(
    system: GroundTransitionSystem,
    candidates: Sequence[TransitionProbe],
    *,
    horizon: int = 2,
    budget: int | Fraction | None = None,
) -> TransitionProbe | None:
    """Select the single probe maximizing expected weighted uncertainty reduction per unit `Fraction` cost."""
    frac_budget = None if budget is None else _exact_fraction(budget, "budget", allow_zero=True)
    base = evaluate_ground_uncertainty(system, horizon=horizon)
    if base.unresolved_pairs == 0 and base.unresolved_cells == 0:
        return None
    avail = [p for p in candidates if frac_budget is None or p.cost <= frac_budget]
    avail.sort(key=lambda p: (p.state, p.action, p.label))
    best_probe: TransitionProbe | None = None
    best_gain = Fraction(0)
    for p in avail:
        gain = _expected_uncertainty_reduction(system, p, horizon=horizon)
        if gain > best_gain:
            best_gain = gain
            best_probe = p
    return best_probe


def plan_probes_beam(
    system: GroundTransitionSystem,
    candidates: Sequence[TransitionProbe],
    *,
    horizon: int = 2,
    beam_width: int = 3,
    depth: int = 2,
    budget: int | Fraction | None = None,
) -> tuple[TransitionProbe, ...]:
    """Select a multi-step sequence of distinguishing probes via width-bounded beam search."""
    if beam_width < 1 or depth < 1:
        raise ProbePlanningError("beam_width and depth must be >= 1.")
    frac_budget = None if budget is None else _exact_fraction(budget, "budget", allow_zero=True)

    # Beam entry: (neg_cumulative_gain, cost_spent, sequence_of_probes, optimistic_representative_system, used_cells)
    beam: list[tuple[Fraction, Fraction, tuple[TransitionProbe, ...], GroundTransitionSystem, frozenset[tuple[str, str]]]] = [
        (Fraction(0), Fraction(0), (), system, frozenset())
    ]

    for _ in range(depth):
        next_beam: list[
            tuple[Fraction, Fraction, tuple[TransitionProbe, ...], GroundTransitionSystem, frozenset[tuple[str, str]]]
        ] = []
        for neg_gain, spent, seq, rep_sys, used in beam:
            m = evaluate_ground_uncertainty(rep_sys, horizon=horizon)
            if m.unresolved_pairs == 0 and m.unresolved_cells == 0:
                next_beam.append((neg_gain, spent, seq, rep_sys, used))
                continue
            expanded_any = False
            for p in sorted(candidates, key=lambda item: (item.state, item.action, item.label)):
                cell = (p.state, p.action)
                if cell in used:
                    continue
                if frac_budget is not None and spent + p.cost > frac_budget:
                    continue
                gain = _expected_uncertainty_reduction(rep_sys, p, horizon=horizon)
                if gain <= 0:
                    continue
                # Advance representative state using highest-weight scenario for tractable beam lookahead
                lead_scenario = max(p.scenarios, key=lambda s: (s.weight, s.outcome.target))
                nxt_sys = apply_counterfactual_probe(rep_sys, p, lead_scenario.outcome)
                next_beam.append((neg_gain - gain, spent + p.cost, (*seq, p), nxt_sys, used | frozenset({cell})))
                expanded_any = True
            if not expanded_any:
                next_beam.append((neg_gain, spent, seq, rep_sys, used))
        next_beam.sort(key=lambda item: (item[0], item[1], tuple(p.label for p in item[2])))
        beam = next_beam[:beam_width]

    return beam[0][2]
