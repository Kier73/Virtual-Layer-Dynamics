"""TRACE-002: budgeted active discovery of shared laws across observer-host traces.

An observation at time t adds the regression row [y(t-1), ..., y(t-d)].
That row is known *before* the next value is acquired. Its independence from
existing rows is therefore an exact, outcome-independent information guarantee.

This module chooses the cheapest currently affordable independent row; when no
independent row is exposed it can acquire a bridge observation, which has no
guaranteed immediate gain. The policy is one-step greedy, NOT globally optimal.
External observations carry references, not verified source authentication.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest
from vld.navigation.trace_law import (
    DiscoveredTraceLaw,
    TraceLawError,
    _prime,
    _unique_linear_solution,
)
from vld.independent_trace_inquiry_audit import verify_trace_inquiry_bundle


class TraceInquiryError(ValueError):
    """Invalid probe, insufficient evidence, exhausted budget, or stale observation."""


class TraceInquiryRefuted(TraceInquiryError):
    """New evidence has refuted the currently examined fixed-order linear family."""


def _rational(value: int | Fraction) -> Fraction:
    if isinstance(value, (float, bool)) or not isinstance(value, (int, Fraction)):
        raise TraceInquiryError("Costs and budgets must be exact int or Fraction")
    fraction = Fraction(value)
    if fraction <= 0:
        raise TraceInquiryError("Costs and budgets must be positive")
    return fraction


def _pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _rank(rows: Sequence[Sequence[int]], columns: int, p: int) -> int:
    """Coefficient/augmented rank via ordinary finite-field elimination."""
    a = [[int(x) % p for x in row] for row in rows]
    piv = 0
    for col in range(columns):
        at = next((r for r in range(piv, len(a)) if a[r][col]), None)
        if at is None:
            continue
        a[piv], a[at] = a[at], a[piv]
        inv = pow(a[piv][col], -1, p)
        a[piv] = [v * inv % p for v in a[piv]]
        for r in range(piv + 1, len(a)):
            k = a[r][col]
            if k:
                a[r] = [(v - k * u) % p for v, u in zip(a[r], a[piv])]
        piv += 1
    return piv


def _rows(traces: Sequence[Sequence[int]], order: int) -> list[list[int]]:
    return [
        [*(trace[t - lag] for lag in range(1, order + 1)), trace[t]]
        for trace in traces
        for t in range(order, len(trace))
    ]


@dataclass(frozen=True, slots=True)
class TraceObservationProbe:
    """A one-step query to one observer, bound to the exact state of inquiry."""

    channel: int
    coordinate: int
    phase: str
    rank_gain: int
    cost: Fraction
    predecessor_digest: str


class ActiveTraceInquiry:
    """Discover a fixed-order recurrence with an exact rank-aware acquisition policy.

    Initial observer prefixes need not share the same length. All samples are
    measured at local sequence coordinates 0,1,...; independent channels are
    assumed to refer to a common underlying transition/clock as declared by
    the caller. That assumption is not established by this implementation.
    """

    SCHEMA = "vld.trace-inquiry.v1"

    def __init__(
        self,
        traces: Sequence[Sequence[int]],
        *,
        prime: int,
        order: int,
        observer_costs: Sequence[int | Fraction] | None = None,
        budget: int | Fraction = 16,
        holdouts_per_channel: int = 2,
        max_observations: int = 128,
    ) -> None:
        if not _prime(prime):
            raise TraceInquiryError("prime must be a supported prime field")
        if isinstance(order, bool) or not isinstance(order, int) or not 1 <= order <= 32:
            raise TraceInquiryError("order must be in [1,32]")
        if isinstance(traces, (str, bytes)) or not 1 <= len(traces) <= 32:
            raise TraceInquiryError("traces must have 1..32 channels")
        initial = tuple(tuple(seq) for seq in traces)
        if any(len(seq) < order or len(seq) > 512 for seq in initial):
            raise TraceInquiryError("Every initial observer prefix must contain order..512 samples")
        if any(type(v) is not int or v < 0 or v >= prime for seq in initial for v in seq):
            raise TraceInquiryError("Trace values must be canonical field residues")
        if type(holdouts_per_channel) is not int or not 2 <= holdouts_per_channel <= 32:
            raise TraceInquiryError("At least 2 and at most 32 holdouts per observer are required")
        if type(max_observations) is not int or not 1 <= max_observations <= 128:
            raise TraceInquiryError("max_observations must be in [1,128]")
        costs = tuple(_rational(x) for x in (observer_costs if observer_costs is not None else [1] * len(initial)))
        if len(costs) != len(initial):
            raise TraceInquiryError("Observer costs must match observer count")

        self.prime = prime
        self.order = order
        self.initial_traces = initial
        self.costs = costs
        self.budget = _rational(budget)
        self.holdouts_per_channel = holdouts_per_channel
        self.max_observations = max_observations
        self._traces: list[list[int]] = [list(x) for x in initial]
        self._events: list[dict[str, Any]] = []
        self._remaining = self.budget
        self._refuted = False
        self._coefficients: tuple[int, ...] | None = None
        self._training_cutoff: int | None = None

        _, self._coefficients, inconsistent = self._fit()
        if inconsistent:
            raise TraceInquiryRefuted("Initial observations contradict the fixed-order linear family")
        if self._coefficients is not None:
            self._training_cutoff = max(3, *(len(t) for t in self._traces))

    def _fit(self) -> tuple[int, tuple[int, ...] | None, bool]:
        rows = _rows(self._traces, self.order)
        rank = _rank([row[:-1] for row in rows], self.order, self.prime)
        aug = _rank(rows, self.order + 1, self.prime)
        coefficients = None
        if aug == rank == self.order:
            coefficients = _unique_linear_solution(rows, self.order, self.prime)
        return rank, coefficients, aug > rank

    @property
    def rank(self) -> int:
        return self._fit()[0]

    @property
    def coefficients(self) -> tuple[int, ...] | None:
        return self._coefficients

    @property
    def status(self) -> str:
        if self._refuted:
            return "REFUTED"
        if self._coefficients is None:
            return "UNDERDETERMINED"
        assert self._training_cutoff is not None
        target = self._training_cutoff + self.holdouts_per_channel
        if all(len(seq) >= target for seq in self._traces):
            return "HELDOUT_SUPPORTED"
        return "VALIDATING"

    def _genesis(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "prime": self.prime,
            "order": self.order,
            "initial_traces": [list(t) for t in self.initial_traces],
            "observer_costs": [_pair(c) for c in self.costs],
            "budget": _pair(self.budget),
            "holdouts_per_channel": self.holdouts_per_channel,
            "max_observations": self.max_observations,
        }

    @property
    def head_digest(self) -> str:
        if self._events:
            return str(self._events[-1]["event_digest"])
        return domain_digest("vld.trace-inquiry-genesis.v1", self._genesis())

    def next_probe(self) -> TraceObservationProbe | None:
        """Select one known-rank-gain probe, or a clearly labeled speculative bridge."""
        if self._refuted or self.status == "HELDOUT_SUPPORTED" or len(self._events) >= self.max_observations:
            return None
        candidates: list[tuple[tuple[Any, ...], TraceObservationProbe]] = []
        current_rank = self.rank
        for channel, seq in enumerate(self._traces):
            if self.costs[channel] > self._remaining or len(seq) >= 512:
                continue
            if self._coefficients is not None:
                assert self._training_cutoff is not None
                if len(seq) >= self._training_cutoff + self.holdouts_per_channel:
                    continue
                phase, gain = "validate", 0
                key = (self.costs[channel], len(seq), channel)
            else:
                regressor = [seq[-lag] for lag in range(1, self.order + 1)]
                gain = _rank(
                    [row[:-1] for row in _rows(self._traces, self.order)] + [regressor],
                    self.order,
                    self.prime,
                ) - current_rank
                phase = "identify" if gain else "bridge"
                # Greedy: guaranteed information first; lowest cost breaks ties.
                key = (-gain, self.costs[channel], len(seq), channel)
            candidates.append((key, TraceObservationProbe(
                channel=channel, coordinate=len(seq), phase=phase,
                rank_gain=gain, cost=self.costs[channel],
                predecessor_digest=self.head_digest,
            )))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def observe(self, probe: TraceObservationProbe, value: int, *, evidence_ref: str) -> dict[str, Any]:
        """Accept the result of exactly the currently proposed observation.

        The evidence_ref is caller-attributed metadata, not a verified source
        signature. An inconsistent result is retained as refutation evidence.
        """
        if not isinstance(probe, TraceObservationProbe) or probe != self.next_probe():
            raise TraceInquiryError("Stale, unauthorized, or non-selected inquiry probe")
        if type(value) is not int or value < 0 or value >= self.prime:
            raise TraceInquiryError("Observation must be an exact prime-field residue")
        if not isinstance(evidence_ref, str) or not 1 <= len(evidence_ref) <= 512 or not evidence_ref.strip():
            raise TraceInquiryError("An attributed evidence reference is required")
        before = self.rank
        self._traces[probe.channel].append(value)
        after, solution, inconsistent = self._fit()
        if inconsistent:
            self._refuted = True
        elif self._coefficients is None and solution is not None:
            self._coefficients = solution
            self._training_cutoff = max(3, *(len(t) for t in self._traces))
        self._remaining -= probe.cost
        body = {
            "seq": len(self._events), "predecessor_digest": probe.predecessor_digest,
            "channel": probe.channel, "coordinate": probe.coordinate,
            "phase": probe.phase, "rank_gain": probe.rank_gain,
            "rank_before": before, "rank_after": after,
            "cost": _pair(probe.cost), "value": value,
            "evidence_ref": evidence_ref, "refuted": self._refuted,
        }
        event = {**body, "event_digest": domain_digest("vld.trace-inquiry-event.v1", body)}
        self._events.append(event)
        return dict(event)

    def bundle(self) -> dict[str, Any]:
        """Export a hash-chained, independently replayable inquiry transcript."""
        body = {
            **self._genesis(),
            "events": [dict(ev) for ev in self._events],
            "final_traces": [list(t) for t in self._traces],
            "final_coefficients": None if self._coefficients is None else list(self._coefficients),
            "training_cutoff": self._training_cutoff,
            "remaining_budget": _pair(self._remaining),
            "status": self.status,
        }
        result = {**body, "bundle_digest": domain_digest("vld.trace-inquiry-bundle.v1", body)}
        verify_trace_inquiry_bundle(result)
        return result

    def navigate(self, coordinate: int, *, operator_witness: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Produce correlated inquiry and TRACE-001 receipts, refusing unvalidated hypotheses."""
        if self.status != "HELDOUT_SUPPORTED":
            raise TraceInquiryError("Navigating requires completed held-out validation")
        bundle = self.bundle()
        assert self._training_cutoff is not None and self._coefficients is not None
        end = self._training_cutoff + self.holdouts_per_channel
        law = DiscoveredTraceLaw(
            self.prime, self._coefficients,
            tuple(tuple(t[:end]) for t in self._traces),
            self._training_cutoff,
        )
        navigation = law.navigate(coordinate, operator_witness=operator_witness)
        return {"inquiry": bundle, "navigation": navigation}
