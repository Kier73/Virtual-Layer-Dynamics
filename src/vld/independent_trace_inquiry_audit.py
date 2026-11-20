"""Standalone TRACE-002 transcript auditor (stdlib only; zero vld imports).

Replay the full adaptive plan, its finite-field ranks, cost ledger and ordered
observations. A caller-supplied evidence_ref is *not* an authenticated observation.
Cryptographic attribution of remote observer-hosts remains a separate contract.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from typing import Any, Mapping


class TraceInquiryAuditError(ValueError):
    """Invalid inquiry transcript, forged selection, or inconsistent ledger."""


def _int(x: Any, low: int = 0, high: int | None = None) -> int:
    if type(x) is not int or x < low or (high is not None and x > high):
        raise TraceInquiryAuditError("Expected canonical bounded integer")
    return x


def _field_prime(p: int) -> bool:
    if p == 2:
        return True
    if p % 2 == 0:
        return False
    q = 3
    while q <= p // q:
        if p % q == 0:
            return False
        q += 2
    return True


def _frac(raw: Any, *, allow_zero: bool = False) -> Fraction:
    if not isinstance(raw, list) or len(raw) != 2:
        raise TraceInquiryAuditError("Rational must be a canonical pair")
    n, d = raw
    _int(n)
    _int(d, 1)
    f = Fraction(n, d)
    if (n, d) != (f.numerator, f.denominator) or (f == 0 and not allow_zero):
        raise TraceInquiryAuditError("Noncanonical or disallowed rational")
    return f


def _hash(domain: str, body: Any) -> str:
    try:
        raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError, TypeError) as exc:
        raise TraceInquiryAuditError("Invalid canonical JSON body") from exc
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + raw).hexdigest()


def _rank(matrix: list[list[int]], d: int, p: int) -> int:
    """Independent full-column Gauss-Jordan elimination."""
    grid = [row[:] for row in matrix]
    rank = 0
    for col in range(d):
        pivot = None
        for r in range(rank, len(grid)):
            if grid[r][col] % p:
                pivot = r
                break
        if pivot is None:
            continue
        grid[rank], grid[pivot] = grid[pivot], grid[rank]
        k = pow(grid[rank][col], p - 2, p)
        grid[rank] = [v * k % p for v in grid[rank]]
        for r in range(len(grid)):
            if r != rank:
                factor = grid[r][col]
                if factor:
                    grid[r] = [(a - factor * b) % p for a, b in zip(grid[r], grid[rank])]
        rank += 1
    return rank


def _fit(traces: list[list[int]], degree: int, p: int) -> tuple[int, list[int] | None, bool]:
    rows = []
    for trace in traces:
        for t in range(degree, len(trace)):
            rows.append([trace[t - j] for j in range(1, degree + 1)] + [trace[t]])
    coeff_rank = _rank([r[:degree] for r in rows], degree, p)
    augmented_rank = _rank(rows, degree + 1, p)
    if augmented_rank != coeff_rank:
        return coeff_rank, None, True
    if coeff_rank < degree:
        return coeff_rank, None, False

    # Independently solve the uniquely determined coefficients by Gauss-Jordan.
    a = [row[:] for row in rows]
    pivots: list[int] = []
    pivotrow = 0
    for col in range(degree):
        target = next((i for i in range(pivotrow, len(a)) if a[i][col] % p), None)
        if target is None:
            continue
        a[target], a[pivotrow] = a[pivotrow], a[target]
        inv = pow(a[pivotrow][col], -1, p)
        a[pivotrow] = [v * inv % p for v in a[pivotrow]]
        for i in range(len(a)):
            if i != pivotrow:
                factor = a[i][col]
                if factor:
                    a[i] = [(x - factor * y) % p for x, y in zip(a[i], a[pivotrow])]
        pivots.append(col)
        pivotrow += 1
    solution = [0] * degree
    for r, col in enumerate(pivots):
        solution[col] = a[r][degree]
    return coeff_rank, solution, False


def verify_trace_inquiry_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Check that every query selected the promised one-step rank/cost policy."""
    genesis = {"schema", "prime", "order", "initial_traces", "observer_costs", "budget",
               "holdouts_per_channel", "max_observations"}
    final = {"events", "final_traces", "final_coefficients", "training_cutoff",
             "remaining_budget", "status", "bundle_digest"}
    if not isinstance(bundle, Mapping) or set(bundle) != genesis | final:
        raise TraceInquiryAuditError("Invalid inquiry bundle fields")
    if bundle["schema"] != "vld.trace-inquiry.v1":
        raise TraceInquiryAuditError("Unsupported inquiry schema")
    prime = _int(bundle["prime"], 2, 2**31 - 1)
    if not _field_prime(prime):
        raise TraceInquiryAuditError("Modulus must be prime")
    degree = _int(bundle["order"], 1, 32)
    holdouts = _int(bundle["holdouts_per_channel"], 2, 32)
    max_obs = _int(bundle["max_observations"], 1, 128)
    initial = bundle["initial_traces"]
    if not isinstance(initial, list) or not 1 <= len(initial) <= 32:
        raise TraceInquiryAuditError("Initial trace channel count")
    traces = []
    for seq in initial:
        if not isinstance(seq, list) or not degree <= len(seq) <= 512:
            raise TraceInquiryAuditError("Invalid initial trace prefix")
        for x in seq:
            _int(x, 0, prime - 1)
        traces.append(seq[:])
    costs_raw = bundle["observer_costs"]
    if not isinstance(costs_raw, list) or len(costs_raw) != len(traces):
        raise TraceInquiryAuditError("Observer cost shape")
    costs = [_frac(x) for x in costs_raw]
    budget = _frac(bundle["budget"])
    remaining = budget

    def state_fit() -> tuple[int, list[int] | None, bool]:
        return _fit(traces, degree, prime)

    rank, solution, inconsistent = state_fit()
    if inconsistent:
        raise TraceInquiryAuditError("Initial training already contradicts model")
    training_cutoff = max(3, max(map(len, traces))) if solution is not None else None
    refuted = False
    origin_body = {key: bundle[key] for key in genesis}
    predecessor = _hash("vld.trace-inquiry-genesis.v1", origin_body)

    events = bundle["events"]
    if not isinstance(events, list) or len(events) > max_obs:
        raise TraceInquiryAuditError("Event count exceeds budget")
    required_event = {"seq", "predecessor_digest", "channel", "coordinate", "phase", "rank_gain",
                      "rank_before", "rank_after", "cost", "value", "evidence_ref",
                      "refuted", "event_digest"}
    for event_index, event in enumerate(events):
        if refuted:
            raise TraceInquiryAuditError("Cannot append observations after refutation")
        if not isinstance(event, dict) or set(event) != required_event:
            raise TraceInquiryAuditError("Invalid event structure")
        if type(event["seq"]) is not int or event["seq"] != event_index:
            raise TraceInquiryAuditError("Nonmonotone event sequence")
        if event["predecessor_digest"] != predecessor:
            raise TraceInquiryAuditError("Broken event predecessor binding")
        if solution is not None and training_cutoff is not None and all(
            len(t) >= training_cutoff + holdouts for t in traces
        ):
            raise TraceInquiryAuditError("Cannot probe after completed validation")
        candidate: list[tuple[tuple[Any, ...], int, str, int]] = []
        for channel, trace in enumerate(traces):
            if costs[channel] > remaining or len(trace) >= 512:
                continue
            if solution is not None:
                if len(trace) >= training_cutoff + holdouts:
                    continue
                phase, gain = "validate", 0
                key = (costs[channel], len(trace), channel)
            else:
                regressors = [
                    [t[pos - j] for j in range(1, degree + 1)]
                    for t in traces for pos in range(degree, len(t))
                ]
                feature = [trace[-j] for j in range(1, degree + 1)]
                gain = _rank(regressors + [feature], degree, prime) - rank
                phase = "identify" if gain else "bridge"
                key = (-gain, costs[channel], len(trace), channel)
            candidate.append((key, channel, phase, gain))
        if not candidate:
            raise TraceInquiryAuditError("Event executed without an affordable eligible probe")
        _, selected_channel, selected_phase, selected_gain = min(candidate, key=lambda z: z[0])
        if (event["channel"] != selected_channel or
            event["coordinate"] != len(traces[selected_channel]) or
            event["phase"] != selected_phase or event["rank_gain"] != selected_gain or
            event["rank_before"] != rank or
            event["cost"] != [costs[selected_channel].numerator, costs[selected_channel].denominator]):
            raise TraceInquiryAuditError("Forged probe selection/rank/cost")
        val = _int(event["value"], 0, prime - 1)
        ref = event["evidence_ref"]
        if not isinstance(ref, str) or not 1 <= len(ref) <= 512 or not ref.strip():
            raise TraceInquiryAuditError("Missing attributed evidence reference")
        traces[selected_channel].append(val)
        remaining -= costs[selected_channel]
        new_rank, new_solution, inconsistent = state_fit()
        refuted = inconsistent
        if new_solution is not None and solution is None:
            solution = new_solution
            training_cutoff = max(3, max(map(len, traces)))
        if (event["rank_after"] != new_rank or type(event["refuted"]) is not bool
            or event["refuted"] != refuted):
            raise TraceInquiryAuditError("Fabricated rank or refutation outcome")
        body = {key: event[key] for key in required_event - {"event_digest"}}
        event_digest = _hash("vld.trace-inquiry-event.v1", body)
        if event["event_digest"] != event_digest:
            raise TraceInquiryAuditError("Tampered observation event")
        rank = new_rank
        predecessor = event_digest

    expected_status = (
        "REFUTED" if refuted else
        "UNDERDETERMINED" if solution is None else
        "HELDOUT_SUPPORTED" if all(len(t) >= training_cutoff + holdouts for t in traces)
        else "VALIDATING"
    )
    if bundle["status"] != expected_status:
        raise TraceInquiryAuditError("Unsupported epistemic status")
    if bundle["final_traces"] != traces:
        raise TraceInquiryAuditError("Transcript does not reconstruct final observations")
    if bundle["final_coefficients"] != solution:
        raise TraceInquiryAuditError("Incorrect learned law coefficients")
    if bundle["training_cutoff"] != training_cutoff:
        raise TraceInquiryAuditError("Incorrect training/held-out separation")
    if bundle["remaining_budget"] != [remaining.numerator, remaining.denominator]:
        raise TraceInquiryAuditError("Incorrect remaining cost budget")
    body = {key: bundle[key] for key in genesis | (final - {"bundle_digest"})}
    digest = _hash("vld.trace-inquiry-bundle.v1", body)
    if bundle["bundle_digest"] != digest:
        raise TraceInquiryAuditError("Altered bundle commitment")
    return {
        "status": expected_status,
        "scope": "INTERNALLY_AUDITED_UNAUTHENTICATED_OBSERVATIONS",
        "rank": rank,
        "order": degree,
        "observations_acquired": len(events),
        "remaining_budget": [remaining.numerator, remaining.denominator],
        "learned_coefficients": None if solution is None else solution[:],
        "bundle_digest": digest,
    }
