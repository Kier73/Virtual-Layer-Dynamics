"""Discover common finite-field recurrences across observers, then navigate time without replay.

Finite observations support a *hypothesis*, never prove that future samples follow it.
An optional declared linear state operator admits a stronger CONDITIONAL proof: every
state of that operator yields the recurrence under the supplied observers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest
from vld.independent_trace_audit import verify_trace_navigation_bundle


class TraceLawError(ValueError):
    """Invalid data, underdetermined law, or held-out counterexample."""


class TraceLawUnderdetermined(TraceLawError):
    """Observations cannot uniquely identify a common recurrence of permitted order."""


class TraceLawRefuted(TraceLawError):
    """Held-out observations contradict the recurrence inferred from training data."""


def _prime(p: int) -> bool:
    if not isinstance(p, int) or isinstance(p, bool) or not 2 <= p <= 2**31 - 1:
        return False
    if p % 2 == 0:
        return p == 2
    d = 3
    while d * d <= p:
        if p % d == 0:
            return False
        d += 2
    return True


def _unique_linear_solution(rows: list[list[int]], order: int, p: int) -> tuple[int, ...] | None:
    """Solve augmented finite-field rows; return only a *unique*, consistent solution."""
    a = [row[:] for row in rows]
    pivot_cols: list[int] = []
    rank = 0
    for col in range(order):
        pivot = next((r for r in range(rank, len(a)) if a[r][col] % p), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inv = pow(a[rank][col], -1, p)
        a[rank] = [(x * inv) % p for x in a[rank]]
        for r in range(len(a)):
            if r != rank and a[r][col] % p:
                factor = a[r][col]
                a[r] = [(x - factor * y) % p for x, y in zip(a[r], a[rank])]
        pivot_cols.append(col)
        rank += 1
    if any(not any(row[:order]) and row[order] for row in a):
        return None
    if rank != order:
        return None
    solution = [0] * order
    for r, col in enumerate(pivot_cols):
        solution[col] = a[r][order]
    return tuple(solution)


def _multiply_in_quotient(a: list[int], b: list[int], c: tuple[int, ...], p: int) -> list[int]:
    """Multiply polynomials modulo x^d - sum(c[j]*x^(d-1-j))."""
    d = len(c)
    accum = [0] * (2 * d - 1)
    for i in range(d):
        for j in range(d):
            accum[i + j] = (accum[i + j] + a[i] * b[j]) % p
    for i in range(2 * d - 2, d - 1, -1):
        v = accum[i]
        for j in range(1, d + 1):
            accum[i - j] = (accum[i - j] + v * c[j - 1]) % p
    return accum[:d]


def _navigate(initial: Sequence[int], c: tuple[int, ...], p: int, coordinate: int) -> tuple[int, int]:
    """Return value and number of quotient-polynomial multiplications (not scalar FLOPs)."""
    d = len(c)
    if coordinate < d:
        return initial[coordinate], 0
    acc = [1] + [0] * (d - 1)
    base = [0] * d
    if d == 1:
        base[0] = c[0]
    else:
        base[1] = 1
    count = 0
    n = coordinate
    while n:
        if n & 1:
            acc = _multiply_in_quotient(acc, base, c, p)
            count += 1
        n >>= 1
        if n:
            base = _multiply_in_quotient(base, base, c, p)
            count += 1
    return sum(acc[i] * initial[i] for i in range(d)) % p, count


@dataclass(frozen=True, slots=True)
class DiscoveredTraceLaw:
    """Minimal *uniquely identifiable* common recurrence on the training prefix."""

    prime: int
    coefficients: tuple[int, ...]
    traces: tuple[tuple[int, ...], ...]
    training_length: int

    @property
    def law_digest(self) -> str:
        return domain_digest("vld.trace-law.v1", {"prime": self.prime, "coefficients": list(self.coefficients)})

    def navigate(self, coordinate: int, *, operator_witness: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Emit independently auditable projection; global validity is conditional on an operator witness."""
        if not isinstance(coordinate, int) or isinstance(coordinate, bool) or coordinate < 0 or coordinate.bit_length() > 4096:
            raise TraceLawError("coordinate must be a non-negative integer of at most 4096 bits")
        values = []
        for trace in self.traces:
            value, _ = _navigate(trace[:len(self.coefficients)], self.coefficients, self.prime, coordinate)
            values.append(value)
        body: dict[str, Any] = {
            "schema": "vld.trace-navigation.v1",
            "prime": self.prime,
            "coefficients": list(self.coefficients),
            "traces": [list(t) for t in self.traces],
            "training_length": self.training_length,
            "coordinate": coordinate,
            "values": values,
            "scope": "DECLARED_OPERATOR_CONDITIONAL" if operator_witness is not None else "OBSERVATIONAL_HYPOTHESIS",
            "operator_witness": None if operator_witness is None else dict(operator_witness),
        }
        receipt = {**body, "claim_digest": domain_digest("vld.trace-navigation.v1", body)}
        # Fail closed: even the producer's output must survive the independently implemented checker.
        verify_trace_navigation_bundle(receipt)
        return receipt


def discover_trace_law(
    traces: Sequence[Sequence[int]],
    *,
    prime: int,
    training_length: int,
    max_order: int = 32,
) -> DiscoveredTraceLaw:
    """Find the lowest-order common *unique* F_p recurrence; test on unseen suffix.

    Multiple observer traces must align to the same time axis; all are used as
    simultaneous constraints. At least two held-out points per observer are required.
    The returned law remains an observation-derived hypothesis until grounded in
    an independently attributable transition operator.
    """
    if not _prime(prime):
        raise TraceLawError("prime must be a prime in [2, 2^31-1]")
    if isinstance(traces, (str, bytes)) or not 1 <= len(traces) <= 32:
        raise TraceLawError("traces must have 1..32 observer channels")
    if not isinstance(training_length, int) or isinstance(training_length, bool) or training_length < 3:
        raise TraceLawError("training_length must be an exact int >= 3")
    if not isinstance(max_order, int) or isinstance(max_order, bool) or not 1 <= max_order <= 32:
        raise TraceLawError("max_order must be in [1,32]")
    normalized = tuple(tuple(trace) for trace in traces)
    length = len(normalized[0])
    if length > 512 or training_length > length - 2 or any(len(t) != length for t in normalized):
        raise TraceLawError("All traces must be aligned, length <= 512, with >= 2 held-out steps")
    for t in normalized:
        if any(not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < prime for v in t):
            raise TraceLawError("Trace entries must be canonical finite-field integers")

    for order in range(1, min(max_order, training_length - 1) + 1):
        rows = [
            [*(trace[t - j] for j in range(1, order + 1)), trace[t]]
            for trace in normalized
            for t in range(order, training_length)
        ]
        coefficients = _unique_linear_solution(rows, order, prime)
        if coefficients is None:
            continue
        for ch, trace in enumerate(normalized):
            for t in range(training_length, length):
                predicted = sum(coefficients[j - 1] * trace[t - j] for j in range(1, order + 1)) % prime
                if predicted != trace[t]:
                    raise TraceLawRefuted(
                        f"held-out counterexample at channel={ch}, coordinate={t}: "
                        f"predicted={predicted}, observed={trace[t]}"
                    )
        return DiscoveredTraceLaw(prime, coefficients, normalized, training_length)
    raise TraceLawUnderdetermined("No uniquely identified common recurrence within max_order")
