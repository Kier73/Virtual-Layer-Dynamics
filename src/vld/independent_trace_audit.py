"""Independent, stdlib-only checker for proof-carrying trace-law navigation.

No imports from vld.*. Recomputes distant state with a companion matrix, not
with the producer's quotient-polynomial algorithm. An attributed linear-operator
witness proves a recurrence for *all states of that declared operator*; it does
not establish that an unknown external system obeys the declaration.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


class TraceAuditError(ValueError):
    """Malformed, unsupported, or mathematically invalid navigation claim."""


def _integer(x: Any, *, lower: int = 0, upper: int | None = None) -> int:
    if type(x) is not int or x < lower or (upper is not None and x > upper):
        raise TraceAuditError("Expected bounded exact integer")
    return x


def _is_prime(p: int) -> bool:
    if p == 2:
        return True
    if p < 2 or not p % 2:
        return False
    divisor = 3
    while divisor * divisor <= p:
        if p % divisor == 0:
            return False
        divisor += 2
    return True


def _matmul(left: list[list[int]], right: list[list[int]], p: int) -> list[list[int]]:
    return [[sum(x * right[k][j] for k, x in enumerate(row)) % p for j in range(len(right[0]))] for row in left]


def _matpow(base: list[list[int]], n: int, p: int) -> list[list[int]]:
    d = len(base)
    acc = [[int(i == j) for j in range(d)] for i in range(d)]
    while n:
        if n & 1:
            acc = _matmul(acc, base, p)
        n >>= 1
        if n:
            base = _matmul(base, base, p)
    return acc


def _verify_operator_witness(
    witness: Any, traces: list[list[int]], coefficients: list[int], p: int
) -> None:
    if not isinstance(witness, dict) or set(witness) != {"transition", "observers", "initial_state"}:
        raise TraceAuditError("Operator witness requires transition, observers and initial_state")
    matrix = witness["transition"]
    initial = witness["initial_state"]
    observers = witness["observers"]
    if not isinstance(matrix, list) or not 1 <= len(matrix) <= 32:
        raise TraceAuditError("Operator must be a square matrix of dimension 1..32")
    dim = len(matrix)
    if (not isinstance(initial, list) or len(initial) != dim or
        not isinstance(observers, list) or len(observers) != len(traces)):
        raise TraceAuditError("Operator dimensions disagree with trace channels")
    for row in matrix + observers:
        if not isinstance(row, list) or len(row) != dim:
            raise TraceAuditError("Operator/observer row dimension mismatch")
        for val in row:
            _integer(val, upper=p - 1)
    for val in initial:
        _integer(val, upper=p - 1)

    # Check observations independently against the declared state transition.
    state = initial[:]
    for t in range(len(traces[0])):
        for ch, row in enumerate(observers):
            if sum(x * y for x, y in zip(row, state)) % p != traces[ch][t]:
                raise TraceAuditError(f"Operator witness contradicts observation at ({ch},{t})")
        state = [sum(x * state[j] for j, x in enumerate(row)) % p for row in matrix]

    # Universal identity on the *row space*, not a finite-data fit:
    # h A^d == sum_j c_j h A^(d-j), for every observer h.
    d = len(coefficients)
    for h in observers:
        powers = [h[:]]
        for _ in range(d):
            powers.append([sum(powers[-1][j] * matrix[j][col] for j in range(dim)) % p for col in range(dim)])
        for col in range(dim):
            rhs = sum(coefficients[j - 1] * powers[d - j][col] for j in range(1, d + 1)) % p
            if rhs != powers[d][col]:
                raise TraceAuditError("Declared operator does not universally entail recurrence")


def verify_trace_navigation_bundle(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck a JSON-compatible receipt in isolation; no trust in its discovery implementation."""
    required = {"schema", "prime", "coefficients", "traces", "training_length", "coordinate",
                "values", "scope", "operator_witness", "claim_digest"}
    if not isinstance(receipt, Mapping) or set(receipt) != required:
        raise TraceAuditError("Wrong trace-navigation receipt shape")
    if receipt["schema"] != "vld.trace-navigation.v1":
        raise TraceAuditError("Unknown trace-navigation schema")
    p = _integer(receipt["prime"], lower=2, upper=2**31 - 1)
    if not _is_prime(p):
        raise TraceAuditError("Composite field modulus")
    c = receipt["coefficients"]
    traces = receipt["traces"]
    if not isinstance(c, list) or not 1 <= len(c) <= 32:
        raise TraceAuditError("Invalid coefficient degree")
    if not isinstance(traces, list) or not 1 <= len(traces) <= 32:
        raise TraceAuditError("Invalid trace channels")
    for value in c:
        _integer(value, upper=p - 1)
    if not isinstance(traces[0], list):
        raise TraceAuditError("Invalid trace")
    length = len(traces[0])
    if not 5 <= length <= 512:
        raise TraceAuditError("Trace length outside permitted range")
    for trace in traces:
        if not isinstance(trace, list) or len(trace) != length:
            raise TraceAuditError("Unaligned observer trace")
        for val in trace:
            _integer(val, upper=p - 1)
    order = len(c)
    training_length = _integer(receipt["training_length"], lower=3, upper=length - 2)
    if order >= training_length:
        raise TraceAuditError("Order cannot exceed training horizon")
    coordinate = _integer(receipt["coordinate"])
    if coordinate.bit_length() > 4096:
        raise TraceAuditError("Coordinate too large")
    for trace in traces:
        for t in range(order, length):
            if trace[t] != sum(c[j] * trace[t - j - 1] for j in range(order)) % p:
                raise TraceAuditError(f"Observation refutes claimed recurrence at coordinate {t}")

    values = receipt["values"]
    if not isinstance(values, list) or len(values) != len(traces):
        raise TraceAuditError("Projection cardinality mismatch")
    for value in values:
        _integer(value, upper=p - 1)
    # Independent companion-state exponentiation, deliberately distinct from
    # the quotient-ring polynomial powering used by the discoverer.
    companion = [[int(j == i + 1) for j in range(order)] for i in range(order - 1)]
    companion.append(list(reversed(c)))
    if coordinate >= order:
        projection = _matpow(companion, coordinate - order + 1, p)
    for ch, trace in enumerate(traces):
        predicted = trace[coordinate] if coordinate < order else (
            sum(projection[-1][i] * trace[i] for i in range(order)) % p
        )
        if predicted != values[ch]:
            raise TraceAuditError(f"Companion projection mismatch for observer {ch}")
    witness = receipt["operator_witness"]
    required_scope = "OBSERVATIONAL_HYPOTHESIS" if witness is None else "DECLARED_OPERATOR_CONDITIONAL"
    if receipt["scope"] != required_scope:
        raise TraceAuditError("Epistemic scope does not match operator evidence")
    if witness is not None:
        _verify_operator_witness(witness, traces, c, p)

    body = {key: receipt[key] for key in required - {"claim_digest"}}
    digest = hashlib.sha256(
        b"vld.trace-navigation.v1\x00" +
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if receipt["claim_digest"] != digest:
        raise TraceAuditError("Receipt digest mismatch")
    return {"status": "VERIFIED_CONDITIONAL_ON_DECLARED_OPERATOR" if witness is not None
            else "VERIFIED_OBSERVATIONAL_HYPOTHESIS", "scope": required_scope,
            "channels": len(traces), "recurrence_order": order,
            "target_coordinate": coordinate, "values": values[:], "claim_digest": digest}
