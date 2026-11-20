from __future__ import annotations

from copy import deepcopy
import random

import pytest

from vld.navigation.trace_law import (
    TraceLawError,
    TraceLawRefuted,
    TraceLawUnderdetermined,
    discover_trace_law,
)
from vld.independent_trace_audit import TraceAuditError, verify_trace_navigation_bundle


def fibonacci_observers(p: int, length: int):
    a, b = 1, 0
    ch1, ch2 = [], []
    for _ in range(length):
        ch1.append(a)
        ch2.append(b)
        a, b = (a + b) % p, a
    witness = {"transition": [[1, 1], [1, 0]], "observers": [[1, 0], [0, 1]], "initial_state": [1, 0]}
    return [ch1, ch2], witness


def test_multi_observer_identifies_law_from_three_training_frames_and_jumps_centuries():
    traces, witness = fibonacci_observers(65521, 13)
    law = discover_trace_law(traces, prime=65521, training_length=3)
    assert law.coefficients == (1, 1)
    with pytest.raises(TraceLawUnderdetermined):
        discover_trace_law([traces[0]], prime=65521, training_length=3)

    coordinate = 10**60 + 17
    receipt = law.navigate(coordinate, operator_witness=witness)
    independently = verify_trace_navigation_bundle(receipt)
    assert independently["status"] == "VERIFIED_CONDITIONAL_ON_DECLARED_OPERATOR"
    assert independently["target_coordinate"] == coordinate
    assert independently["values"] == [17164, 51927]
    assert law.law_digest
    assert len(str(coordinate)) > 60


def test_without_operator_witness_remains_observational_not_globally_proven():
    traces, witness = fibonacci_observers(101, 12)
    law = discover_trace_law(traces, prime=101, training_length=5)
    projection = law.navigate(10**34)
    assert verify_trace_navigation_bundle(projection)["status"] == "VERIFIED_OBSERVATIONAL_HYPOTHESIS"
    assert projection["operator_witness"] is None
    witness["transition"][0][0] = 3
    with pytest.raises(TraceAuditError, match="contradicts|universally"):
        law.navigate(10**34, operator_witness=witness)


def test_heldout_counterexample_is_refused_even_after_excellent_training_fit():
    traces, _ = fibonacci_observers(101, 18)
    altered = [t[:] for t in traces]
    altered[1][-1] = (altered[1][-1] + 1) % 101
    with pytest.raises(TraceLawRefuted, match="counterexample"):
        discover_trace_law(altered, prime=101, training_length=11)


def test_adversarial_receipt_mutations_are_rejected():
    traces, witness = fibonacci_observers(101, 13)
    receipt = discover_trace_law(traces, prime=101, training_length=6).navigate(10**25, operator_witness=witness)
    for key, bad in [
        ("coordinate", 10**25 + 1),
        ("prime", 99),
        ("scope", "GLOBAL_CERTIFIED"),
        ("values", [(receipt["values"][0] + 1) % 101, receipt["values"][1]]),
        ("coefficients", [1, 2]),
        ("training_length", True),
    ]:
        mutant = deepcopy(receipt)
        mutant[key] = bad
        with pytest.raises(TraceAuditError):
            verify_trace_navigation_bundle(mutant)
    mutant = deepcopy(receipt)
    mutant["operator_witness"]["observers"][0][0] = 2
    with pytest.raises(TraceAuditError):
        verify_trace_navigation_bundle(mutant)


def test_invalid_fields_underdetermined_traces_and_boolean_rejection():
    traces, _ = fibonacci_observers(11, 8)
    with pytest.raises(TraceLawError, match="prime"):
        discover_trace_law(traces, prime=12, training_length=4)
    with pytest.raises(TraceLawError, match="Trace entries"):
        discover_trace_law([[True, 0, 0, 0, 0]], prime=11, training_length=3)
    with pytest.raises(TraceLawUnderdetermined):
        discover_trace_law([[0] * 9], prime=11, training_length=4)
    with pytest.raises(TraceLawError, match="held-out"):
        discover_trace_law(traces, prime=11, training_length=7)
    law = discover_trace_law(traces, prime=11, training_length=4)
    with pytest.raises(TraceLawError, match="coordinate"):
        law.navigate(-1)
    with pytest.raises(TraceLawError, match="coordinate"):
        law.navigate(True)


def test_random_prime_field_linear_operators_agree_with_explicit_execution():
    random.seed(29)
    p = 97
    for dim in (2, 3, 4):
        for trial in range(5):
            A = [[random.randrange(p) for _ in range(dim)] for _ in range(dim)]
            observers = [[int(i == j) for j in range(dim)] for i in range(dim)]
            state = [random.randrange(p) for _ in range(dim)]
            initial = state[:]
            traces = [[] for _ in range(dim)]
            for _ in range(25):
                for ch in range(dim):
                    traces[ch].append(state[ch])
                state = [sum(A[i][j] * state[j] for j in range(dim)) % p for i in range(dim)]
            law = discover_trace_law(traces, prime=p, training_length=14, max_order=dim)
            witness = {"transition": A, "observers": observers, "initial_state": initial}
            for coordinate in (0, 1, dim - 1, 14, 24, 1000, 10**20):
                receipt = law.navigate(coordinate, operator_witness=witness)
                assert verify_trace_navigation_bundle(receipt)["status"] == "VERIFIED_CONDITIONAL_ON_DECLARED_OPERATOR"
                if coordinate < 25:
                    assert receipt["values"] == [trace[coordinate] for trace in traces]
