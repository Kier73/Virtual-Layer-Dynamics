"""Unit and mathematical invariant tests for vld.triad."""

from __future__ import annotations

import math
import pytest

from vld.triad import (
    IncompleteGroundError,
    IndeterminateGuardTrap,
    TransitionClass,
    TruthDisposition,
    analyze_finite_ground,
    evaluate_interval_guard,
    require_determinate_branch,
)


def test_kleene_k3_truth_tables() -> None:
    T, F, I = TruthDisposition.TRUE, TruthDisposition.FALSE, TruthDisposition.INDETERMINATE
    assert (~T, ~F, ~I) == (F, T, I)
    assert (F & I) is F
    assert (T & I) is I
    assert (T | I) is T
    assert (F | I) is I
    assert (I & I) is I
    assert (I | I) is I
    with pytest.raises(TypeError):
        _ = T & True  # type: ignore[operator]
    with pytest.raises(TypeError):
        _ = F | "false"  # type: ignore[operator]


def test_interval_guard_boundaries_and_validation() -> None:
    # Exact boundary U == tau is TRUE
    assert evaluate_interval_guard(0.15, 0.05, 0.20) is TruthDisposition.TRUE
    # L == tau means [tau, U] touches tau -> INDETERMINATE (strict > required for FALSE)
    assert evaluate_interval_guard(0.25, 0.05, 0.20) is TruthDisposition.INDETERMINATE
    # Strictly outside: L = 0.21 > 0.20 -> FALSE
    assert evaluate_interval_guard(0.26, 0.05, 0.20) is TruthDisposition.FALSE

    assert require_determinate_branch(TruthDisposition.TRUE) is True
    assert require_determinate_branch(TruthDisposition.FALSE) is False
    with pytest.raises(IndeterminateGuardTrap):
        require_determinate_branch(TruthDisposition.INDETERMINATE, context="unit-test")

    with pytest.raises(ValueError):
        evaluate_interval_guard(-0.1, 0.0, 0.1)
    with pytest.raises(ValueError):
        evaluate_interval_guard(4.0, 0.0, 0.1)
    with pytest.raises(ValueError):
        evaluate_interval_guard(0.1, math.nan, 0.1)


def test_asymmetric_contraction_erasure_and_multi_step_quotient() -> None:
    # 4 states: s0, s1, s2 map to s0 under 'contract'; s3 maps to s3.
    # Under 'shift': s0->s1->s2->s3->s0 (bijection).
    # Immediate observer only sees s3 vs {s0, s1, s2}, so multi-step refinement must split s0, s1, s2!
    states = ("s0", "s1", "s2", "s3")
    actions = ("shift", "contract")
    transitions = {
        ("s0", "shift"): ("s1",),
        ("s1", "shift"): ("s2",),
        ("s2", "shift"): ("s3",),
        ("s3", "shift"): ("s0",),
        ("s0", "contract"): ("s0",),
        ("s1", "contract"): ("s0",),
        ("s2", "contract"): ("s0",),
        ("s3", "contract"): ("s3",),
    }
    observer = {"s0": "low", "s1": "low", "s2": "low", "s3": "high"}
    report = analyze_finite_ground(states, actions, transitions, observer, initial_states=("s0",))

    assert report.reachable_count == 4
    # Because shift^3(s0)=s3, shift^2(s1)=s3, shift^1(s2)=s3, all 4 states are distinguishable!
    assert report.quotient_class_count == 4
    assert math.isclose(report.compression_ratio, 1.0)

    # Shift is bijective -> 0 bits erased
    assert math.isclose(report.action_summaries["shift"].erasure_bits, 0.0)
    # Contract has preimages of size 3 (for s0) and size 1 (for s3): (3/4)*log2(3) + (1/4)*log2(1)
    expected_erasure = 0.75 * math.log2(3.0)
    assert math.isclose(report.action_summaries["contract"].erasure_bits, expected_erasure, rel_tol=1e-12)
    assert report.cell_classes[("s0", "contract")] is TransitionClass.PREDICTABLE_CONTRACTING
    assert report.cell_classes[("s3", "contract")] is TransitionClass.PREDICTABLE_PRESERVING


def test_unreachable_states_unknown_ambiguous_and_external_couplings() -> None:
    states = ("a", "b", "island")
    actions = ("step", "sense")
    transitions = {
        ("a", "step"): ("b",),
        ("b", "step"): ("a", "b"),  # ambiguous!
        # ("island", "step") omitted -> UNKNOWN!
        ("a", "sense"): ("a", "b"),
        ("b", "sense"): ("b",),
        ("island", "sense"): ("island",),
    }
    observer = {"a": "0", "b": "1", "island": "0"}
    report = analyze_finite_ground(
        states,
        actions,
        transitions,
        observer,
        initial_states=("a",),
        external_couplings=(("a", "sense"),),
    )

    assert report.reachable_count == 2
    assert report.cell_classes[("b", "step")] is TransitionClass.AMBIGUOUS
    assert report.cell_classes[("island", "step")] is TransitionClass.UNKNOWN
    assert report.cell_classes[("a", "sense")] is TransitionClass.INFORMATION_PRODUCING
    assert not report.is_executable_deterministic

    with pytest.raises(IncompleteGroundError):
        report.require_executable_deterministic()


def test_analyze_finite_ground_input_validation() -> None:
    with pytest.raises(ValueError, match="at least one state"):
        analyze_finite_ground((), ("a",), {}, {})
    with pytest.raises(ValueError, match="at least one action"):
        analyze_finite_ground(("s0",), (), {}, {"s0": "o"})
    with pytest.raises(ValueError, match="Observer projection missing"):
        analyze_finite_ground(("s0",), ("a",), {}, {})
    with pytest.raises(ValueError, match="real number"):
        evaluate_interval_guard(True, 0.0, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one reachable target"):
        analyze_finite_ground(("s0",), ("a",), {}, {"s0": "o"}, external_couplings=(("s0", "a"),))


def test_myhill_nerode_forward_invariance_under_backward_contraction_and_relational_preimage() -> None:
    # Regression 1: backward preimage asymmetry (contracting vs preserving) MUST NOT split
    # forward-indistinguishable states under Moore / Myhill-Nerode equivalence ~_P.
    # Both u0 (colliding with u0_twin at u1) and v0 (injective into v1) emit "0", "0", "1", "1", ... forever!
    states = ("u0", "u0_twin", "u1", "v0", "v1", "sink")
    actions = ("step",)
    transitions = {
        ("u0", "step"): ("u1",),
        ("u0_twin", "step"): ("u1",),
        ("u1", "step"): ("sink",),
        ("v0", "step"): ("v1",),
        ("v1", "step"): ("sink",),
        ("sink", "step"): ("sink",),
    }
    observer = {"u0": "0", "u0_twin": "0", "u1": "0", "v0": "0", "v1": "0", "sink": "1"}
    report = analyze_finite_ground(states, actions, transitions, observer)
    assert report.cell_classes[("u0", "step")] is TransitionClass.PREDICTABLE_CONTRACTING
    assert report.cell_classes[("v0", "step")] is TransitionClass.PREDICTABLE_PRESERVING
    # Exact 3 forward equivalence classes: distance-2 to sink {u0, u0_twin, v0}, distance-1 {u1, v1}, and {sink}
    assert report.quotient_classes == (("sink",), ("u0", "u0_twin", "v0"), ("u1", "v1"))

    # Regression 2: full relational preimage counts collisions from ambiguous/external cells too.
    mixed = analyze_finite_ground(
        states=("s0", "s1"),
        actions=("a",),
        transitions={("s0", "a"): ("s0",), ("s1", "a"): ("s0", "s1")},
        observer={"s0": "0", "s1": "1"},
    )
    assert mixed.cell_classes[("s0", "a")] is TransitionClass.PREDICTABLE_CONTRACTING
