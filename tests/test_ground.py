"""Unit, boundary, and regression tests for vld.ground."""

from __future__ import annotations

import pytest

from vld.ground import (
    AffineOrbitLaw,
    ByteLens,
    LensKind,
    LensProjectionError,
    NavigationCostVector,
    UnderdeterminedGroundError,
    VirtualGroundSpec,
    certify_modular_progression,
    expected_shortcut_cost,
    propose_modular_progression,
)
from vld.triad import EpistemicStatus


def test_virtual_ground_spec_fingerprint_stability_and_validation() -> None:
    spec_a = VirtualGroundSpec(
        ground_id="vld.test.cpu",
        version="1.0.0",
        language="vld.isa.v1",
        state_domain="registers_x_memory",
        transitions=("step",),
        invariants=("pc_aligned",),
        observers=("halt_code",),
    )
    spec_b = VirtualGroundSpec(
        ground_id="vld.test.cpu",
        version="1.0.1",
        language="vld.isa.v1",
        state_domain="registers_x_memory",
        transitions=("step",),
        invariants=("pc_aligned",),
        observers=("halt_code",),
    )
    assert len(spec_a.fingerprint) == 64
    assert spec_a.fingerprint != spec_b.fingerprint

    with pytest.raises(ValueError):
        VirtualGroundSpec("", "1.0", "lang", "dom", (), (), ("obs",))
    with pytest.raises(ValueError):
        VirtualGroundSpec("id", "1.0", "lang", "dom", (), (), ())


def test_byte_lens_round_trips_and_edge_cases() -> None:
    # Empty bytes allowed for RAW_Z256 and UTF8_EXACT, rejected for UINT_LE/BE
    assert ByteLens(LensKind.RAW_Z256).lift(b"").value == ()
    assert ByteLens(LensKind.UTF8_EXACT).lift(b"").value == ""

    with pytest.raises(LensProjectionError):
        ByteLens(LensKind.UINT_LE).lift(b"")
    with pytest.raises(LensProjectionError):
        ByteLens(LensKind.UINT_BE).lift(b"")

    # Endianness distinction
    le = ByteLens(LensKind.UINT_LE, expected_length=2).lift(b"\x01\x00")
    be = ByteLens(LensKind.UINT_BE, expected_length=2).lift(b"\x01\x00")
    assert le.value == 1
    assert be.value == 256
    assert ByteLens(LensKind.UINT_LE, expected_length=2).lower(1) == b"\x01\x00"
    assert ByteLens(LensKind.UINT_BE, expected_length=2).lower(256) == b"\x01\x00"

    # Overflow & invalid UTF-8 rejections
    with pytest.raises(LensProjectionError):
        ByteLens(LensKind.UINT_LE, expected_length=1).lower(256)
    with pytest.raises(LensProjectionError):
        ByteLens(LensKind.UTF8_EXACT).lift(b"\xff\xfe")
    with pytest.raises(LensProjectionError):
        ByteLens(LensKind.RAW_Z256, expected_length=4).lift(b"abc")


@pytest.mark.parametrize(
    ("a", "b", "m", "x0"),
    [
        (1664525, 1013904223, 2**32, 73),
        (5, 7, 97, 19),
        (1, 13, 64, 0),
        (0, 42, 100, 99),
    ],
)
def test_affine_orbit_law_matches_iterative_stepping(a: int, b: int, m: int, x0: int) -> None:
    law = AffineOrbitLaw.from_parameters(a, b, m)
    cur = x0
    for step in range(75):
        receipt = law.navigate(x0, step)
        assert receipt.value == cur
        cur = (a * cur + b) % m

    # Distant jump check
    distant = law.navigate(x0, 10**18)
    assert 0 <= distant.value < m
    assert distant.binary_steps <= 60


def test_affine_orbit_law_rejects_invalid_inputs() -> None:
    law = AffineOrbitLaw(3, 1, 10)
    with pytest.raises(ValueError):
        AffineOrbitLaw(1, 0, 1)
    with pytest.raises(ValueError):
        law.navigate(-1, 5)
    with pytest.raises(ValueError):
        law.navigate(10, 5)
    with pytest.raises(ValueError):
        law.navigate(0, -1)
    with pytest.raises(ValueError):
        law.navigate(True, 1)  # type: ignore[arg-type]


def test_modular_progression_identifiability_regression() -> None:
    """Regression test preventing unwrapped progressions from certifying false moduli or int laws."""
    # True underlying system: x_{t+1} = (x_t + 17) mod 100, starting at x_0 = 20.
    # First 5 samples [20, 37, 54, 71, 88] never wrap!
    unwrapped = [20, 37, 54, 71, 88]
    proposal = propose_modular_progression(unwrapped)
    assert proposal.status is EpistemicStatus.HYPOTHESIS
    assert proposal.wrap_witnessed is False
    assert proposal.law is None

    with pytest.raises(UnderdeterminedGroundError):
        certify_modular_progression(unwrapped)

    # Adding 6th sample witnesses the wrap: 88 + 17 = 105 ≡ 5 (mod 100)!
    wrapped = [20, 37, 54, 71, 88, 5, 22]
    law = certify_modular_progression(wrapped)
    assert law.modulus == 100
    assert law.increment == 17
    assert law.navigate(20, 20).value == (20 + 20 * 17) % 100

    # Caller-declared modulus certifies unwrapped prefix when consistent, and rejects inconsistent traces.
    declared_law = certify_modular_progression(unwrapped, declared_modulus=100)
    assert declared_law.modulus == 100 and declared_law.increment == 17

    with pytest.raises(UnderdeterminedGroundError):
        certify_modular_progression([20, 37, 55], declared_modulus=100)


def test_pareto_cost_vector_dominance() -> None:
    fast_heavy = NavigationCostVector(latency_ms=2.0, compute_ops=500.0, bytes_moved=4096.0, verification_ops=10.0)
    slow_light = NavigationCostVector(latency_ms=20.0, compute_ops=100.0, bytes_moved=512.0, verification_ops=10.0)
    strictly_better = NavigationCostVector(latency_ms=1.0, compute_ops=100.0, bytes_moved=512.0, verification_ops=5.0)

    assert not fast_heavy.pareto_dominates(fast_heavy)
    assert not fast_heavy.pareto_dominates(slow_light)
    assert not slow_light.pareto_dominates(fast_heavy)
    assert strictly_better.pareto_dominates(fast_heavy)
    assert strictly_better.pareto_dominates(slow_light)

    combined = expected_shortcut_cost(
        lift_cost=NavigationCostVector(0.5, 10.0, 64.0, 1.0),
        guard_cost=NavigationCostVector(0.2, 5.0, 0.0, 1.0),
        shortcut_cost=NavigationCostVector(0.1, 20.0, 0.0, 1.0),
        exit_probability=0.0,
        fallback_cost=slow_light,
    )
    assert combined.pareto_dominates(slow_light)

    with pytest.raises(ValueError):
        expected_shortcut_cost(
            lift_cost=strictly_better,
            guard_cost=strictly_better,
            shortcut_cost=strictly_better,
            exit_probability=1.5,
            fallback_cost=slow_light,
        )
