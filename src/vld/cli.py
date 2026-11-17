r"""Command-line interface and mathematical self-verification suite for Virtual Layer Dynamics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Sequence

from vld import __version__ as PACKAGE_VERSION
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.ground import (
    AffineOrbitLaw,
    ByteLens,
    LensKind,
    NavigationCostVector,
    UnderdeterminedGroundError,
    VirtualGroundSpec,
    certify_modular_progression,
    expected_shortcut_cost,
    propose_modular_progression,
)
from vld.journal import HashChainedJournal, IdempotencyConflictError
from vld.triad import (
    EpistemicStatus,
    IndeterminateGuardTrap,
    TransitionClass,
    TruthDisposition,
    analyze_finite_ground,
    evaluate_interval_guard,
    require_determinate_branch,
)


def run_self_verification() -> dict[str, Any]:
    """Execute deterministic mathematical invariant checks for the Phase 0 kernel."""
    checks: dict[str, int] = {}

    # 1. Exhaustive Kleene K_3 algebraic law verification (3 + 9 + 27 = 39 cases)
    vals = (TruthDisposition.FALSE, TruthDisposition.INDETERMINATE, TruthDisposition.TRUE)
    k3_cases = 0
    for a in vals:
        assert ~(~a) is a
        k3_cases += 1
        for b in vals:
            assert (a & b) is (b & a)
            assert (a | b) is (b | a)
            assert ~(a & b) is ((~a) | (~b))
            assert ~(a | b) is ((~a) & (~b))
            assert (a & (a | b)) is a
            assert (a | (a & b)) is a
            k3_cases += 1
            for c in vals:
                assert ((a & b) & c) is (a & (b & c))
                assert ((a | b) | c) is (a | (b | c))
                assert (a & (b | c)) is ((a & b) | (a & c))
                k3_cases += 1

    assert evaluate_interval_guard(0.1, 0.05, 0.2) is TruthDisposition.TRUE
    assert evaluate_interval_guard(0.5, 0.05, 0.2) is TruthDisposition.FALSE
    indet = evaluate_interval_guard(0.18, 0.05, 0.2)
    assert indet is TruthDisposition.INDETERMINATE
    try:
        require_determinate_branch(indet, context="self-verify")
        raise AssertionError("Expected IndeterminateGuardTrap")
    except IndeterminateGuardTrap:
        k3_cases += 4
    checks["kleene_k3_and_guards"] = k3_cases

    # 2. Canonical serialization & domain separation
    sample = {"z": [1, -0.0, "α"], "a": {"nested": True, "empty": None}}
    roundtrip = parse_strict_json(canonical_bytes(sample))
    assert canonical_json(roundtrip) == canonical_json(sample)
    assert domain_digest("domain.a", sample) != domain_digest("domain.b", sample)
    checks["canonical_serialization"] = 2

    # 3. ByteLens invertibility over Z/256Z, UINT_LE/BE, UTF-8
    payload = "Virtual Layer Δ".encode("utf-8")
    for lens in (
        ByteLens(LensKind.RAW_Z256),
        ByteLens(LensKind.UTF8_EXACT),
        ByteLens(LensKind.UINT_LE, expected_length=len(payload)),
        ByteLens(LensKind.UINT_BE, expected_length=len(payload)),
    ):
        proj = lens.lift(payload)
        assert lens.lower(proj.value, byte_length=len(payload)) == payload
    checks["byte_lenses"] = 4

    # 4. Dual-derivation AffineOrbitLaw & identifiability guard
    vp_lcg = AffineOrbitLaw(multiplier=1664525, increment=1013904223, modulus=2**32)
    r0 = vp_lcg.navigate(73, 0)
    r1 = vp_lcg.navigate(73, 1)
    r_trillion = vp_lcg.navigate(73, 10**12)
    assert r0.value == 73
    assert r1.value == (1664525 * 73 + 1013904223) % (2**32)
    assert r_trillion.binary_steps <= 40

    unwrapped = [13, 15, 17, 19, 21]
    prop_unwrapped = propose_modular_progression(unwrapped)
    assert prop_unwrapped.status is EpistemicStatus.HYPOTHESIS
    try:
        certify_modular_progression(unwrapped)
        raise AssertionError("Unwrapped trace must not certify modulus")
    except UnderdeterminedGroundError:
        pass

    zero_dof_wrap = [7, 42, 19]
    prop_zero_dof = propose_modular_progression(zero_dof_wrap)
    assert prop_zero_dof.status is EpistemicStatus.HYPOTHESIS and prop_zero_dof.law is None
    try:
        certify_modular_progression(zero_dof_wrap)
        raise AssertionError("Zero-dof 3-sample trace must not certify modulus")
    except UnderdeterminedGroundError:
        pass

    wrapped = [90, 95, 3, 8, 13]
    certified_law = certify_modular_progression(wrapped)
    assert certified_law.modulus == 97 and certified_law.increment == 5
    checks["affine_navigation_and_identifiability"] = 8

    # 5. Finite Ground capacity, Myhill-Nerode quotient, and erasure bits
    states = ("s0", "s1", "s2", "s3")
    actions = ("permute", "collapse")
    transitions = {
        ("s0", "permute"): ("s1",),
        ("s1", "permute"): ("s0",),
        ("s2", "permute"): ("s3",),
        ("s3", "permute"): ("s2",),
        ("s0", "collapse"): ("s0",),
        ("s1", "collapse"): ("s0",),
        ("s2", "collapse"): ("s2",),
        ("s3", "collapse"): ("s2",),
    }
    observer = {"s0": "even_pair", "s1": "even_pair", "s2": "odd_pair", "s3": "odd_pair"}
    report = analyze_finite_ground(states, actions, transitions, observer)
    report.require_executable_deterministic()
    assert report.quotient_class_count == 2
    assert math.isclose(report.action_summaries["permute"].erasure_bits, 0.0)
    assert math.isclose(report.action_summaries["collapse"].erasure_bits, 1.0)
    assert report.cell_classes[("s0", "permute")] is TransitionClass.PREDICTABLE_PRESERVING
    assert report.cell_classes[("s0", "collapse")] is TransitionClass.PREDICTABLE_CONTRACTING

    # Forward Myhill-Nerode invariance under asymmetric backward preimages
    asym = analyze_finite_ground(
        ("s0", "s1", "s2"),
        ("a",),
        {("s0", "a"): ("s2",), ("s1", "a"): ("s2",), ("s2", "a"): ("s0",)},
        {"s0": "obs", "s1": "obs", "s2": "obs"},
    )
    assert asym.quotient_class_count == 1
    checks["finite_ground_triad_and_quotient"] = 6

    # 6. Pareto shortcut inequality & HashChainedJournal idempotency
    physical = NavigationCostVector(latency_ms=120.0, compute_ops=1_000_000.0, bytes_moved=65536.0, verification_ops=10.0)
    shortcut = expected_shortcut_cost(
        lift_cost=NavigationCostVector(1.0, 64.0, 256.0, 4.0),
        guard_cost=NavigationCostVector(0.5, 32.0, 0.0, 2.0),
        shortcut_cost=NavigationCostVector(0.2, 40.0, 64.0, 2.0),
        exit_probability=0.0,
        fallback_cost=physical,
    )
    assert shortcut.pareto_dominates(physical)

    spec = VirtualGroundSpec(
        ground_id="vld.verify.lcg32",
        version="0.1.0",
        language="vld.affine-modular.v1",
        state_domain="Z/2^32Z",
        transitions=("advance", "navigate"),
        invariants=("0 <= x < 2^32",),
        observers=("identity_residue",),
    )
    journal = HashChainedJournal()
    rec1, replayed1 = journal.record(
        request_id="req-001",
        ground_fingerprint=spec.fingerprint,
        operation="navigate",
        payload={"coordinate": 10**12, "x0": 73},
        result_fn=lambda: {"value": r_trillion.value},
    )
    rec2, replayed2 = journal.record(
        request_id="req-001",
        ground_fingerprint=spec.fingerprint,
        operation="navigate",
        payload={"coordinate": 10**12, "x0": 73},
        result_fn=lambda: {"value": -1},
    )
    assert not replayed1 and replayed2 and rec1 == rec2
    try:
        journal.record(
            request_id="req-001",
            ground_fingerprint=spec.fingerprint,
            operation="navigate",
            payload={"coordinate": 999, "x0": 73},
            result_fn=lambda: {"value": 0},
        )
        raise AssertionError("Expected IdempotencyConflictError")
    except IdempotencyConflictError:
        pass
    assert journal.verify() == rec1.event_digest
    checks["pareto_and_journal"] = 4

    return {
        "checks": checks,
        "ground_spec_fingerprint": spec.fingerprint,
        "journal_head_digest": journal.head_digest,
        "package": "virtual-layer-dynamics",
        "status": "PASS",
        "total_assertions": sum(checks.values()),
        "version": PACKAGE_VERSION,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vld",
        description="Virtual Layer Dynamics CLI — Ground inspection and mathematical kernel verification.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Print package architecture metadata as JSON.")
    sub.add_parser("verify", help="Run deterministic mathematical self-verification suite.")
    demo = sub.add_parser("navigate", help="Navigate VP-001 canonical 32-bit affine Ground to coordinate n.")
    demo.add_argument("--coordinate", type=int, default=10**12, help="Non-negative target coordinate n.")
    demo.add_argument("--seed", type=int, default=73, help="Initial state x0 in [0, 2^32).")

    args = parser.parse_args(argv)
    if args.command == "info":
        payload = {
            "layers": [
                "vld.canonical (deterministic JSON & domain-separated SHA-256)",
                "vld.triad (Kleene K_3 guards, epistemic classification, Myhill-Nerode quotient & erasure bits)",
                "vld.ground (VirtualGroundSpec, reversible ByteLens, dual-derivation AffineOrbitLaw, Pareto costs)",
                "vld.journal (atomic hash-chained receipt journal with idempotency fencing)",
            ],
            "package": "virtual-layer-dynamics",
            "repository": "https://github.com/Kier73/Virtual-Layer-Dynamics",
            "version": PACKAGE_VERSION,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.command == "verify":
        report = run_self_verification()
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "navigate":
        law = AffineOrbitLaw(multiplier=1664525, increment=1013904223, modulus=2**32)
        try:
            receipt = law.navigate(args.seed, args.coordinate)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "affine_increment": receipt.affine_increment,
                    "affine_multiplier": receipt.affine_multiplier,
                    "binary_steps": receipt.binary_steps,
                    "coordinate": receipt.coordinate,
                    "initial_state": receipt.initial_state,
                    "law_digest": receipt.law_digest,
                    "value": receipt.value,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
