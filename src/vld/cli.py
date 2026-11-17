r"""Command-line interface, invariant verifier, Pareto benchmarker, and Vantage Server for Virtual Layer Dynamics."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import sys
from typing import Any, Sequence

from vld import __version__ as PACKAGE_VERSION
from vld.accel import benchmark_virtual_vs_traditional, get_native_accelerator, has_numpy
from vld.agents import (
    HypervisorRefusalError,
    InLayerModelAgent,
    ReceiverHypervisor,
    create_default_exploration_grant,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.fabric import VLDHttpVantageServer, VLDVantageNode
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
from vld.memory import InMemoryPageBacking, PagedObjectStore, StructuralMemoryRegistry, certify_sequence_law
from vld.models import (
    TransformerArchitectureSpec,
    compile_piecewise_affine_relu,
    route_moe_topk,
    verify_compiled_capsule,
)
from vld.signal import verify_differential_backends, verify_signal8_ground
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
    """Execute deterministic mathematical invariant checks across Phases 0, 1, 2, 6, 7 & Accel."""
    checks: dict[str, int] = {}

    # 1. Exhaustive Kleene K_3 algebraic law verification
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

    # 2. Canonical serialization & ByteLens invertibility
    sample = {"z": [1, -0.0, "α"], "a": {"nested": True, "empty": None}}
    roundtrip = parse_strict_json(canonical_bytes(sample))
    assert canonical_json(roundtrip) == canonical_json(sample)
    assert domain_digest("domain.a", sample) != domain_digest("domain.b", sample)
    payload = "Virtual Layer Δ".encode("utf-8")
    for lens in (
        ByteLens(LensKind.RAW_Z256),
        ByteLens(LensKind.UTF8_EXACT),
        ByteLens(LensKind.UINT_LE, expected_length=len(payload)),
        ByteLens(LensKind.UINT_BE, expected_length=len(payload)),
    ):
        proj = lens.lift(payload)
        assert lens.lower(proj.value, byte_length=len(payload)) == payload
    checks["canonical_and_byte_lenses"] = 6

    # 3. Dual-derivation AffineOrbitLaw & identifiability dof guard
    vp_lcg = AffineOrbitLaw(multiplier=1664525, increment=1013904223, modulus=2**32)
    r_trillion = vp_lcg.navigate(73, 10**12)
    assert r_trillion.binary_steps <= 40
    for bad_trace in ([13, 15, 17, 19, 21], [7, 42, 19]):
        assert propose_modular_progression(bad_trace).status is EpistemicStatus.HYPOTHESIS
        try:
            certify_modular_progression(bad_trace)
            raise AssertionError("Underdetermined trace must not certify")
        except UnderdeterminedGroundError:
            pass
    certified_law = certify_modular_progression([90, 95, 3, 8, 13])
    assert certified_law.modulus == 97 and certified_law.increment == 5
    checks["affine_navigation_and_identifiability"] = 6

    # 4. Finite Ground Myhill-Nerode quotient & forward invariance
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
    checks["finite_ground_triad_and_quotient"] = 4

    # 5. Phase 1: Exhaustive 4,772-case Signal8 + 512-case tri-backend differential verification
    sig8_report = verify_signal8_ground()
    assert sig8_report["status"] == "PASS" and sig8_report["total_cases"] == 4772
    diff_report = verify_differential_backends()
    assert diff_report["status"] == "PASS" and diff_report["cases_checked"] == 512
    checks["phase1_signal8_and_differential_backends"] = sig8_report["total_cases"] + diff_report["cases_checked"]

    # 6. Phase 2: M1 Paged Store, 2-adic LCG inverse lookup & counterexample invalidation
    m1_backing = InMemoryPageBacking(page_size=64)
    m1_store = PagedObjectStore(m1_backing)
    m1_manifest = m1_store.put_bytes(bytes(i % 256 for i in range(300)))
    m1_backing.pages_read = 0
    assert len(m1_store.read_range(m1_manifest, 70, 50)) == 50 and m1_backing.pages_read == 1

    lcg_seq = [73]
    for _ in range(6):
        lcg_seq.append((1664525 * lcg_seq[-1] + 1013904223) % (2**32))
    lcg_cert = certify_sequence_law(lcg_seq, declared_modulus=2**32)
    assert lcg_cert.locate_value(lcg_cert.evaluate(123_456_789)) == 123_456_789

    sm_reg = StructuralMemoryRegistry()
    sm_reg.certify_and_admit("verify.mem", [1, 4, 7, 2, 5], domain="Z/8Z", declared_modulus=8)
    _, late_rcpt = sm_reg.read("verify.mem", 100)
    sm_reg.inject_counterexample("verify.mem", 10, 0)
    assert sm_reg.verify_read_receipt(late_rcpt)[0] is False
    checks["phase2_m1_and_structural_memory"] = 5

    # 7. Phases 6 & 7: MoE routing, SNC whole-domain ReLU compilation & Hypervisor-governed Model Agent
    moe_spec = TransformerArchitectureSpec.from_published_profile("qwen3-235b-a22b")
    moe_route = route_moe_topk(
        moe_spec,
        layer_index=0,
        token_position=0,
        router_logits=[float(i % 17) * 0.1 for i in range(128)],
    )
    assert len(moe_route.selected_experts) == 8 and math.isclose(sum(moe_route.normalized_weights), 1.0, abs_tol=1e-9)

    capsule = compile_piecewise_affine_relu(
        [
            {"left": "-inf", "right": "0", "affines": [["-2", "1"]]},
            {"left": "0", "right": "3", "affines": [["1", "1"]]},
            {"left": "3", "right": "+inf", "affines": [["0", "4"]]},
        ]
    )
    assert verify_compiled_capsule(capsule)["status"] == "VERIFIED_WHOLE_DOMAIN"

    hyp = ReceiverHypervisor()
    hyp.register_agent(create_default_exploration_grant("verify-agent", probe_budget=Fraction(10)))
    agent = InLayerModelAgent("verify-agent", hyp)
    ag_rep = agent.explore_certify_and_navigate(lambda t: (40 + 13 * t) % 47, memory_id="agent.v47", target_coordinate=10**9)
    assert ag_rep.navigated_value == (40 + 13 * (10**9)) % 47
    try:
        hyp.propose_structural_law("verify-agent", memory_id="agent.unwrapped", samples=[2, 4, 6, 8])
        raise AssertionError("Expected HypervisorRefusalError")
    except HypervisorRefusalError:
        pass
    checks["phases6_7_models_snc_and_agent_hypervisor"] = 6

    # 8. Pareto shortcut inequality & HashChainedJournal idempotency
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
        version="0.2.0",
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
        "accelerators": {
            "native_rust_cdll": get_native_accelerator().available,
            "numpy_simd": has_numpy(),
        },
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
        description="Virtual Layer Dynamics CLI — Ground verification, navigation, Pareto benchmarking & Vantage hosting.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Print package architecture and active acceleration metadata as JSON.")
    sub.add_parser("verify", help="Run deterministic mathematical self-verification suite across all layers.")

    demo = sub.add_parser("navigate", help="Navigate VP-001 canonical 32-bit affine Ground to coordinate n.")
    demo.add_argument("--coordinate", type=int, default=10**12, help="Non-negative target coordinate n.")
    demo.add_argument("--seed", type=int, default=73, help="Initial state x0 in [0, 2^32).")

    bench = sub.add_parser("benchmark", help="Benchmark Virtual Layer structural contraction & horizon leaping vs traditional compute.")
    bench.add_argument("--dim", type=int, default=1024, help="Matrix dimension M=N for StructuralLinearOperator.")
    bench.add_argument("--steps", type=int, default=25000, help="Horizon T for LinearRecurrenceGround.")

    serve = sub.add_parser("serve", help="Host a live HTTP/1.1 Information-Flow Vantage Node.")
    serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8730, help="Bind port (default: 8730).")
    serve.add_argument("--node-id", type=str, default="vld.vantage.local", help="Logical receiver node ID.")

    args = parser.parse_args(argv)
    if args.command == "info":
        payload = {
            "accelerators": {
                "native_rust_cdll": get_native_accelerator().available,
                "numpy_simd": has_numpy(),
            },
            "layers": [
                "vld.canonical (deterministic JSON & domain-separated SHA-256)",
                "vld.triad (Kleene K_3 guards, epistemic classification, Myhill-Nerode quotient & erasure bits)",
                "vld.ground (VirtualGroundSpec, reversible ByteLens, dual-derivation AffineOrbitLaw, Pareto costs)",
                "vld.journal (multi-process atomic hash-chained receipt journal with idempotency fencing)",
                "vld.signal (Signal8, ModularRing & ExactRational backends, static footprints, base-8 compiler)",
                "vld.memory (M1 PagedObjectStore, 5-rung Law Certifier, 2-adic inverse locators, counterexample demotion)",
                "vld.inquiry (greedy, beam & exact minimax active perception planners, HypothesisRevisionLedger)",
                "vld.accel (StructuralLinearOperator contraction, LinearRecurrenceGround, NumPy + Rust CDLL kernels)",
                "vld.fabric (VantageEnvelope multi-framing codec, StreamPipeVantage, HTTP Range Vantage Server/Client)",
                "vld.models (Safetensors HFR, BPE Tokenizer Ground, Qwen3 dense/MoE realizers, SNC/PNF procedural models)",
                "vld.agents (ProofGround hypergraph, ReceiverHypervisor CAS/capabilities, InLayerModelAgent)",
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

    if args.command == "benchmark":
        if args.dim <= 0 or args.steps <= 0:
            parser.error("--dim and --steps must be positive integers.")
        result = benchmark_virtual_vs_traditional(matrix_dim=args.dim, recurrence_steps=args.steps)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "serve":
        node = VLDVantageNode(args.node_id)
        server = VLDHttpVantageServer(node, host=args.host, port=args.port)
        print(json.dumps({"base_url": server.base_url, "node_id": node.node_id, "status": "LISTENING"}, sort_keys=True), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.close()
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
