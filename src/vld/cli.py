r"""Command-line interface, invariant verifier, Pareto benchmarker, and Vantage Server for Virtual Layer Dynamics."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import sys
from typing import Any, Sequence

from vld import __version__ as PACKAGE_VERSION
from vld.accel import (
    NumericEngine,
    benchmark_virtual_vs_traditional,
    decode_f8_e4m3,
    decode_f8_e8m0,
    dequantize_fp4_e2m1_packed,
    dequantize_fp8_blockwise,
    encode_f8_e4m3,
    encode_f8_e8m0,
    encode_fp4_e2m1_packed,
    get_native_accelerator,
    has_numpy,
    hc_split_sinkhorn,
)
from vld.agents import (
    HypervisorRefusalError,
    InLayerModelAgent,
    ReceiverHypervisor,
    create_default_exploration_grant,
)
from vld.authority import (
    CausalOrder,
    CrossSchedulerComparisonError,
    QuorumAuthorityMember,
    QuorumEquivocationError,
    QuorumLineageCoordinator,
    QuorumMembership,
    QuorumProposal,
    RollbackDetectedError,
    TemporalCoordinate,
    VectorClock,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.continuous import (
    ContinuousStageSpec,
    ContinuousVirtualSystem,
    build_compact_deepseek_v41_fixture,
    build_compact_glm53_fixture,
    run_self_improving_continuous_campaign,
)
from vld.fabric import (
    ComputationalFabricRouter,
    FabricBlockerCode,
    FabricLink,
    FabricResolutionStatus,
    HostAdvertisement,
    LossyInMemoryLink,
    ReliableChannel,
    ReplayRejectedError,
    ReplayWindowGuard,
    RouteMetric,
    VLDHttpVantageServer,
    VLDVantageNode,
)
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
from vld.memory import DiskPageBacking, InMemoryPageBacking, PagedObjectStore, StructuralMemoryRegistry, certify_sequence_law
from vld.models import (
    AffineKernelAttentionGround,
    DeepSeekV41LayerRealizer,
    DeepSeekV41ReferenceOracle,
    Glm53LayerRealizer,
    HostedModelGlance,
    PUBLISHED_FRONTIER_PROFILES,
    TransformerArchitectureSpec,
    compile_multivariate_hinge_relu,
    compile_piecewise_affine_relu,
    direct_affine_kernel_oracle,
    glance_huggingface_model,
    route_moe_topk,
    verify_compiled_capsule,
    verify_mla_matrix_absorption,
    verify_multivariate_capsule,
    verify_softmax_insufficiency_witness,
)
from vld.navigation import (
    AttributedObserverReading,
    ComputationNode,
    CrossObserverCongruenceGate,
    FiniteFieldOperatorChain,
    IntervalBound,
    NavigableComputationDAG,
    SubgraphThunk,
    analyze_dag_intervals,
    check_strong_bisimulation,
    discover_similarity_blocks,
)
from vld.signal import Signal8Backend, verify_differential_backends, verify_signal8_ground
from vld.substrates import (
    BarrierDivergenceError,
    HeadlessOS,
    InternetPathRefused,
    O_RDONLY,
    PF_R,
    PF_W,
    PF_X,
    SEEK_SET,
    SIMTInstruction,
    SIMTWorkgroupExecutor,
    TTL_PROBES,
    VMInstruction,
    VirtualPageSpace,
    WasmRealizer,
    assemble_wasm_affine_orbit_module,
    assemble_wasm_f32_dot_indirect_module,
    build_ethernet_ipv4_udp_frame,
    build_minimal_elf64_image,
    build_room_intention,
    execute_room_on_realizer,
    lift_elf64_image,
    lift_wasm_module,
    verify_ipv4_router_hop_ground,
)
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
    _ = m1_store.put_bytes(b"ephemeral-unpinned" * 5)
    m1_store.pin_manifest(m1_manifest, lease_id="verify-root")
    gc_rep = m1_store.collect_garbage()
    assert gc_rep.pinned_manifest_count == 1 and gc_rep.reclaimed_manifest_count == 1 and m1_store.verify(m1_manifest)

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
    wrap_poly = certify_sequence_law([(t * t + 3 * t + 5) % 97 for t in range(12)], declared_modulus=97)
    assert wrap_poly.family == "poly" and wrap_poly.evaluate(10_000) == (10_000 * 10_000 + 30_005) % 97
    checks["phase2_m1_and_structural_memory"] = 7

    # 7. Phase 3: Scheduler-scoped vector clocks (G13) & strict-majority (2q > N) quorum fencing
    tc_a = TemporalCoordinate("sched.verify.a", 1, VectorClock.from_mapping({"p0": 1}))
    tc_b = tc_a.advance("p0", 2)
    assert tc_a.compare(tc_b) is CausalOrder.BEFORE
    try:
        tc_a.compare(TemporalCoordinate("sched.verify.b", 1, VectorClock.from_mapping({"p0": 1})))
        raise AssertionError("Expected CrossSchedulerComparisonError")
    except CrossSchedulerComparisonError:
        pass

    q_mem = QuorumMembership("verify.lineage", ("m0", "m1", "m2"))
    q_secs = {m: f"verify-secret-{m}-0123456789".encode("utf-8") for m in q_mem.member_ids}
    q_nodes = [QuorumAuthorityMember(m, q_secs[m]) for m in q_mem.member_ids]
    q_coord = QuorumLineageCoordinator(q_mem, q_nodes, q_secs)
    s1_dg = domain_digest("verify.m1", "s1")
    s2_dg = domain_digest("verify.m1", "s2")
    q_coord.propose_and_seal(
        slot=1,
        term=1,
        predecessor_manifest_digest="0" * 64,
        state_manifest_digest=s1_dg,
        proposer_id="h0",
        operation_id="init",
        active_member_ids=("m0", "m1"),
    )
    try:
        q_nodes[0].vote(
            QuorumProposal(q_mem.lineage_id, q_mem.membership_digest, 1, 1, "0" * 64, s2_dg, "fork", "fork")
        )
        raise AssertionError("Expected QuorumEquivocationError")
    except QuorumEquivocationError:
        pass
    q_coord.propose_and_seal(
        slot=2,
        term=2,
        predecessor_manifest_digest=s1_dg,
        state_manifest_digest=s2_dg,
        proposer_id="h1",
        operation_id="step",
        active_member_ids=("m1", "m2"),
    )
    try:
        q_coord.verify_fresh_checkpoint(slot=1, state_manifest_digest=s1_dg)
        raise AssertionError("Expected RollbackDetectedError")
    except RollbackDetectedError:
        pass
    checks["phase3_temporal_and_quorum_fencing"] = 6

    # 8. Phase 4: Lazy NavigableComputationDAG, interval & bisimulation gates, ReliableChannel & replay guard
    v_ctx = domain_digest("verify.dag", "ctx")
    v_dag = NavigableComputationDAG("verify.dag")
    v_dag.add_node(ComputationNode("in_x", 1, "const", "/r", v_ctx, (), {"value": "2"}))
    v_dag.add_subgraph_thunk(
        SubgraphThunk(
            "thunk.1",
            ("in_x",),
            domain_digest("verify.thunk", "t1"),
            lambda _c: [ComputationNode("aff_y", 1, "affine", "/r", v_ctx, ("in_x",), {"bias": "1", "scale": "3"})],
        )
    )
    v_dag.commit_result("in_x", 2)
    assert len(v_dag.ready_frontier()) == 1
    ig_rep = analyze_dag_intervals(
        v_dag,
        input_bounds={"in_x": IntervalBound(Fraction(1), Fraction(3))},
        safety_contracts={"aff_y": IntervalBound(Fraction(4), Fraction(10))},
    )
    assert ig_rep.overall_disposition is TruthDisposition.TRUE
    bisim = check_strong_bisimulation(
        actions=("a",),
        left_states=("s0", "s1"),
        left_transitions={("s0", "a"): ("s1",), ("s1", "a"): ("s0",)},
        left_observations={"s0": 0, "s1": 0},
        left_initial="s0",
        right_states=("q0",),
        right_transitions={("q0", "a"): ("q0",)},
        right_observations={"q0": 0},
        right_initial="q0",
    )
    assert bisim.equivalent is True

    rw = ReplayWindowGuard(16)
    rw.check_and_record("peer0", 5)
    try:
        rw.check_and_record("peer0", 5)
        raise AssertionError("Expected ReplayRejectedError")
    except ReplayRejectedError:
        pass
    link = LossyInMemoryLink()
    ch_s = ReliableChannel(channel_id="v.ch", local_peer="a", remote_peer="b", send_fn=link.send)
    ch_r = ReliableChannel(channel_id="v.ch", local_peer="b", remote_peer="a", send_fn=link.send)
    ch_s.send_operation("op.1", {"k": 7})
    deliv = ch_r.receive_frame(link.pop_for("b"))  # type: ignore[arg-type]
    assert len(deliv) == 1 and deliv[0][1] == "op.1"
    checks["phase4_navigation_gates_and_fabric_reliability"] = 6

    # 9. Phase 5: CoW VirtualPageSpace, ELF64 & WASM lifting, SIMT barrier guard & Portable ROOM parity
    vps = VirtualPageSpace(page_size=64)
    vps.map_region(0, 64, flags=PF_R | PF_W, initial_bytes=b"ORIG")
    vps_cow = vps.fork_cow()
    vps_cow.write_bytes(0, b"FORK")
    assert vps.read_bytes(0, 4) == b"ORIG" and vps_cow.read_bytes(0, 4) == b"FORK"

    elf_img = build_minimal_elf64_image([(0x401000, PF_R | PF_X, b"\x90\xc3", 16)])
    elf_g, _ = lift_elf64_image(elf_img)
    assert elf_g.entry_point == 0x401000

    wasm_mod = lift_wasm_module(assemble_wasm_affine_orbit_module())
    room_intent = build_room_intention(coordinate=1_000_000_007)
    alg_val = execute_room_on_realizer(room_intent, "algebraic-orbit")
    vm_val = execute_room_on_realizer(room_intent, "vm-kernel")
    (wasm_val,) = WasmRealizer(wasm_mod).invoke("affine_jump", (17, 29, 65521, 5, 1_000_000_007))
    assert alg_val == vm_val == wasm_val

    wasm_f32_runner = WasmRealizer(lift_wasm_module(assemble_wasm_f32_dot_indirect_module()))
    (grow_res,) = wasm_f32_runner.invoke("grow_and_size", (1,))
    assert (grow_res >> 16) == 1 and (grow_res & 0xFFFF) == 2

    try:
        SIMTWorkgroupExecutor(workgroup_size=2).execute_kernel(
            (SIMTInstruction("JZ_EVEN_LANE", ("r0", 2)), SIMTInstruction("BARRIER", ("b0",)), SIMTInstruction("HALT", ())),
            [0.0, 0.0],
        )
        raise AssertionError("Expected BarrierDivergenceError")
    except BarrierDivergenceError:
        pass
    checks["phase5_substrates_cow_elf_wasm_simt_and_rooms"] = 8

    # 10. Phases 6 & 7: MoE routing, 1D + R^d SNC whole-domain ReLU compilation & Hypervisor-governed Model Agent
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
    mv_capsule = compile_multivariate_hinge_relu(
        input_dim=2,
        base_linear=[["1", "-1"]],
        base_bias=["0"],
        hinges=[{"normal": ["1", "1"], "offset": "-1", "weights": ["2"]}],
    )
    assert verify_multivariate_capsule(mv_capsule)["status"] == "VERIFIED_WHOLE_DOMAIN"
    hi_capsule = compile_multivariate_hinge_relu(
        input_dim=16,
        base_linear=[[str(i % 3) for i in range(16)]],
        base_bias=["1"],
        hinges=[
            {"normal": ["1" if j == (k % 16) else ("-1" if j == ((k + 3) % 16) else "0") for j in range(16)], "offset": str(k - 4), "weights": ["1"]}
            for k in range(20)
        ],
    )
    assert verify_multivariate_capsule(hi_capsule, default_chamber_limit=16)["verification_method"] == "algebraic_ridge_exact"

    aks_g = AffineKernelAttentionGround(modulus_p=257, key_dim=2, value_dim=2)
    aks_g.admit_token([3, 5], [7, 11])
    aks_g.admit_token([13, 17], [19, 23])
    assert aks_g.query([2, 9]).value == direct_affine_kernel_oracle(257, ([3, 5], [13, 17]), ([7, 11], [19, 23]), [2, 9])
    assert verify_softmax_insufficiency_witness()["identical_affine_sufficient_state"] is True

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
    checks["phases6_7_models_snc_and_agent_hypervisor"] = 10

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

    # 11. Phase 8: FP8/UE8M0/FP4 codecs, Birkhoff mHC, DeepSeek-V4.1 & GLM-5.3 MLA Absorption
    f8_vals = [0.0, -0.5, 1.5, -12.0, 448.0]
    assert decode_f8_e4m3(encode_f8_e4m3(f8_vals)) == f8_vals
    assert decode_f8_e8m0(encode_f8_e8m0([0.125, 1.0, 8.0])) == [0.125, 1.0, 8.0]
    fp4_row = [[0.0, 0.5, -1.0, 1.5, -2.0, 3.0, -4.0, 6.0] * 4]
    fp4_p, fp4_s = encode_fp4_e2m1_packed(fp4_row, block_k=32)
    assert dequantize_fp4_e2m1_packed(fp4_p, (1, 32), fp4_s, (1, 1), use_numpy=False) == fp4_row
    b_raw = dequantize_fp8_blockwise(encode_f8_e4m3([2.0] * (32 * 32)), (32, 32), encode_f8_e8m0([0.5]), (1, 1), use_numpy=False)
    assert b_raw[0][0] == 1.0 and b_raw[31][31] == 1.0
    _, _, comb_hc = hc_split_sinkhorn(
        [[0.1 * ((j * 7 + 3) % 9 - 4) for j in range(24)]],
        [0.2, 0.2, 0.2],
        [0.05 * (j % 5 - 2) for j in range(24)],
        hc_mult=4,
        sinkhorn_iters=20,
    )
    for col_j in range(4):
        col_sum = sum(comb_hc[0][r_i][col_j] for r_i in range(4))
        assert 1.0 - 5e-5 < col_sum <= 1.0 + 1e-6

    ds_gl = HostedModelGlance.from_published_profile("deepseek-v4.1-flash")
    glm_gl = HostedModelGlance.from_published_profile("glm-5.3-bf16")
    assert ds_gl.kv_compression_factor == 128.0 and round(glm_gl.kv_compression_factor, 2) == 56.89

    ds_arch, ds_wg = build_compact_deepseek_v41_fixture()
    ds_realizer = DeepSeekV41LayerRealizer(ds_arch, ds_wg)
    hc_s, pm_s = ds_realizer.initial_hc_stream([1, 5])
    l0_ds = ds_realizer.forward_layer(0, hc_s, pm_s)
    l1_ds = ds_realizer.forward_layer(1, l0_ds.hc_stream, l0_ds.next_pre_mix)
    ds_logits = ds_realizer.collapse_and_logits(l1_ds.hc_stream, l1_ds.next_pre_mix)
    ds_ref = DeepSeekV41ReferenceOracle.forward_full(ds_arch, ds_wg, [1, 5])
    assert NumericEngine.max_abs_diff(ds_logits, ds_ref) < 1e-4

    glm_arch, glm_wg = build_compact_glm53_fixture()
    glm_realizer = Glm53LayerRealizer(glm_arch, glm_wg, mla_mode="latent_absorbed")
    glm_emb = glm_realizer.embed_tokens([2, 7])
    mla_check = verify_mla_matrix_absorption(glm_arch, glm_wg, glm_emb, layer_index=0)
    assert mla_check["max_abs_diff"] < 1e-5
    checks["phase8_frontier_deepseek_v41_and_glm_53"] = 12

    # 12. Phase 8: F_p Commutant Blocks, Observer Congruence, Fabric Router, HeadlessOS, Router Path & Continuous System
    fp_chain = FiniteFieldOperatorChain(
        dimension=4,
        modulus=97,
        factors=(
            ((3, 5, 0, 0), (2, 7, 0, 0), (0, 0, 11, 4), (0, 0, 6, 9)),
            ((2, 1, 0, 0), (4, 3, 0, 0), (0, 0, 5, 2), (0, 0, 3, 7)),
        ),
    )
    fp_decomp = discover_similarity_blocks(fp_chain)
    assert fp_decomp.block_sizes == (2, 2)
    j_vec, _ = fp_decomp.navigate_horizon([1, 2, 3, 4], 10_000)
    assert len(j_vec) == 4

    s8_val = Signal8Backend().encode_symbol(5, observer=3)
    c_gate = CrossObserverCongruenceGate({"o1": b"s1", "o2": b"s2"}, quotient_modulus=8)
    r_o1 = AttributedObserverReading.sign(b"s1", observer_id="o1", ground_fingerprint=spec.fingerprint, scheduler_id="s", coordinate=9, representation_kind="canonical_int", raw_reading={"value": 21})
    r_o2 = AttributedObserverReading.sign(b"s2", observer_id="o2", ground_fingerprint=spec.fingerprint, scheduler_id="s", coordinate=9, representation_kind="signal8_phasor", raw_reading={"im": s8_val.phasor.imag, "observer_unit": 3, "phase_eps": 0.0, "re": s8_val.phasor.real})
    assert c_gate.verify_congruence((r_o1, r_o2))["agreed_quotient_value"] == 5

    f_router = ComputationalFabricRouter("p.src", {"p.dst": b"peer-secret-01234567"})
    f_router.add_link(FabricLink("p.src", "p.dst", (RouteMetric("latency_ms", "ms", 3),)))
    f_router.register_advertisement(
        HostAdvertisement.signed(
            b"peer-secret-01234567",
            peer_id="p.dst",
            host_id="h.dst",
            tls_dns="dst.vld.local",
            endpoint_host="127.0.0.1",
            port=9000,
            ground_fingerprint=spec.fingerprint,
            operator_contract="op.v1",
            substrate_kind="vm",
            local_metrics=(RouteMetric("latency_ms", "ms", 1),),
            observed_at=0,
            expires_at=50,
        )
    )
    assert (
        f_router.resolve_routes(
            ground_fingerprint=spec.fingerprint,
            operator_contract="op.v1",
            now=10,
            metric_order=(("latency_ms", "ms"),),
        ).status
        is FabricResolutionStatus.FOUND
    )
    exp_res = f_router.resolve_routes(
        ground_fingerprint=spec.fingerprint,
        operator_contract="op.v1",
        now=100,
        metric_order=(("latency_ms", "ms"),),
    )
    assert exp_res.status is FabricResolutionStatus.BLOCKED and FabricBlockerCode.LEASE_EXPIRED in exp_res.blockers

    hos = HeadlessOS(page_size=128, structural_memory=sm_reg)
    sm_reg.certify_and_admit("hos.seq", [10, 13, 16, 19, 22], domain="Z/97Z", declared_modulus=97)
    hos.vfs.register_structural_file("/proc/seq", memory_id="hos.seq", virtual_size=10**9)
    pid0 = hos.spawn_process("init", (VMInstruction("HALT", ()),))
    fd0 = hos.posix_invoke(pid0, 2, path="/proc/seq", flags=O_RDONLY).result
    hos.posix_invoke(pid0, 8, fd=fd0, offset=1000, whence=SEEK_SET)
    hos.posix_invoke(pid0, 0, fd=fd0, address=16, size=1)
    assert hos._procs[pid0].vm_proc.memory.read_bytes(16, 1)[0] == (10 + 3000) % 97

    sf = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t) for i, t in enumerate(TTL_PROBES)]
    rf = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t - 1) for i, t in enumerate(TTL_PROBES)]
    assert verify_ipv4_router_hop_ground("one", sender_frames=sf, hop_captures=[("one.r1", sf, rf)], receiver_ttls=[t - 1 for t in TTL_PROBES], require_in_path_transform=True).l3_hops == 1
    try:
        verify_ipv4_router_hop_ground("bridge", sender_frames=sf, hop_captures=[("bridge.r1", sf, sf)], receiver_ttls=list(TTL_PROBES), require_in_path_transform=True)
        raise AssertionError("Expected InternetPathRefused on L2 bridge negative control")
    except InternetPathRefused:
        pass

    c_mem = QuorumMembership("verify.cont", ("c0", "c1", "c2"))
    c_secs = {m: f"cont-secret-{m}-0123456789".encode("utf-8") for m in c_mem.member_ids}
    c_nodes = [QuorumAuthorityMember(m, c_secs[m]) for m in c_mem.member_ids]
    c_quorum = QuorumLineageCoordinator(c_mem, c_nodes, c_secs)
    c_store = PagedObjectStore(InMemoryPageBacking(512))
    c_jour = HashChainedJournal()
    c_stages = (
        ContinuousStageSpec("cs0", spec, "in", "mid", "sched.c0", lambda st: ({"v": int(st["x"]) + 5}, NavigationCostVector(0.1, 1.0, 8.0, 1.0), "PASS:cs0")),
        ContinuousStageSpec("cs1", spec, "mid", "out", "sched.c1", lambda st: ({"v": int(st["v"]) * 3}, NavigationCostVector(0.1, 1.0, 8.0, 1.0), "PASS:cs1")),
    )
    c_sys1 = ContinuousVirtualSystem("verify.pipe", store=c_store, quorum=c_quorum, active_quorum_members=("c0", "c1"), journal=c_jour)
    c_part = c_sys1.execute_stages(c_stages, {"x": 7}, host_id="h.1", interrupt_after_stage=0)
    assert c_part.stages_executed_this_invocation == 1
    c_sys2 = ContinuousVirtualSystem("verify.pipe", store=c_store, quorum=c_quorum, active_quorum_members=("c1", "c2"), journal=c_jour)
    c_done = c_sys2.execute_stages(c_stages, {"x": 7}, host_id="h.2")
    assert c_done.stages_recovered_from_quorum == 1 and c_done.stages_executed_this_invocation == 1 and c_done.final_state["v"] == 36
    checks["phase8_os_fabric_orbit_and_continuous_system"] = 12

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
        description="Virtual Layer Dynamics CLI — Ground verification, frontier model glancing, continuous execution & Vantage hosting.",
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

    gl_cmd = sub.add_parser("glance", help="Glance at the architectural Ground structure of a published or hosted frontier model.")
    gl_cmd.add_argument("--profile", type=str, default="deepseek-v4.1-flash", help=f"Published profile ({', '.join(sorted(PUBLISHED_FRONTIER_PROFILES))}).")
    gl_cmd.add_argument("--repo", type=str, default="", help="Optional live Hugging Face repo_id (e.g. deepseek-ai/DeepSeek-V4.1-Flash).")
    gl_cmd.add_argument("--revision", type=str, default="", help="40-char Git revision SHA when --repo is supplied.")

    cont_cmd = sub.add_parser("continuous", help="Run the 6-stage self-improving ContinuousVirtualSystem campaign across Grounds & hosts.")
    cont_cmd.add_argument("--horizon", type=int, default=10**12, help="Target algebraic orbit horizon (default: 10^12).")

    serve = sub.add_parser("serve", help="Host a live HTTP/1.1 Information-Flow Vantage Node.")
    serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    serve.add_argument("--port", type=int, default=8730, help="Bind port (default: 8730).")
    serve.add_argument("--node-id", type=str, default="vld.vantage.local", help="Logical receiver node ID.")
    serve.add_argument("--journal", type=str, default="", help="Optional path for durable HashChainedJournal.")
    serve.add_argument("--storage-dir", type=str, default="", help="Optional directory for durable M1 DiskPageBacking.")

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
                "vld.authority (scheduler-scoped vector clocks G13, decoupled provenance G21, 2q>N quorum M3 sessions & VirtualProcessor)",
                "vld.navigation (unbounded lazy NavigableComputationDAG, K_3 guards, ComposedGroundNavigator, F_p commutant blocks & observer congruence)",
                "vld.accel (StructuralLinearOperator, LinearRecurrenceGround, FP8/UE8M0/FP4 codecs, Birkhoff Sinkhorn mHC, NumPy + Rust CDLL)",
                "vld.fabric (VantageEnvelope, ReliableChannel, X.509 v3 mTLS mesh, ComputationalFabricRouter & PartitionedTensorCutCoordinator)",
                "vld.substrates (HeadlessOS x86_64 POSIX ABI, ELF64/WASM lifters, SIMT barriers, delta tensor log, IPv4/Internet Path Grounds & ROOM)",
                "vld.models (HostedModelGlance, LazyFrontierWeightGround, DeepSeek-V4.1, GLM-5.3 MLA Absorption, Qwen3 & SNC/PNF compilers)",
                "vld.agents (ProofGround hypergraph, ReceiverHypervisor CAS/capabilities, InLayerModelAgent & ContinuousLayerBuilderAgent)",
                "vld.continuous (quorum-fenced crash-resumable ContinuousVirtualSystem across changing Grounds, hosts & physical paths)",
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

    if args.command == "glance":
        try:
            if args.repo:
                if not args.revision:
                    parser.error("--revision (40-char Git SHA) is required when --repo is set.")
                gl = glance_huggingface_model(args.repo, revision=args.revision)
            else:
                gl = HostedModelGlance.from_published_profile(args.profile)
        except Exception as exc:
            parser.error(str(exc))
        print(json.dumps(gl.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "continuous":
        if args.horizon <= 0:
            parser.error("--horizon must be a positive integer.")
        camp = run_self_improving_continuous_campaign(horizon=args.horizon)
        out = {
            "admitted_upgrades": [u.upgrade_id for u in camp.build_cycle.admitted_upgrades],
            "origin_hits_delta_pass1": camp.origin_hits_delta_pass1,
            "origin_hits_delta_pass2": camp.origin_hits_delta_pass2,
            "pass1_cost": camp.pass1.total_cost.to_dict(),
            "pass1_final_state": dict(camp.pass1.final_state),
            "pass2_cost": camp.pass2_resumed.total_cost.to_dict(),
            "pass2_final_continuity_digest": camp.pass2_resumed.final_continuity_digest,
            "pass2_final_state": dict(camp.pass2_resumed.final_state),
            "pass2_pareto_dominates_pass1": camp.pass2_pareto_dominates_pass1,
            "pass2_stages_executed_after_host_switch": camp.pass2_resumed.stages_executed_this_invocation,
            "pass2_stages_recovered_from_quorum": camp.pass2_resumed.stages_recovered_from_quorum,
            "refused_upgrades": [r[0] for r in camp.build_cycle.refused_proposals],
            "warm_weight_bytes_pass2": camp.warm_weight_bytes_pass2,
        }
        print(json.dumps(out, indent=2, sort_keys=True))
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
        srv_journal = HashChainedJournal(args.journal) if args.journal else None
        srv_store = PagedObjectStore(DiskPageBacking(args.storage_dir)) if args.storage_dir else None
        node = VLDVantageNode(args.node_id, store=srv_store, journal=srv_journal)
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
