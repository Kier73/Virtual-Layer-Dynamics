"""Phase 9 — Adversarial Audit Hardening, Fluid In-Transit Substrate, Multi-Family Models & Recursive Arbitration Tests."""

from __future__ import annotations

import ast
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import threading
from typing import Any

import pytest

from vld.accel import NumericEngine, encode_bf16_le, has_numpy
from vld.agents import (
    AgentCapability,
    CapabilityGrant,
    HypervisorRefusalError,
    ReceiverHypervisor,
)
from vld.arbitration import (
    ArbitrationError,
    ContinuousArbitrationService,
    ExplorationTaskSpec,
    IdeaLedger,
    InLayerMetaArbitratorModel,
    build_deepseek_v41_worker,
    build_external_http_endpoint_worker,
    build_glm_53_worker,
    build_nested_sub_arbitrator_worker,
    build_transformer_ground_worker,
    create_meta_arbitrator_grant,
)
from vld.authority import QuorumAuthorityMember, QuorumLineageCoordinator, QuorumMembership
from vld.cli import main as cli_main
from vld.continuous import ContinuousStageSpec, ContinuousSystemError, ContinuousVirtualSystem
from vld.fabric.mtls import generate_ephemeral_mtls_pki
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.independent_audit import (
    IndependentAuditError,
    verify_continuity_receipt_chain_standalone,
    verify_quorum_certificate_standalone,
    verify_snc_1d_capsule_standalone,
    verify_transit_packet_receipts_standalone,
)
from vld.journal import HashChainedJournal
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.models import (
    IndependentReferenceOracle,
    ModelWeightSnapshot,
    TransformerArchitectureSpec,
    TransformerLayerRealizer,
    build_safetensors_bytes,
    compile_piecewise_affine_relu,
)
from vld.substrates import (
    FiniteTransducerSummary,
    FluidTransitError,
    InTransitPathMachine,
    OnlineFlashStreamState,
    StreamInterpreterNode,
    TransitOperation,
    TropicalMinPlusMatrix,
    VirtualMemoryMovementView,
    fold_cpu_stream_in_transit,
    optimize_transit_schedule,
    stream_flash_attention_in_transit,
    verify_premature_normalization_counterexample,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_github_workflow_yaml_syntax_and_colon_safety() -> None:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    yml_files = sorted(workflows_dir.glob("*.yml"))
    assert len(yml_files) >= 1
    for yml_path in yml_files:
        raw = yml_path.read_text(encoding="utf-8")
        # Zero-dependency check: single-line `run:` scalars must never contain unquoted `: ` mapping syntax
        for lineno, line in enumerate(raw.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("run:") or stripped.startswith("- run:"):
                rhs = stripped.split("run:", 1)[1].strip()
                if rhs and rhs not in ("|", ">", "|-", ">-"):
                    assert ": " not in rhs, (
                        f"{yml_path.name}:{lineno} has unquoted ': ' inside single-line YAML run scalar: {rhs!r}"
                    )
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            yaml = None
        if yaml is not None:
            parsed = yaml.safe_load(raw)
            assert isinstance(parsed, dict) and "jobs" in parsed


def test_ephemeral_mtls_private_key_0600_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_key_opens: list[tuple[str, int]] = []
    real_os_open = os.open

    def _spy_open(path: Any, flags: int, mode: int = 0o777, *args: Any, **kwargs: Any) -> int:
        p_str = str(path)
        if p_str.endswith(".key.pem"):
            observed_key_opens.append((p_str, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", _spy_open)
    pki_dir = tmp_path / "mtls_sec"
    bundles = generate_ephemeral_mtls_pki(pki_dir, ("alpha.vld.local", "beta.vld.local"))
    assert len(observed_key_opens) == 2
    for _, mode in observed_key_opens:
        assert mode == 0o600
    for b in bundles.values():
        assert b.key_path.read_text(encoding="ascii").startswith("-----BEGIN RSA PRIVATE KEY-----")
        if os.name != "nt":
            assert (b.key_path.stat().st_mode & 0o077) == 0
            assert (pki_dir.stat().st_mode & 0o077) == 0


def test_continuous_recovery_identity_refusal_and_independent_verifier() -> None:
    g0 = VirtualGroundSpec("audit.g0", "1.0", "math", "int", ("add",), ("inv",), ("v",))
    g1 = VirtualGroundSpec("audit.g1", "1.0", "math", "int", ("mul",), ("inv",), ("v",))
    g1_alt = VirtualGroundSpec("audit.g1.mutated", "1.0", "math", "int", ("mul",), ("inv",), ("v",))

    def _s0_run(st: Any) -> tuple[dict[str, Any], NavigationCostVector, str]:
        return {"v": int(st["x"]) + 10}, NavigationCostVector(0.1, 2.0, 8.0, 1.0), "PASS:s0"

    def _s0_ver(inp: Any, out: Any) -> tuple[bool, str]:
        return int(out["v"]) == int(inp["x"]) + 10, "s0_exact"

    def _s1_run(st: Any) -> tuple[dict[str, Any], NavigationCostVector, str]:
        return {"v": int(st["v"]) * 3}, NavigationCostVector(0.1, 2.0, 8.0, 1.0), "PASS:s1"

    def _s1_ver(inp: Any, out: Any) -> tuple[bool, str]:
        return int(out["v"]) == int(inp["v"]) * 3, "s1_exact"

    stages = (
        ContinuousStageSpec("s0", g0, "init_x", "mid_v", "sched.0", _s0_run, _s0_ver),
        ContinuousStageSpec("s1", g1, "mid_v", "out_v", "sched.1", _s1_run, _s1_ver),
    )

    q_mem = QuorumMembership("audit.rec.lin", ("q0", "q1", "q2"))
    secs = {m: f"audit-sec-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
    nodes = [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids]
    store = PagedObjectStore(InMemoryPageBacking(512))
    jour = HashChainedJournal()

    # Commit Stage 0 on host 1
    sys1 = ContinuousVirtualSystem(
        "audit.pipe",
        store=store,
        quorum=QuorumLineageCoordinator(q_mem, nodes, secs),
        active_quorum_members=("q0", "q1"),
        journal=jour,
        require_independent_verifier=True,
    )
    part = sys1.execute_stages(stages, {"x": 5}, host_id="h1", interrupt_after_stage=0)
    assert part.stages_executed_this_invocation == 1
    assert len(part.stage_telemetry) == 1 and part.stage_telemetry[0].m1_bytes_persisted > 0

    def _resume_sys() -> ContinuousVirtualSystem:
        return ContinuousVirtualSystem(
            "audit.pipe",
            store=store,
            quorum=QuorumLineageCoordinator(q_mem, nodes, secs),
            active_quorum_members=("q1", "q2"),
            journal=jour,
            require_independent_verifier=True,
            reverify_recovered_stages=True,
        )

    # 1. Changing genesis initial_input MUST be refused!
    with pytest.raises(ContinuousSystemError, match="source_ground_fingerprint"):
        _resume_sys().execute_stages(stages, {"x": 6}, host_id="h2.bad_input")

    # 2. Changing Stage 0 Ground spec MUST be refused!
    stages_bad_g0 = (
        ContinuousStageSpec("s0", g1_alt, "init_x", "mid_v", "sched.0", _s0_run, _s0_ver),
        stages[1],
    )
    with pytest.raises(ContinuousSystemError, match="target_ground_fingerprint"):
        _resume_sys().execute_stages(stages_bad_g0, {"x": 5}, host_id="h2.bad_g0")

    # 3. Changing Stage 0 representation contract MUST be refused!
    stages_bad_rep = (
        ContinuousStageSpec("s0", g0, "init_x", "tampered_rep", "sched.0", _s0_run, _s0_ver),
        stages[1],
    )
    with pytest.raises(ContinuousSystemError, match="representation contract"):
        _resume_sys().execute_stages(stages_bad_rep, {"x": 5}, host_id="h2.bad_rep")

    # 4. Valid resumption with exact matching input & specs succeeds with 0 replayed stages!
    done = _resume_sys().execute_stages(stages, {"x": 5}, host_id="h2.good")
    assert done.stages_recovered_from_quorum == 1
    assert done.stages_executed_this_invocation == 1
    assert done.final_state == {"v": 45}
    assert done.stage_telemetry[0].recovered_from_quorum is True
    assert done.stage_telemetry[1].recovered_from_quorum is False

    # 5. Truncating stage list below committed high-water slot (2 > 1) MUST be refused!
    with pytest.raises(ContinuousSystemError, match="exceeds supplied stage count"):
        _resume_sys().execute_stages((stages[0],), {"x": 5}, host_id="h3.truncated")

    # 6. Forged worker "PASS" verdict with wrong arithmetic MUST be rejected by independent verify_fn!
    forged_stage = ContinuousStageSpec(
        "forged",
        g0,
        "init_x",
        "mid_v",
        "sched.0",
        lambda _st: ({"v": 999999}, NavigationCostVector(0.1, 1.0, 1.0, 1.0), "PASS:forged_by_worker"),
        _s0_ver,
    )
    q_f = QuorumLineageCoordinator(
        QuorumMembership("forged.lin", ("q0", "q1", "q2")),
        [QuorumAuthorityMember(m, secs[m]) for m in ("q0", "q1", "q2")],
        secs,
    )
    sys_f = ContinuousVirtualSystem(
        "forged.pipe",
        store=PagedObjectStore(InMemoryPageBacking(512)),
        quorum=q_f,
        active_quorum_members=("q0", "q1"),
        require_independent_verifier=True,
    )
    with pytest.raises(ContinuousSystemError, match="rejected by mandatory independent verifier"):
        sys_f.execute_stages((forged_stage,), {"x": 5}, host_id="h.forged")
    assert q_f.high_water_slot == 0

    # 7. Missing verify_fn when require_independent_verifier=True MUST be refused!
    no_ver_stage = ContinuousStageSpec("nover", g0, "in", "out", "sc", _s0_run, None)
    with pytest.raises(ContinuousSystemError, match="lacks mandatory independent verify_fn"):
        sys_f.execute_stages((no_ver_stage,), {"x": 5}, host_id="h.nover")


def test_standalone_zero_import_auditor_and_tamper_rejection() -> None:
    # Prove zero vld.* imports inside src/vld/independent_audit.py
    src_path = REPO_ROOT / "src" / "vld" / "independent_audit.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("vld")
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module and node.module.startswith("vld"))

    # Verify continuity chain + negative tamper tests
    g = VirtualGroundSpec("sa.g", "1.0", "math", "int", ("inc",), ("inv",), ("v",))
    q_mem = QuorumMembership("sa.lin", ("q0", "q1", "q2"))
    secs = {m: f"sa-secret-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
    quorum = QuorumLineageCoordinator(q_mem, [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids], secs)
    st = PagedObjectStore(InMemoryPageBacking(512))
    cvs = ContinuousVirtualSystem(
        "sa.pipe",
        store=st,
        quorum=quorum,
        active_quorum_members=("q0", "q1"),
        require_independent_verifier=True,
    )
    stage = ContinuousStageSpec(
        "s0",
        g,
        "in",
        "out",
        "sched.0",
        lambda s: ({"v": int(s["x"]) + 2}, NavigationCostVector(0.1, 1.0, 4.0, 1.0), "PASS:s0"),
        lambda i, o: (int(o["v"]) == int(i["x"]) + 2, "ok"),
    )
    res = cvs.execute_stages((stage,), {"x": 11}, host_id="h0")
    rcpt_dicts = [r.to_dict() for r in res.receipts]

    rep = verify_continuity_receipt_chain_standalone(
        rcpt_dicts,
        pipeline_id="sa.pipe",
        initial_input={"x": 11},
        expected_stages=[
            {
                "ground_fingerprint": g.fingerprint,
                "input_representation": "in",
                "output_representation": "out",
                "stage_id": "s0",
            }
        ],
        quorum_member_ids=("q0", "q1", "q2"),
        quorum_member_secrets=secs,
        require_verified_tag=True,
    )
    assert rep["status"] == "VERIFIED"

    # Tampered initial_input fails standalone verification
    with pytest.raises(IndependentAuditError, match="source_ground_fingerprint"):
        verify_continuity_receipt_chain_standalone(rcpt_dicts, pipeline_id="sa.pipe", initial_input={"x": 12})

    # Tampered HMAC vote secret fails standalone quorum verification
    bad_secs = dict(secs)
    bad_secs["q0"] = b"wrong-secret-000000000000"
    with pytest.raises(IndependentAuditError, match="HMAC"):
        verify_quorum_certificate_standalone(
            rcpt_dicts[0]["quorum_certificate"], member_ids=("q0", "q1", "q2"), member_secrets=bad_secs
        )

    # 1D SNC ReLU standalone check + weight tamper rejection
    cap = compile_piecewise_affine_relu(
        [
            {"left": "-inf", "right": "-1", "affines": [["-2", "1"]]},
            {"left": "-1", "right": "2", "affines": [["1", "4"]]},
            {"left": "2", "right": "+inf", "affines": [["3", "0"]]},
        ]
    )
    assert verify_snc_1d_capsule_standalone(cap)["status"] == "VERIFIED"
    bad_cap = json.loads(json.dumps(cap))
    bad_cap["network"]["output_bias"] = ["999"]
    with pytest.raises(IndependentAuditError):
        verify_snc_1d_capsule_standalone(bad_cap)


def test_fluid_in_transit_cpu_gpu_vm_and_path_machines() -> None:
    # 1. CPU Tropical Min-Plus DP lifting vs sequential Bellman-Ford
    w = TropicalMinPlusMatrix(3, ((0.0, 2.0, 9.0), (4.0, 0.0, 1.5), (1.0, 3.0, 0.0)))
    s0 = (0.0, math.inf, math.inf)
    seq = s0
    for _ in range(19):
        seq = w.apply_left_state(seq)
    jumped = w.power(19).apply_left_state(s0)
    assert all(abs(a - b) < 1e-12 for a, b in zip(seq, jumped))
    with pytest.raises(FluidTransitError):
        TropicalMinPlusMatrix(1, ((float("nan"),),))

    # Branching CPU transducer stream fold across 5 chunks vs step-by-step execution
    def _branch_step(state: int, sym: int) -> int:
        if sym % 3 == 0:
            return (state ^ 5) % 8
        return (3 * state + sym) % 8

    stream_syms = [(7 * i + 3) % 11 for i in range(125)]
    chunks = [stream_syms[i : i + 25] for i in range(0, 125, 25)]
    folded_s, summary = fold_cpu_stream_in_transit(chunks, num_states=8, step_fn=_branch_step, initial_state=2)
    cur_s = 2
    for x in stream_syms:
        cur_s = _branch_step(cur_s, x)
    assert folded_s == cur_s and summary.steps_summarized == 125

    # 2. GPU Online Flash-Stream Monoid across 4 KV shards (including causally masked future shard!)
    queries = [[0.3, -0.4, 0.8, 0.1], [-0.2, 0.7, 0.1, -0.5], [0.6, 0.2, -0.3, 0.9]]
    all_k = [[math.sin(t * 0.7 + d) for d in range(4)] for t in range(8)]
    all_v = [[math.cos(t * 0.5 - d) for d in range(3)] for t in range(8)]
    # query_start_pos = 3 -> queries are at global positions 3, 4, 5; shard 3 (keys 6, 7) is completely masked out!
    shards = [(all_k[0:2], all_v[0:2]), (all_k[2:5], all_v[2:5]), (all_k[5:6], all_v[5:6]), (all_k[6:8], all_v[6:8])]
    flash_out, fl_tel = stream_flash_attention_in_transit(queries, shards, query_start_pos=3, causal=True)
    ref_out, _ = stream_flash_attention_in_transit(queries, [(all_k, all_v)], query_start_pos=3, causal=True)
    assert NumericEngine.max_abs_diff(flash_out, ref_out) < 1e-6
    assert fl_tel["hops"] == 4
    assert verify_premature_normalization_counterexample()["discrepancy"] > 1.0
    with pytest.raises(FluidTransitError, match="zero valid keys"):
        OnlineFlashStreamState.identity(2, 3).finalize()

    # 3. Virtual Memory Movement Lifting: zero-copy descriptor + pullback reduction commutation
    raw_buf = bytes((17 * i + 5) % 256 for i in range(4096))
    view = (
        VirtualMemoryMovementView.from_buffer("vm.page0", 4096)
        .transpose_2d(64, 64)
        .rotate_view(137)
        .reverse_view()
        .cow_write(10, 250)
        .cow_write(2048, 3)
    )
    assert view.descriptor_bytes_size < 300  # < 300 bytes in transit vs 4096-byte buffer!
    mat = view.materialize(raw_buf)
    assert view.commute_invariant_reduction(raw_buf, "sum") == sum(mat)
    xor_ref = 0
    for b in mat:
        xor_ref ^= b
    assert view.commute_invariant_reduction(raw_buf, "xor") == xor_ref

    sub_v = view.slice_view(5, 85, 3).cow_write(2, 99)
    sub_mat = sub_v.materialize(raw_buf)
    assert sub_v.commute_invariant_reduction(raw_buf, "sum") == sum(sub_mat)

    # 4. Path-Encoded Machine & Schedule Optimizer across both stream and HeadlessOS POSIX pipes
    schedule = (
        TransitOperation("reverse", {}),
        TransitOperation("reverse", {}),
        TransitOperation("affine_map", {"scale": 4, "bias": 3, "modulus": 101}),
        TransitOperation("affine_map", {"scale": 7, "bias": 5, "modulus": 101}),
        TransitOperation("transpose_2d", {"rows": 3, "cols": 4}),
        TransitOperation("rotate", {"shift": 5}),
        TransitOperation("reduce_sum", {}),
    )
    opt = optimize_transit_schedule(schedule, vector_length=12)
    assert len(opt) == 2  # Fused to 1 affine_map + 1 reduce_sum!
    vec_in = list(range(1, 13))
    pm = InTransitPathMachine((StreamInterpreterNode("hop.0", max_ops_per_hop=1), StreamInterpreterNode("hop.1", max_ops_per_hop=2)))
    pkt_pipe, _ = pm.execute_in_transit(
        "pkt.pipe", initial_payload=vec_in, schedule=schedule, target_representation="interpretable_scalar", transport="headless_os_pipe"
    )
    naive_out, _ = InTransitPathMachine.naive_hardware_reference(vec_in, schedule)
    assert pkt_pipe.payload == naive_out
    assert verify_transit_packet_receipts_standalone(json.loads(pkt_pipe.to_wire_bytes().decode("utf-8")))["hops_verified"] == 2


def _build_synthetic_checkpoint(
    arch: TransformerArchitectureSpec, seed: int = 3
) -> tuple[ModelWeightSnapshot, PagedObjectStore, dict[str, bytes]]:
    space = arch.build_parameter_space()
    raw_tensors: dict[str, bytes] = {}
    for idx, t in enumerate(space.tensors):
        n = math.prod(t.shape)
        if "norm" in t.name:
            vals = [1.0 + 0.05 * ((i + idx) % 5 - 2) for i in range(n)]
        elif t.name.endswith(".bias"):
            vals = [0.02 * ((i + idx) % 7 - 3) for i in range(n)]
        else:
            vals = [0.12 * math.sin((i + 1) * 0.31 + idx * 0.73 + seed) for i in range(n)]
        raw_tensors[t.name] = encode_bf16_le(vals)
    store = PagedObjectStore(InMemoryPageBacking(4096))
    snap = ModelWeightSnapshot.publish_tensors(space, store, raw_tensors)
    return snap, store, raw_tensors


def test_multi_family_llama_qwen2_and_glm4_realizers_vs_fp64_oracle() -> None:
    specs = (
        TransformerArchitectureSpec.from_huggingface_config(
            "HuggingFaceTB/SmolLM2-Compact",
            "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            {
                "model_type": "llama",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "vocab_size": 24,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-5,
                "rope_theta": 100000.0,
                "tie_word_embeddings": True,
                "attention_bias": False,
            },
        ),
        TransformerArchitectureSpec.from_huggingface_config(
            "deepseek-ai/DeepSeek-R1-Distill-Compact",
            "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
            {
                "model_type": "qwen2",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 48,
                "vocab_size": 24,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "attention_bias": True,
            },
        ),
        TransformerArchitectureSpec.from_huggingface_config(
            "THUDM/GLM-Z1-Compact",
            "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
            {
                "model_type": "glm4",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 48,
                "vocab_size": 24,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
                "attention_bias": True,
                "partial_rotary_factor": 0.5,
            },
        ),
    )

    toks = [2, 7, 13, 19]
    for arch in specs:
        snap_local, st, raw_map = _build_synthetic_checkpoint(arch)
        oracle_logits = IndependentReferenceOracle.forward_full_sequence(arch, raw_map, toks)

        for mode in ("numpy", "stdlib"):
            if mode == "numpy" and not has_numpy():
                continue
            eng = NumericEngine(mode)
            realizer = TransformerLayerRealizer(arch, snap_local, store=st, engine=eng)
            # Test prefill (first 3 tokens) + decode step (4th token with KV cache!)
            h_pre = realizer.embed_tokens(toks[:3])
            kv_k: list[Any] = []
            kv_v: list[Any] = []
            for l_i in range(arch.num_hidden_layers):
                out_pre = realizer.forward_layer(l_i, h_pre)
                h_pre = out_pre.hidden_states
                kv_k.append(out_pre.k_cache_full)
                kv_v.append(out_pre.v_cache_full)
            h_dec = realizer.embed_tokens(toks[3:])
            for l_i in range(arch.num_hidden_layers):
                out_dec = realizer.forward_layer(l_i, h_dec, past_k=kv_k[l_i], past_v=kv_v[l_i])
                h_dec = out_dec.hidden_states
            logits_last = realizer.final_logits(h_dec)[0]
            assert NumericEngine.max_abs_diff([logits_last], [oracle_logits[-1]]) < 1e-4, f"Mismatch on {arch.model_type} ({mode})"


def test_recursive_in_layer_arbitration_service_and_idea_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Spin up a tiny loopback OpenAI-compatible HTTP model endpoint to test external HTTP source arbitration
    class _MockLLMHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            req_doc = json.loads(self.rfile.read(length).decode("utf-8"))
            prompt = req_doc["messages"][0]["content"]
            reply = json.dumps({"choices": [{"message": {"content": f"SYNTHESIZED:{prompt[:24]}"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    http_srv = HTTPServer(("127.0.0.1", 0), _MockLLMHandler)
    th = threading.Thread(target=http_srv.serve_forever, daemon=True)
    th.start()
    try:
        hyp = ReceiverHypervisor("test.arb.hyp")
        hyp.register_agent(create_meta_arbitrator_grant("arb.top", probe_budget=5, max_arbitration_depth=2))
        hyp.register_agent(create_meta_arbitrator_grant("arb.sub", probe_budget=10, max_arbitration_depth=2))
        hyp.register_agent(
            CapabilityGrant("agent.no_arb", frozenset({AgentCapability.OBSERVE_TELEMETRY}), probe_budget=Fraction(5))
        )

        # Hypervisor negative controls: missing capability, self-arbitration, depth exceeded
        with pytest.raises(HypervisorRefusalError, match="lacks capability"):
            hyp.authorize_model_arbitration("agent.no_arb", target_worker_id="w0", task_id="t0")
        with pytest.raises(HypervisorRefusalError, match="self-arbitrate"):
            hyp.authorize_model_arbitration("arb.top", target_worker_id="arb.top", task_id="t0")
        with pytest.raises(HypervisorRefusalError, match="exceeds max_arbitration_depth"):
            hyp.authorize_model_arbitration("arb.top", target_worker_id="w0", task_id="t0", arbitration_depth=3)

        # Build multi-source fleet: DeepSeek-V4.1, GLM-5.3, Llama Transformer, HTTP Endpoint & Depth-2 Sub-Arbitrator
        llama_arch = TransformerArchitectureSpec.from_huggingface_config(
            "HuggingFaceTB/SmolLM2-Mini",
            "12fd25f77366fa6b3b4b768ec3050bf629380bac",
            {
                "model_type": "llama",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "vocab_size": 16,
                "max_position_embeddings": 32,
                "tie_word_embeddings": True,
            },
        )
        snap_llama, st_llama, _ = _build_synthetic_checkpoint(llama_arch)

        class _BoundSnap:
            def __init__(self, s: ModelWeightSnapshot, store: PagedObjectStore) -> None:
                self.space = s.space
                self._s = s
                self._st = store

            def read_tensor_bytes(self, name: str, **_: Any) -> bytes:
                return self._s.read_tensor_bytes(name, store=self._st)

        w_ds = build_deepseek_v41_worker("w.ds")
        w_glm = build_glm_53_worker("w.glm")
        w_llama = build_transformer_ground_worker(
            "w.smollm2", arch=llama_arch, snapshot=_BoundSnap(snap_llama, st_llama), capabilities=("text_generation", "sparse_moe")
        )
        w_http = build_external_http_endpoint_worker(
            "w.http",
            endpoint_url=f"http://127.0.0.1:{http_srv.server_port}/v1/chat/completions",
            model_name="local-mock",
            capabilities=("text_generation", "frontier_reasoning"),
        )
        sub_arb = InLayerMetaArbitratorModel("arb.sub", hyp, arbitration_depth=2)
        w_nested = build_nested_sub_arbitrator_worker(
            "w.nested", sub_arbitrator=sub_arb, sub_workers={"w.ds": w_ds, "w.glm": w_glm, "w.llama": w_llama}
        )

        top_arb = InLayerMetaArbitratorModel("arb.top", hyp, arbitration_depth=1)
        q_mem = QuorumMembership("arb.test.q", ("q0", "q1", "q2"))
        secs = {m: f"q-sec-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
        rep = top_arb.arbitrate_and_execute(
            ExplorationTaskSpec(
                "task.multi",
                "Multi-Source Exploration",
                ("frontier_reasoning", "text_generation"),
                "Explore cross-ground invariants.",
                desired_worker_count=3,
            ),
            {"w.ds": w_ds, "w.glm": w_glm, "w.llama": w_llama, "w.http": w_http, "w.nested": w_nested},
            cycle_index=0,
            store=PagedObjectStore(InMemoryPageBacking(4096)),
            quorum=QuorumLineageCoordinator(q_mem, [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids], secs),
            active_quorum_members=("q0", "q1"),
            journal=HashChainedJournal(),
        )
        assert "w.http" in rep.selected_workers
        w_outs = rep.pipeline_result.final_state["worker_outputs"]
        assert w_outs["w.http"]["response_text"].startswith("SYNTHESIZED:")

        # Exhaust remaining budget on arb.top -> must deterministically refuse!
        with pytest.raises(HypervisorRefusalError, match="exceeds remaining budget"):
            hyp.authorize_model_arbitration("arb.top", target_worker_id="w.ds", task_id="t.over", cost=10)
    finally:
        http_srv.shutdown()
        http_srv.server_close()

    # CLI `vld arbitrate` across two separate invocations -> cursor resumes seamlessly at cycle 2 -> 4!
    state_dir = tmp_path / "arb_svc"
    ifs_dir = tmp_path / "ifs_out"
    assert cli_main(["arbitrate", "--state-dir", str(state_dir), "--ifs-output-dir", str(ifs_dir), "--cycles", "2"]) == 0
    out1 = json.loads(capsys.readouterr().out)
    assert out1["completed_cycles_total"] == 2
    assert cli_main(["arbitrate", "--state-dir", str(state_dir), "--ifs-output-dir", str(ifs_dir), "--cycles", "2"]) == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["completed_cycles_total"] == 4
    assert len(list(ifs_dir.glob("cycle_*.json"))) == 4

    # IdeaLedger persistence, reload, hash-chain verification & tamper refusal
    ledger_file = tmp_path / "ideas.jsonl"
    ledger = IdeaLedger(ledger_file)
    e1 = ledger.record_idea(
        idea_id="IDEA-001",
        epistemic_status="Hypothesis",
        title="Homomorphic Transit Sketching",
        equation_or_invariant="S(A + B) = S(A) + S(B)",
        rationale_and_next_step="Verify Johnson-Lindenstrauss inner-product preservation across mTLS hops.",
        origin_agent="vld.architect",
    )
    e2 = ledger.record_idea(
        idea_id="IDEA-002",
        epistemic_status="Verified",
        title="Online Flash-Stream Associativity",
        equation_or_invariant="(m_A, l_A, O_A) otimes (m_B, l_B, O_B)",
        rationale_and_next_step="Verified < 1e-6 parity across 4 shards.",
        origin_agent="vld.architect",
    )
    # Reloaded IdeaLedger deduplicates existing idea_id idempotently, and tamper is refused
    reloaded = IdeaLedger(ledger_file)
    assert len(reloaded.entries) == 2 and reloaded.head_digest == e2.entry_digest
    assert reloaded.record_idea(
        idea_id="IDEA-001",
        epistemic_status="Hypothesis",
        title="Duplicate Retry",
        equation_or_invariant="S(A+B)=S(A)+S(B)",
        rationale_and_next_step="Retry",
        origin_agent="vld.architect",
    ).entry_digest == e1.entry_digest
    assert len(reloaded.entries) == 2
    assert "IDEA-001" in reloaded.to_markdown() and e1.entry_digest != e2.entry_digest
    # Corrupt persisted JSONL -> must raise ArbitrationError on reload!
    ledger_file.write_text(ledger_file.read_text(encoding="utf-8").replace("IDEA-001", "TAMPERED"), encoding="utf-8")
    with pytest.raises(ArbitrationError):
        IdeaLedger(ledger_file)


def test_review_regressions_recovery_verdict_m1_corruption_snc_flash_and_schedule(tmp_path: Path) -> None:
    from vld.independent_audit import independent_domain_digest
    from vld.memory import DiskPageBacking
    from vld.models import ModelArtifactError

    # 1. Forged unverified recovered stage MUST be rejected under require_independent_verifier=True!
    g0 = VirtualGroundSpec("reg.g0", "1.0", "math", "int", ("add",), ("inv",), ("v",))
    q_mem = QuorumMembership("reg.lin", ("q0", "q1", "q2"))
    secs = {m: f"reg-sec-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
    nodes = [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids]
    disk_backing = DiskPageBacking(tmp_path / "m1_reg", page_size=512)
    store = PagedObjectStore(disk_backing)
    jour = HashChainedJournal()

    s0_unver = ContinuousStageSpec(
        "s0", g0, "in", "out", "sc.0", lambda _st: ({"v": 999999}, NavigationCostVector(0.1, 1.0, 1.0, 1.0), "PASS:forged"), None
    )
    ContinuousVirtualSystem(
        "reg.pipe",
        store=store,
        quorum=QuorumLineageCoordinator(q_mem, nodes, secs),
        active_quorum_members=("q0", "q1"),
        journal=jour,
        require_independent_verifier=False,
    ).execute_stages((s0_unver,), {"x": 5}, host_id="h.unver")

    s0_ver = ContinuousStageSpec(
        "s0",
        g0,
        "in",
        "out",
        "sc.0",
        lambda st: ({"v": int(st["x"]) + 10}, NavigationCostVector(0.1, 1.0, 1.0, 1.0), "PASS:s0"),
        lambda i, o: (int(o["v"]) == int(i["x"]) + 10, "exact"),
    )
    for explicit_reverify in (None, False, True):
        kwargs: dict[str, Any] = {} if explicit_reverify is None else {"reverify_recovered_stages": explicit_reverify}
        with pytest.raises(ContinuousSystemError, match="VERIFIED"):
            ContinuousVirtualSystem(
                "reg.pipe",
                store=store,
                quorum=QuorumLineageCoordinator(q_mem, nodes, secs),
                active_quorum_members=("q1", "q2"),
                journal=jour,
                require_independent_verifier=True,
                **kwargs,
            ).execute_stages((s0_ver,), {"x": 5}, host_id="h.ver")

    # Out-of-band M1 page truncation on disk raises ContinuousSystemError cleanly during recovery
    for p_digest in disk_backing.list_pages():
        p_file = (tmp_path / "m1_reg") / p_digest[:2] / f"{p_digest}.page"
        p_file.write_bytes(b"truncated")
    with pytest.raises(ContinuousSystemError, match="Corrupted"):
        ContinuousVirtualSystem(
            "reg.pipe",
            store=store,
            quorum=QuorumLineageCoordinator(q_mem, nodes, secs),
            active_quorum_members=("q1", "q2"),
            journal=jour,
            require_independent_verifier=False,
        ).execute_stages((s0_unver,), {"x": 5}, host_id="h.corrupt")

    # 2. Standalone 1D SNC verifier rejects non-ReLU activation, 2D weights, and shadowed/discontinuous atlases
    shadow_atlas = [
        {"left": "-inf", "right": "+inf", "affines": [["0", "0"]]},
        {"left": "0", "right": "+inf", "affines": [["999", "999"]]},
    ]
    forged_body = {
        "network": {
            "hidden_activation": "tanh",
            "hidden_bias": ["0"],
            "hidden_weight": [["0", "9999"]],
            "output_bias": ["0"],
            "output_weight": [["0"]],
        },
        "schema": "vld.snc-relu-capsule.v1",
        "source_atlas": shadow_atlas,
        "source_atlas_digest": independent_domain_digest("vld.semantic-atlas.v1", shadow_atlas),
    }
    forged_cap = {**forged_body, "capsule_digest": independent_domain_digest("vld.snc-relu-capsule.v1", forged_body)}
    with pytest.raises(IndependentAuditError):
        verify_snc_1d_capsule_standalone(forged_cap)

    disc_atlas = [
        {"left": "-inf", "right": "0", "affines": [["1", "0"]]},
        {"left": "0", "right": "+inf", "affines": [["1", "5"]]},
    ]
    disc_body = {
        "network": {
            "hidden_activation": "relu",
            "hidden_bias": ["0"],
            "hidden_weight": [["1"]],
            "output_bias": ["0"],
            "output_weight": [["1"]],
        },
        "schema": "vld.snc-relu-capsule.v1",
        "source_atlas": disc_atlas,
        "source_atlas_digest": independent_domain_digest("vld.semantic-atlas.v1", disc_atlas),
    }
    with pytest.raises(IndependentAuditError, match="Discontinuous"):
        verify_snc_1d_capsule_standalone({**disc_body, "capsule_digest": independent_domain_digest("vld.snc-relu-capsule.v1", disc_body)})

    # 3. OnlineFlashStreamState rejects non-finite values & malformed support invariants
    with pytest.raises(FluidTransitError, match="Non-finite"):
        OnlineFlashStreamState.from_kv_shard([[1.0]], [[1.0]], [[float("nan")]], causal=False)
    with pytest.raises(FluidTransitError):
        OnlineFlashStreamState(1, 1, (0.0,), (0.0,), ((0.0,),))

    # 4. optimize_transit_schedule preserves errors & HeadlessOS pipe streams >16 KiB payloads cleanly
    pm = InTransitPathMachine((StreamInterpreterNode("h0", max_ops_per_hop=4),))
    with pytest.raises(FluidTransitError, match="transpose_2d"):
        pm.execute_in_transit(
            "bad.tp",
            initial_payload=[1, 2, 3, 4, 5],
            schedule=(TransitOperation("transpose_2d", {"rows": -2, "cols": -3}), TransitOperation("reduce_sum", {})),
            target_representation="interpretable_scalar",
            optimize=True,
        )
    with pytest.raises(FluidTransitError, match="int_vector"):
        pm.execute_in_transit(
            "bad.post_scalar",
            initial_payload=[1, 2, 3, 4],
            schedule=(
                TransitOperation("reduce_sum", {}),
                TransitOperation("reverse", {}),
                TransitOperation("reverse", {}),
            ),
            target_representation="interpretable_scalar",
            optimize=True,
        )
    big_vec = list(range(3000))
    big_pkt, big_tel = pm.execute_in_transit(
        "big.pipe",
        initial_payload=big_vec,
        schedule=(TransitOperation("affine_map", {"scale": 3, "bias": 1, "modulus": 1009}), TransitOperation("reduce_sum", {})),
        target_representation="interpretable_scalar",
        transport="headless_os_pipe",
    )
    assert big_pkt.payload == InTransitPathMachine.naive_hardware_reference(
        big_vec, (TransitOperation("affine_map", {"scale": 3, "bias": 1, "modulus": 1009}), TransitOperation("reduce_sum", {}))
    )[0]
    assert big_tel["wire_bytes_moved"] > 12_000

    # 5. TransformerArchitectureSpec fingerprint collision prevention, falsy 0.0 rejection, GLM-4 prf & Qwen2 sliding_window
    base_q3 = TransformerArchitectureSpec.from_published_profile("qwen3-0.6b")
    mod_q3 = TransformerArchitectureSpec(**{**base_q3.to_dict(), "attention_bias": True, "partial_rotary_factor": 0.5})
    assert base_q3.fingerprint != mod_q3.fingerprint
    with pytest.raises(ModelArtifactError):
        TransformerArchitectureSpec.from_huggingface_config(
            "Qwen/Qwen3-Bad",
            "c1899de289a04d12100db370d81485cdf75e47ca",
            {"model_type": "qwen3", "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2, "intermediate_size": 32, "vocab_size": 16, "partial_rotary_factor": 0.0},
        )

    for prf_val in (0.25, 1.0):
        glm_var = TransformerArchitectureSpec.from_huggingface_config(
            "THUDM/GLM-Z1-PRF",
            "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
            {
                "model_type": "glm4",
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "intermediate_size": 32,
                "vocab_size": 16,
                "partial_rotary_factor": prf_val,
            },
        )
        sn, st_g, rm = _build_synthetic_checkpoint(glm_var)
        ref_g = IndependentReferenceOracle.forward_full_sequence(glm_var, rm, [1, 4, 9])
        for mode in ("numpy", "stdlib"):
            if mode == "numpy" and not has_numpy():
                continue
            rlz = TransformerLayerRealizer(glm_var, sn, store=st_g, engine=NumericEngine(mode))
            h = rlz.forward_layer(0, rlz.embed_tokens([1, 4, 9])).hidden_states
            assert NumericEngine.max_abs_diff(rlz.final_logits(h), ref_g) < 1e-4

    q2_sw = TransformerArchitectureSpec.from_huggingface_config(
        "Qwen/Qwen2-SW",
        "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        {
            "model_type": "qwen2",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 48,
            "vocab_size": 24,
            "use_sliding_window": True,
            "sliding_window": 2,
            "max_window_layers": 1,
        },
    )
    sn_sw, st_sw, rm_sw = _build_synthetic_checkpoint(q2_sw)
    toks_sw = [2, 5, 9, 13, 17]
    ref_sw = IndependentReferenceOracle.forward_full_sequence(q2_sw, rm_sw, toks_sw)
    for mode in ("numpy", "stdlib"):
        if mode == "numpy" and not has_numpy():
            continue
        rlz_sw = TransformerLayerRealizer(q2_sw, sn_sw, store=st_sw, engine=NumericEngine(mode))
        h_p = rlz_sw.embed_tokens(toks_sw[:3])
        kvk, kvv = [], []
        for li in range(2):
            o_p = rlz_sw.forward_layer(li, h_p)
            h_p = o_p.hidden_states
            kvk.append(o_p.k_cache_full)
            kvv.append(o_p.v_cache_full)
        h_d = rlz_sw.embed_tokens(toks_sw[3:])
        for li in range(2):
            h_d = rlz_sw.forward_layer(li, h_d, past_k=kvk[li], past_v=kvv[li]).hidden_states
        assert NumericEngine.max_abs_diff(rlz_sw.final_logits(h_d), ref_sw[3:]) < 1e-4


def test_crash_between_quorum_and_cursor_and_mtls_fabric_worker(tmp_path: Path) -> None:
    from vld.arbitration import build_mtls_fabric_worker
    from vld.fabric.mtls import MTLSPeerClient, MTLSPeerServer, create_mtls_client_context, create_mtls_server_context

    # 1. Crash right after quorum commit + IdeaLedger append, BEFORE service_cursor.json write!
    svc_dir = tmp_path / "crash_window_svc"
    task = ExplorationTaskSpec("t.idem", "Idempotent Crash Recovery", ("sparse_moe", "mla_absorption"), "probe", desired_worker_count=2)
    svc1 = ContinuousArbitrationService(svc_dir)

    def _boom_before_cursor(cyc: int) -> None:
        if cyc == 0:
            raise RuntimeError("SIMULATED_CRASH_BEFORE_CURSOR_WRITE")

    with pytest.raises(RuntimeError, match="SIMULATED_CRASH_BEFORE_CURSOR_WRITE"):
        svc1.run_cycles((task,), num_cycles=1, _fault_before_cursor_write=_boom_before_cursor)
    assert svc1.completed_cycles() == 0
    budget_after_crash = svc1.hypervisor.remaining_budget("agent.meta.arbitrator.primary")
    ledger_count_after_crash = len(IdeaLedger(svc_dir / "idea_ledger.jsonl").entries)
    assert ledger_count_after_crash == 1

    # Restart fresh service -> must recover cycle 0 from quorum (0 stages re-executed), NOT double-charge budget, and NOT duplicate IdeaLedger!
    svc2 = ContinuousArbitrationService(svc_dir)
    reps = svc2.run_cycles((task,), num_cycles=2)
    assert svc2.completed_cycles() == 2
    assert reps[0].pipeline_result.stages_recovered_from_quorum == 2
    assert reps[0].pipeline_result.stages_executed_this_invocation == 0
    assert reps[1].pipeline_result.stages_executed_this_invocation == 2
    # Cycle 0 did NOT re-deduct budget; only cycle 1 deducted 2 units!
    assert svc2.hypervisor.remaining_budget("agent.meta.arbitrator.primary") == budget_after_crash - Fraction(2)
    assert len(svc2.idea_ledger.entries) == 2

    # 2. Arbitrate a live mutually-authenticated X.509 v3 mTLS VLD peer server (`build_mtls_fabric_worker`)
    pki = generate_ephemeral_mtls_pki(tmp_path / "mtls_arb_pki", ("arb.client.vld", "worker.remote.vld"))
    srv_store = PagedObjectStore(InMemoryPageBacking(1024))
    cli_store = PagedObjectStore(InMemoryPageBacking(1024))
    ds_worker = build_deepseek_v41_worker("remote.ds")

    def _remote_ctrl(caller_dns: str, body: dict[str, Any]) -> dict[str, Any]:
        assert caller_dns == "arb.client.vld" and body["op"] == "execute_arbitrated_model"
        r_task = ExplorationTaskSpec(body["task_id"], "Remote", ("sparse_moe",), body["prompt_text"], tuple(body["prompt_tokens"]))
        out_doc, _ = ds_worker.run_fn(r_task, {})
        m = srv_store.put_bytes( json.dumps(out_doc, sort_keys=True).encode("utf-8"))
        return {"result_manifest_digest": m.manifest_digest}

    with MTLSPeerServer(
        peer_id="remote.node.0",
        ssl_context=create_mtls_server_context(pki["worker.remote.vld"]),
        allowed_peer_dns=frozenset({"arb.client.vld"}),
        store=srv_store,
        control_handler=_remote_ctrl,
    ) as srv:
        host, port = srv.endpoint
        m_client = MTLSPeerClient(
            host=host,
            port=port,
            expected_server_dns="worker.remote.vld",
            ssl_context=create_mtls_client_context(pki["arb.client.vld"]),
        )
        w_mtls = build_mtls_fabric_worker(
            "w.mtls.deepseek",
            mtls_client=m_client,
            target_store=cli_store,
            capabilities=("mtls_fabric", "frontier_reasoning"),
            independent_verifier_fn=lambda t, rem: ds_worker.verify_fn(t, {}, rem),
        )
        hyp = ReceiverHypervisor("mtls.hyp")
        hyp.register_agent(create_meta_arbitrator_grant("arb.mtls", probe_budget=8))
        arb = InLayerMetaArbitratorModel("arb.mtls", hyp)
        qm = QuorumMembership("mtls.q", ("q0", "q1", "q2"))
        qs = {m: f"mtls-sec-{m}-0123456789abcdef".encode("utf-8") for m in qm.member_ids}
        rep_m = arb.arbitrate_and_execute(
            ExplorationTaskSpec("t.mtls", "mTLS Remote Model", ("mtls_fabric",), "test", desired_worker_count=1),
            {"w.mtls.deepseek": w_mtls},
            cycle_index=0,
            store=cli_store,
            quorum=QuorumLineageCoordinator(qm, [QuorumAuthorityMember(m, qs[m]) for m in qm.member_ids], qs),
            active_quorum_members=("q0", "q1"),
            journal=HashChainedJournal(),
        )
        assert rep_m.selected_workers == ("w.mtls.deepseek",)
        assert srv.bulk_bytes_streamed > 0
