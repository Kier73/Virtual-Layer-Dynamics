"""Phases 6 & 7 tests: Hosted/partitioned LLMs, sparse MoE locality, hard-kill orbit recovery, procedural compilation, and agentic hypervisors."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from vld.accel import NumericEngine, encode_bf16_le, encode_f32_le, has_numpy
from vld.agents import (
    AgentCapability,
    CASConflictError,
    CapabilityGrant,
    HypervisorRefusalError,
    InLayerModelAgent,
    ProofGroundAdapter,
    VerifiedKernelReceipt,
    create_default_exploration_grant,
    ReceiverHypervisor,
)
from vld.fabric import VLDHttpVantageClient, VLDHttpVantageServer, VLDVantageNode
from vld.memory import DiskPageBacking, InMemoryPageBacking, PagedObjectStore, StructuralMemoryRegistry
from vld.models import (
    AutoregressiveModelRunner,
    ByteLevelBPETokenizerGround,
    FeedForwardBlock,
    GenerativeWeightGround,
    IndependentReferenceOracle,
    ModelArtifactError,
    ModelOrbitCheckpoint,
    ModelOrbitError,
    ModelParameterSpace,
    ModelWeightSnapshot,
    PagedKVCacheGround,
    ProceduralModelError,
    TensorSpace,
    TransformerArchitectureSpec,
    build_safetensors_bytes,
    compile_fsm_to_neural,
    compile_piecewise_affine_relu,
    evaluate_compiled_capsule,
    insert_identity_depth,
    observe_safetensors_header,
    permute_hidden_units,
    verify_compiled_capsule,
    widen_block_dyadic,
)
from vld.signal import GroundTransitionSystem, TransitionObservation, TransitionOutcome


def _make_deterministic_weights(arch: TransformerArchitectureSpec) -> dict[str, bytes]:
    space = arch.build_parameter_space()
    raw: dict[str, bytes] = {}
    for idx, t in enumerate(space.tensors):
        vals = [
            0.15 * math.sin(0.37 * (k + 1) + 1.13 * (idx + 1)) + (0.85 if "norm.weight" in t.name else 0.0)
            for k in range(t.elements)
        ]
        raw[t.name] = encode_bf16_le(vals) if t.dtype == "BF16" else encode_f32_le(vals)
    return raw


def test_safetensors_hfr_over_http_vantage_tokenizer_and_published_profiles() -> None:
    # 1. Verify published Qwen3 dense (0.6B) and sparse MoE (30B / 235B) lazy profiles
    q06 = TransformerArchitectureSpec.from_published_profile("qwen3-0.6b")
    q235 = TransformerArchitectureSpec.from_published_profile("qwen3-235b-a22b")
    inv06 = q06.parameter_inventory()
    inv235 = q235.parameter_inventory()
    assert inv06["active_elements_per_token_upper_bound"] == inv06["declared_elements"]
    # For 235B-A22B (8 of 128 experts active), active bytes per token < 12% of declared bytes!
    assert inv235["active_bytes_per_token_upper_bound"] < 0.12 * inv235["declared_bytes"]

    # 2. Build a real .safetensors shard with a large inactive tensor and host on HTTP Vantage
    small_w = encode_f32_le([1.25, -2.5, 3.75, 0.5])
    huge_inactive = bytes(4096)  # 64 pages of 64 bytes
    st_bytes = build_safetensors_bytes(
        {
            "active.proj": ("F32", (2, 2), small_w),
            "inactive.expert_99": ("U8", (64, 64), huge_inactive),
        },
        metadata={"format": "pt", "producer": "vld-test"},
    )
    backing = InMemoryPageBacking(page_size=64)
    store = PagedObjectStore(backing)
    node = VLDVantageNode("hfr-host", store=store)
    manifest = store.put_bytes(st_bytes)

    with VLDHttpVantageServer(node) as server:
        client = VLDHttpVantageClient(server.base_url)
        backing.pages_read = 0
        obs = observe_safetensors_header(
            lambda off, ln: client.fetch_m1_range(manifest.manifest_digest, off, ln),
            manifest.size_bytes,
            shard_name="remote-shard.safetensors",
        )
        assert obs.metadata["producer"] == "vld-test"
        # Header is ~200 bytes -> at most 4 pages read out of 67+ pages!
        assert backing.pages_read <= 5
        abs_off, byte_len = obs.absolute_byte_range("active.proj")
        fetched_proj = client.fetch_m1_range(manifest.manifest_digest, abs_off, byte_len)
        assert fetched_proj == small_w
        assert backing.pages_read <= 7  # 4096-byte inactive expert tensor was NEVER touched!

    # 3. Deterministic Byte-Level BPE Tokenizer Ground round-trip receipts
    corpus = ["Virtual Layer Dynamics Δ -> Σ", "Virtual Layer Ground navigation", "Virtual Layer Ground"]
    tok = ByteLevelBPETokenizerGround.from_corpus(corpus, num_merges=24)
    sample_text = "Virtual Layer Dynamics Δ -> Σ 🚀 中文"
    receipt = tok.encode_with_receipt(sample_text)
    assert len(receipt.token_ids) < len(sample_text.encode("utf-8"))
    assert tok.decode(receipt.token_ids) == sample_text

    with pytest.raises(ModelArtifactError):
        tok.encode("bad-\ud800-surrogate")


def test_dense_and_sparse_moe_realizers_locality_and_oracle_parity() -> None:
    # 1. Dense Qwen3 2-layer compact architecture (BF16 weights)
    dense_arch = TransformerArchitectureSpec(
        source_model="Qwen/Qwen3-CompactDense",
        revision="a" * 40,
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=32,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        dtype="BF16",
    )
    dense_raw = _make_deterministic_weights(dense_arch)
    dense_store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    dense_snap = ModelWeightSnapshot.publish_tensors(dense_arch.build_parameter_space(), dense_store, dense_raw)

    # Explicit refusal of nonlinear softmax KV affine collapse
    with pytest.raises(ModelOrbitError, match="non-bilinear"):
        PagedKVCacheGround.attempt_affine_kv_collapse()

    prompt = (3, 11, 7)
    runner_std = AutoregressiveModelRunner(
        dense_arch, dense_snap, dense_store, engine=NumericEngine("stdlib")
    )
    gen_std = runner_std.generate(prompt, max_new_tokens=3)
    oracle_logits_step0 = IndependentReferenceOracle.forward_full_sequence(dense_arch, dense_raw, prompt)[-1]
    oracle_full_logits = IndependentReferenceOracle.forward_full_sequence(
        dense_arch, dense_raw, gen_std.all_tokens[:-1]
    )[-1]
    assert NumericEngine.max_abs_diff(gen_std.last_step_logits, oracle_full_logits) < 1e-4
    oracle_first_tok = min(range(len(oracle_logits_step0)), key=lambda i: (-oracle_logits_step0[i], i))
    assert gen_std.generated_tokens[0] == oracle_first_tok

    if has_numpy():
        runner_np = AutoregressiveModelRunner(
            dense_arch, dense_snap, PagedObjectStore(InMemoryPageBacking(256)), weight_store=dense_store, engine=NumericEngine("numpy")
        )
        gen_np = runner_np.generate(prompt, max_new_tokens=3)
        assert gen_np.generated_tokens == gen_std.generated_tokens
        assert NumericEngine.max_abs_diff(gen_np.last_step_logits, gen_std.last_step_logits) < 1e-4

    # 2. Sparse Qwen3-MoE 1-layer model with 6 experts, top_k=2 active per token!
    moe_arch = TransformerArchitectureSpec(
        source_model="Qwen/Qwen3-CompactMoE",
        revision="b" * 40,
        model_type="qwen3_moe",
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=6,
        num_experts_per_tok=2,
        moe_intermediate_size=24,
        vocab_size=24,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        dtype="BF16",
    )
    moe_raw = _make_deterministic_weights(moe_arch)
    moe_store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    moe_snap = ModelWeightSnapshot.publish_tensors(moe_arch.build_parameter_space(), moe_store, moe_raw)
    moe_runner = AutoregressiveModelRunner(moe_arch, moe_snap, moe_store)
    # Single-token decode step: at most 2 of 6 experts can be activated!
    moe_gen = moe_runner.generate((5,), max_new_tokens=1)
    loaded_experts = {
        name.split("mlp.experts.")[1].split(".")[0]
        for name in moe_gen.loaded_tensor_names
        if "mlp.experts." in name
    }
    assert len(loaded_experts) == 2  # 4 of 6 inactive experts were NEVER loaded from M1 store!
    moe_oracle_logits = IndependentReferenceOracle.forward_full_sequence(moe_arch, moe_raw, (5,))[-1]
    assert NumericEngine.max_abs_diff(moe_gen.last_step_logits, moe_oracle_logits) < 1e-4


def test_subprocess_hard_kill_mid_decode_and_second_host_orbit_resumption(tmp_path: Path) -> None:
    arch = TransformerArchitectureSpec(
        source_model="Qwen/Qwen3-CrashTest",
        revision="c" * 40,
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=24,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        dtype="BF16",
    )
    raw_weights = _make_deterministic_weights(arch)

    # Uninterrupted reference run
    ref_store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    ref_snap = ModelWeightSnapshot.publish_tensors(arch.build_parameter_space(), ref_store, raw_weights)
    uninterrupted = AutoregressiveModelRunner(arch, ref_snap, ref_store).generate((2, 9), max_new_tokens=3)

    # Shared durable M1 DiskPageBacking directory + checkpoint file
    m1_dir = tmp_path / "shared_m1"
    ckpt_file = tmp_path / "orbit_ckpt.json"
    spec_file = tmp_path / "arch.json"
    manifests_file = tmp_path / "weight_manifests.json"

    host1_store = PagedObjectStore(DiskPageBacking(m1_dir, page_size=256))
    host1_snap = ModelWeightSnapshot.publish_tensors(arch.build_parameter_space(), host1_store, raw_weights)
    spec_file.write_text(json.dumps(arch.to_dict()), encoding="utf-8")
    manifests_file.write_text(
        json.dumps({k: m.to_dict() for k, m in host1_snap.manifests.items()}),
        encoding="utf-8",
    )

    # Spawn real child process that crashes via os._exit(42) mid-decode (after token 1, layer 0)!
    crash_script = "\n".join(
        [
            "import json, os",
            "from pathlib import Path",
            "from vld.memory import DiskPageBacking, PagedByteManifest, PagedObjectStore",
            "from vld.models import AutoregressiveModelRunner, ModelWeightSnapshot, TransformerArchitectureSpec",
            f"arch = TransformerArchitectureSpec(**json.loads(Path({str(spec_file)!r}).read_text('utf-8')))",
            f"store = PagedObjectStore(DiskPageBacking({str(m1_dir)!r}, page_size=256))",
            f"m_raw = json.loads(Path({str(manifests_file)!r}).read_text('utf-8'))",
            "snap = ModelWeightSnapshot(arch.build_parameter_space(), {k: PagedByteManifest.from_dict(v) for k, v in m_raw.items()})",
            "runner = AutoregressiveModelRunner(arch, snap, store)",
            "def cb(ckpt, gen_count, layer_idx):",
            f"    Path({str(ckpt_file)!r}).write_text(ckpt.to_json(), encoding='utf-8')",
            "    if gen_count == 1 and layer_idx == 0:",
            "        os._exit(42)",
            "runner.generate((2, 9), max_new_tokens=3, on_checkpoint=cb)",
        ]
    )
    proc = subprocess.run([sys.executable, "-c", crash_script], capture_output=True, text=True, timeout=20, check=False)
    assert proc.returncode == 42
    assert ckpt_file.is_file()

    # Host 2 loads checkpoint + shared M1 DiskPageBacking and resumes mid-layer to completion!
    saved_ckpt = ModelOrbitCheckpoint.from_json(ckpt_file.read_text(encoding="utf-8"))
    assert saved_ckpt.generated_tokens == (uninterrupted.generated_tokens[0],)
    assert saved_ckpt.completed_layers_in_step == 1

    host2_store = PagedObjectStore(DiskPageBacking(m1_dir, page_size=256))
    host2_snap = ModelWeightSnapshot(
        arch.build_parameter_space(),
        {k: m for k, m in host1_snap.manifests.items()},
    )
    host2_runner = AutoregressiveModelRunner(arch, host2_snap, host2_store)

    # Wrong expected_checkpoint_digest pin must be deterministically refused!
    with pytest.raises(ModelOrbitError, match="high-water digest pin"):
        host2_runner.generate(
            (2, 9),
            max_new_tokens=3,
            resume_checkpoint=saved_ckpt,
            expected_checkpoint_digest="0" * 64,
        )

    resumed = host2_runner.generate(
        (2, 9),
        max_new_tokens=3,
        resume_checkpoint=saved_ckpt,
        expected_checkpoint_digest=saved_ckpt.checkpoint_digest,
    )
    assert resumed.generated_tokens == uninterrupted.generated_tokens
    assert NumericEngine.max_abs_diff(resumed.last_step_logits, uninterrupted.last_step_logits) < 1e-6


def test_procedural_compilation_cross_process_verification_and_morphisms() -> None:
    # 1. Exact rational piecewise-affine compilation & cross-process standalone verification
    atlas = [
        {"left": "-inf", "right": "-2", "affines": [["-1", "-1"], ["2", "5"]]},
        {"left": "-2", "right": "3/2", "affines": [["2", "5"], ["0", "1"]]},
        {"left": "3/2", "right": "+inf", "affines": [["-4", "14"], ["4", "-5"]]},
    ]
    capsule = compile_piecewise_affine_relu(atlas)
    capsule_json = json.dumps(capsule)

    # Verify in an isolated child process without invoking compiler internals!
    verify_cmd = (
        "import json, sys; "
        "from vld.models.procedural import verify_compiled_capsule, evaluate_compiled_capsule; "
        "cap = json.loads(sys.stdin.read()); "
        "rep = verify_compiled_capsule(cap); "
        "val = [str(v) for v in evaluate_compiled_capsule(cap, '1/2')]; "
        "print(json.dumps({'report': rep, 'val': val}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", verify_cmd],
        input=capsule_json,
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout)
    assert out["report"]["status"] == "VERIFIED_WHOLE_DOMAIN"
    # At x = 1/2 (in [-2, 3/2]): ch0 = 2*(1/2)+5 = 6, ch1 = 0*(1/2)+1 = 1
    assert out["val"] == ["6", "1"]

    # Discontinuous atlas must be rejected!
    with pytest.raises(ProceduralModelError, match="Discontinuous"):
        compile_piecewise_affine_relu(
            [
                {"left": "-inf", "right": "0", "affines": [["1", "0"]]},
                {"left": "0", "right": "+inf", "affines": [["1", "5"]]},
            ]
        )

    # 2. Constructive FSM-to-Neural compilation
    fsm = GroundTransitionSystem(
        states=("S0", "S1", "S2"),
        actions=("inc", "reset"),
        initial_state="S0",
        channels=("parity",),
        observations=(
            TransitionObservation("S0", "inc", (TransitionOutcome("S1", (1,)),)),
            TransitionObservation("S1", "inc", (TransitionOutcome("S2", (0,)),)),
            TransitionObservation("S2", "inc", (TransitionOutcome("S0", (1,)),)),
            TransitionObservation("S0", "reset", (TransitionOutcome("S0", (0,)),)),
            TransitionObservation("S1", "reset", (TransitionOutcome("S0", (0,)),)),
            TransitionObservation("S2", "reset", (TransitionOutcome("S0", (0,)),)),
        ),
    )
    neural_fsm = compile_fsm_to_neural(fsm)
    assert neural_fsm.step_neural("S1", "inc") == ("S2", (0,))
    assert neural_fsm.step_neural("S2", "reset") == ("S0", (0,))

    # 3. Dyadic Net2Wider, exact ReLU identity depth insertion, and hidden permutation
    block = FeedForwardBlock(
        up_weight=((0.5, -0.25, 1.0), (-0.75, 0.5, 0.25)),
        up_bias=(0.125, -0.25, 0.5),
        down_weight=((1.0, -0.5), (0.25, 0.75), (-0.5, 1.25)),
        down_bias=(0.25, -0.125),
        activation="gelu_tanh",
    )
    widened, cert = widen_block_dyadic(block, target_width=5)
    assert cert["morphism"] == "dyadic_net2wider"
    permuted = permute_hidden_units(widened, [4, 2, 0, 3, 1])
    id_layer = insert_identity_depth(2)

    probes = [[-1.5, 2.25], [0.0, -0.75], [3.0, 1.125]]
    y_orig = block.evaluate(probes)
    y_wide = widened.evaluate(probes)
    y_perm = permuted.evaluate(probes)
    y_after_id = id_layer.evaluate(y_orig)

    assert NumericEngine.max_abs_diff(y_orig, y_wide) < 1e-6
    assert NumericEngine.max_abs_diff(y_orig, y_perm) < 1e-6
    assert NumericEngine.max_abs_diff(y_orig, y_after_id) < 1e-6

    # Subnormal underflow during dyadic halving must be refused!
    tiny = float.fromhex("0x1.0p-126")
    tiny_block = FeedForwardBlock(((1.0, 1.0),), (0.0, 0.0), ((tiny,), (1.0,)), (0.0,))
    with pytest.raises(ProceduralModelError, match="underflow"):
        widen_block_dyadic(tiny_block, 3)

    # 4. GenerativeWeightGround backed by StructuralMemoryRegistry + counterexample invalidation
    reg = StructuralMemoryRegistry()
    byte_samples = [(17 + 29 * t) % 256 for t in range(8)]
    spec, _ = reg.certify_and_admit("weights.law0", byte_samples, domain="Z/256Z", declared_modulus=256)
    p_space = ModelParameterSpace("gen.test", {"v": 1}, (TensorSpace("w0", "U8", (16, 16)),))
    recipe = GenerativeWeightGround.build_structural_law_recipe(
        p_space,
        "w0",
        reg,
        memory_id="weights.law0",
        revision=spec.revision,
        chunk_size=128,
    )
    chunk0 = GenerativeWeightGround.read_verified_chunk(p_space, recipe, 0, registry=reg)
    assert list(chunk0[:8]) == byte_samples

    # Injecting a counterexample at byte 50 invalidates chunk 0 reconstruction!
    reg.inject_counterexample("weights.law0", 50, 255 if ((17 + 29 * 50) % 256) != 255 else 0)
    with pytest.raises(ProceduralModelError):
        GenerativeWeightGround.read_verified_chunk(p_space, recipe, 0, registry=reg)


def test_proof_ground_and_autonomous_model_agent_with_hypervisor_refusals() -> None:
    # 1. ProofGround quarantines untrusted claims and propagates elaborated AND/OR subgoals
    trace = {
        "environment": {
            "assistant": "Lean4",
            "assistant_version": "4.14.0",
            "environment_fingerprint": "env-sha-001",
            "imports_fingerprint": "imp-sha-001",
            "options_fingerprint": "opt-sha-001",
        },
        "format": "vld.proof-trace.v1",
        "kernel_claims": ["root_thm"],
        "obligations": [
            {"goal": {"expr": "A /\\ B"}, "key": "root_thm"},
            {"goal": {"expr": "A"}, "key": "sub_a"},
            {"goal": {"expr": "B"}, "key": "sub_b"},
        ],
        "transitions": [
            {
                "attempt_status": "elaborated",
                "source": "root_thm",
                "subgoals": ["sub_a", "sub_b"],
                "tactic": "constructor",
            }
        ],
    }
    pg_untrusted = ProofGroundAdapter.from_trace(trace)
    assert pg_untrusted.summary()["kernel_accepted_obligations"] == 0
    assert len(pg_untrusted.provable_obligations()) == 0  # untrusted claim was NOT promoted!

    root_id, sub_a_id, sub_b_id = pg_untrusted.obligation_ids
    pg_verified = ProofGroundAdapter.from_trace(
        trace,
        verified_receipts=(
            VerifiedKernelReceipt(sub_a_id, "term-a", "env-sha-001", "lean-kernel", "4.14.0", "accepted", "rcpt-a"),
            VerifiedKernelReceipt(sub_b_id, "term-b", "env-sha-001", "lean-kernel", "4.14.0", "accepted", "rcpt-b"),
        ),
    )
    assert root_id in pg_verified.provable_obligations()

    # 2. Autonomous in-layer model agent explores, certifies, and navigates 10^12 steps!
    hypervisor = ReceiverHypervisor()
    grant = create_default_exploration_grant("agent-alpha", probe_budget=Fraction(15), allowed_memory_prefix="agent.")
    hypervisor.register_agent(grant)
    agent = InLayerModelAgent("agent-alpha", hypervisor)

    # Hidden environment: x_t = (50 + 19 * t) mod 61 -> wraps on step 1 (50 -> 8 -> 27 -> 46)
    report = agent.explore_certify_and_navigate(
        lambda t: (50 + 19 * t) % 61,
        memory_id="agent.env61",
        target_coordinate=10**12,
    )
    assert report.probes_executed == 4
    assert report.navigated_value == (50 + 19 * (10**12)) % 61

    # 3. Adversarial / ungrounded proposals through the EXACT same ReceiverHypervisor interface are refused!
    # (a) Unwrapped underdetermined sequence [5, 10, 15, 20]
    with pytest.raises(HypervisorRefusalError, match="refused certification"):
        hypervisor.propose_structural_law("agent-alpha", memory_id="agent.bad", samples=[5, 10, 15, 20])

    # (b) Out-of-scope memory namespace
    with pytest.raises(HypervisorRefusalError, match="outside agent scope"):
        hypervisor.propose_structural_law("agent-alpha", memory_id="system.root", samples=[90, 95, 3, 8, 13])

    # (c) Stale / forged CAS state digest
    with pytest.raises(CASConflictError):
        hypervisor.cas_commit("agent-alpha", expected_state_digest="f" * 64, next_state={"hacked": True})

    # (d) Capability-restricted agent attempting orbit navigation
    hypervisor.register_agent(
        CapabilityGrant("agent-readonly", frozenset({AgentCapability.OBSERVE_TELEMETRY}), probe_budget=Fraction(1))
    )
    with pytest.raises(HypervisorRefusalError, match="lacks capability"):
        hypervisor.navigate_certified_orbit("agent-readonly", memory_id="agent.env61", coordinate=10)

    # (e) Budget exhaustion refusal
    hypervisor.register_agent(
        CapabilityGrant("agent-tinybudget", frozenset({AgentCapability.SELECT_PROBE}), probe_budget=Fraction(1))
    )
    hypervisor.authorize_probe("agent-tinybudget", probe_label="p1", cost=Fraction(1))
    with pytest.raises(HypervisorRefusalError, match="exceeds remaining budget"):
        hypervisor.authorize_probe("agent-tinybudget", probe_label="p2", cost=Fraction(1))

    # Every accepted and refused operation is hash-chained in the hypervisor journal!
    hypervisor.journal.verify()
    refusal_events = [e for e in hypervisor.journal.events if e.operation == "hypervisor.refusal"]
    assert len(refusal_events) == 5


def test_partial_prefix_cache_hit_with_mid_layer_prefill_crash_and_remote_kv_tamper() -> None:
    arch = TransformerArchitectureSpec(
        source_model="Qwen/Qwen3-PrefixCrash",
        revision="d" * 40,
        model_type="qwen3",
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=32,
        vocab_size=24,
        max_position_embeddings=64,
        tie_word_embeddings=True,
        dtype="BF16",
    )
    raw_weights = _make_deterministic_weights(arch)
    store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    snap = ModelWeightSnapshot.publish_tensors(arch.build_parameter_space(), store, raw_weights)

    # Tampered remote KV bytes must be rejected by SHA-256 verification!
    kv_check = PagedKVCacheGround(store)
    kv_page = kv_check.store_layer_kv(
        arch,
        layer_index=0,
        token_prefix=(1, 2),
        k_3d=[[[0.5] * 4, [0.25] * 4], [[0.1] * 4, [0.2] * 4]],
        v_3d=[[[0.3] * 4, [0.4] * 4], [[0.6] * 4, [0.7] * 4]],
    )
    with pytest.raises(ModelOrbitError, match="SHA-256"):
        kv_check.load_layer_kv(kv_page, remote_range_fetcher=lambda _d, _o, ln: b"\x00" * ln)

    # Host A populates prefix cache for (3, 7), then starts (3, 7, 11, 15) hitting prefix (3, 7)
    # and crashes mid-layer (after layer 0) during suffix prefill of (11, 15)!
    runner_ref = AutoregressiveModelRunner(arch, snap, store)
    expected_full = runner_ref.generate((3, 7, 11, 15), max_new_tokens=2)

    runner_a = AutoregressiveModelRunner(arch, snap, store)
    runner_a.generate((3, 7), max_new_tokens=1)  # caches prefix (3, 7) across all layers
    hit_len, _ = runner_a.kv_ground.lookup_longest_prefix(arch, (3, 7, 11, 15))
    assert hit_len == 2

    captured: list[ModelOrbitCheckpoint] = []

    def _crash_suffix_prefill(ckpt: ModelOrbitCheckpoint, gen_idx: int, layer_idx: int) -> None:
        if gen_idx == 0 and layer_idx == 0:
            captured.append(ckpt)
            raise RuntimeError("simulated_mid_prefill_crash")

    with pytest.raises(RuntimeError, match="simulated_mid_prefill_crash"):
        runner_a.generate((3, 7, 11, 15), max_new_tokens=2, on_checkpoint=_crash_suffix_prefill)

    assert len(captured) == 1
    assert captured[0].active_step_tokens == (11, 15)

    # Fresh Host B resumes mid-prefill checkpoint, matches uninterrupted tokens/logits, AND indexes full prefix!
    runner_b = AutoregressiveModelRunner(arch, snap, store)
    with pytest.raises(ModelOrbitError, match="disagrees"):
        runner_b.generate((9, 9, 9, 9), max_new_tokens=2, resume_checkpoint=captured[0])

    resumed = runner_b.generate((3, 7, 11, 15), max_new_tokens=2, resume_checkpoint=captured[0])
    assert resumed.generated_tokens == expected_full.generated_tokens
    assert NumericEngine.max_abs_diff(resumed.last_step_logits, expected_full.last_step_logits) < 1e-6
    pref_b, _ = runner_b.kv_ground.lookup_longest_prefix(arch, (3, 7, 11, 15, 20))
    assert pref_b == 4


def test_multivariate_polyhedral_compilation_dyadic_rescaling_replacement_and_hypervisor_scope() -> None:
    from vld.canonical import domain_digest
    from vld.models import (
        compile_multivariate_hinge_relu,
        evaluate_multivariate_capsule,
        rescale_block_dyadic,
        verify_multivariate_capsule,
    )

    # 1. Exact 2D polyhedral ReLU compilation & whole-domain Fourier-Motzkin chamber verification
    mv_cap = compile_multivariate_hinge_relu(
        input_dim=2,
        base_linear=[["1", "-2"], ["0", "1"]],
        base_bias=["3/2", "-1"],
        hinges=[
            {"normal": ["1", "1"], "offset": "-1", "weights": ["2", "-1"]},
            {"normal": ["2", "-1"], "offset": "0", "weights": ["-3", "4"]},
        ],
    )
    mv_rep = verify_multivariate_capsule(mv_cap)
    assert mv_rep["status"] == "VERIFIED_WHOLE_DOMAIN"
    # 3 lines through (0,0) {x0=0, x1=0, 2x0-x1=0} -> 6 cones, cut by x0+x1=1 -> 6 + 4 = 10 Zaslavsky chambers!
    assert mv_rep["chambers_verified"] == 10
    # At x = (1, 1): x0+x1-1 = 1 > 0, 2*x0-x1 = 1 > 0
    # out0 = (1 - 2 + 3/2) + 2*(1) + (-3)*(1) = 1/2 - 1 = -1/2
    # out1 = (0 + 1 - 1) + (-1)*(1) + 4*(1) = 3
    assert evaluate_multivariate_capsule(mv_cap, [1, 1]) == [Fraction(-1, 2), Fraction(3)]

    # Tampered multivariate weight with recomputed digest MUST fail chamber equivalence!
    bad_mv = json.loads(json.dumps(mv_cap))
    bad_mv["network"]["output_bias"][0] = "99"
    bad_mv_body = {k: v for k, v in bad_mv.items() if k != "capsule_digest"}
    bad_mv["capsule_digest"] = domain_digest("vld.snc-multivariate-relu-capsule.v1", bad_mv_body)
    with pytest.raises(ProceduralModelError, match="disagrees with source_spec"):
        verify_multivariate_capsule(bad_mv)

    # 2. Compensated dyadic positive rescaling symmetry on ReLU FeedForwardBlock
    relu_blk = FeedForwardBlock(
        up_weight=((0.5, -1.25), (0.75, 0.25)),
        up_bias=(0.125, -0.5),
        down_weight=((1.5, -0.25), (-0.75, 2.0)),
        down_bias=(0.5, -0.5),
        activation="relu",
    )
    rescaled, r_cert = rescale_block_dyadic(relu_blk, [3, -2])
    assert r_cert["morphism"] == "dyadic_positive_rescale"
    pts = [[-2.0, 1.5], [0.25, -3.0], [1.0, 1.0]]
    assert relu_blk.evaluate(pts) == rescaled.evaluate(pts)  # exact binary32 bit-for-bit equality!

    # 3. GenerativeWeightGround stored & sparse chunk-replacement recipes
    store = PagedObjectStore(InMemoryPageBacking(page_size=64))
    p_space = ModelParameterSpace("delta.test", {"v": 1}, (TensorSpace("w", "U8", (8, 16)),))
    base_bytes = bytes(range(128))
    base_snap = ModelWeightSnapshot.publish_tensors(p_space, store, {"w": base_bytes})
    stored_rec = GenerativeWeightGround.build_stored_recipe(
        p_space, "w", store, source_snapshot_id="snap0", snapshot=base_snap, chunk_size=64
    )
    assert GenerativeWeightGround.read_verified_chunk(
        p_space, stored_rec, 1, store=store, source_snapshots={"snap0": base_snap}
    ) == base_bytes[64:128]

    patch_chunk1 = bytes([255] * 64)
    repl_rec = GenerativeWeightGround.build_replacement_recipe(
        p_space,
        "w",
        store,
        base_snapshot_id="snap0",
        base_snapshot=base_snap,
        replacement_chunks={1: patch_chunk1},
        chunk_size=64,
    )
    assert GenerativeWeightGround.read_verified_chunk(
        p_space, repl_rec, 0, store=store, source_snapshots={"snap0": base_snap}
    ) == base_bytes[:64]
    assert GenerativeWeightGround.read_verified_chunk(p_space, repl_rec, 1) == patch_chunk1

    # 4. ReceiverHypervisor enforces memory scope on navigate_certified_orbit & journals malformed capsule/cost refusals
    hyp = ReceiverHypervisor()
    hyp.structural_memory.certify_and_admit("vault.secret", [90, 95, 3, 8, 13], domain="Z/97Z")
    hyp.register_agent(
        CapabilityGrant(
            "sandboxed",
            frozenset({AgentCapability.NAVIGATE_ORBIT, AgentCapability.ADMIT_NEURAL_CAPSULE, AgentCapability.SELECT_PROBE}),
            allowed_memory_prefix="sandbox.",
        )
    )
    with pytest.raises(HypervisorRefusalError, match="outside agent scope"):
        hyp.navigate_certified_orbit("sandboxed", memory_id="vault.secret", coordinate=0)

    # Multivariate valid capsule is admitted by Hypervisor; malformed capsule dict is refused & journaled (no KeyError!)
    assert hyp.admit_neural_capsule("sandboxed", mv_cap)["status"] == "VERIFIED_WHOLE_DOMAIN"
    empty_body = {"schema": "vld.snc-relu-capsule.v1"}
    malformed_cap = {**empty_body, "capsule_digest": domain_digest("vld.snc-relu-capsule.v1", empty_body)}
    with pytest.raises(HypervisorRefusalError, match="failed independent verification"):
        hyp.admit_neural_capsule("sandboxed", malformed_cap)

    with pytest.raises(HypervisorRefusalError):
        hyp.authorize_probe("sandboxed", probe_label="bad-float", cost=0.5)  # type: ignore[arg-type]
    with pytest.raises(HypervisorRefusalError):
        hyp.authorize_probe("sandboxed", probe_label="bad-neg", cost=0)

    refusals = [e for e in hyp.journal.events if e.operation == "hypervisor.refusal"]
    assert len(refusals) == 4


def test_high_hinge_multivariate_arrangement_scalability_and_chamber_budget() -> None:
    from vld.accel import NumericEngine
    from vld.canonical import domain_digest
    from vld.models import (
        compile_multivariate_hinge_relu,
        evaluate_multivariate_capsule,
        verify_multivariate_capsule,
    )

    # 26 hinges in R^2 (> 2^26 = 67M naive sign combinations!) verified whole-domain in < 1.5s via normal-ray + redundancy elimination!
    hinges_26 = [
        {"normal": [str((i % 7) + 1), str((i % 5) - 2)], "offset": str(-i), "weights": ["1"]}
        for i in range(26)
    ]
    cap_26 = compile_multivariate_hinge_relu(
        input_dim=2,
        base_linear=[["2", "-3"]],
        base_bias=["1"],
        hinges=hinges_26,
    )
    rep_26 = verify_multivariate_capsule(cap_26)
    assert rep_26["status"] == "VERIFIED_WHOLE_DOMAIN"
    assert rep_26["chambers_verified"] > 100

    # Verify exact evaluation at rational query point
    pt = (Fraction(5, 2), Fraction(-3, 4))
    expected = Fraction(1) + 2 * pt[0] - 3 * pt[1] + sum(
        max(Fraction(0), Fraction(int(h["normal"][0])) * pt[0] + Fraction(int(h["normal"][1])) * pt[1] + Fraction(int(h["offset"])))
        for h in hinges_26
    )
    assert evaluate_multivariate_capsule(cap_26, pt) == [expected]

    with pytest.raises(ProceduralModelError, match="max_chambers"):
        verify_multivariate_capsule(cap_26, max_chambers=5)

    # Crafted multivariate capsule with recomputed digests but truncated/mismatched hinge normal dimensions MUST be refused!
    bad_spec = {
        "base_bias": ["0"],
        "base_linear": [["1", "0"]],
        "hinges": [{"normal": ["1", "0", "999"], "offset": "0", "weights": ["1"]}],
        "input_dim": 2,
    }
    bad_body = {
        "network": cap_26["network"],
        "schema": "vld.snc-multivariate-relu-capsule.v1",
        "source_spec": bad_spec,
        "source_spec_digest": domain_digest("vld.multivariate-hinge-spec.v1", bad_spec),
    }
    bad_cap = {**bad_body, "capsule_digest": domain_digest("vld.snc-multivariate-relu-capsule.v1", bad_body)}
    with pytest.raises(ProceduralModelError, match="dimensions"):
        verify_multivariate_capsule(bad_cap)

    # Large-magnitude RMSNorm float32 squaring overflow safety (2e19^2 overflows binary32 without float64 mean_sq!)
    eng_np = NumericEngine("numpy")
    eng_py = NumericEngine("stdlib")
    large_row = [[2.0e19, -2.0e19, 1.0e19, -1.0e19]]
    w_ones = [1.0, 1.0, 1.0, 1.0]
    assert NumericEngine.max_abs_diff(eng_np.rms_norm_2d(large_row, w_ones, 1e-6), eng_py.rms_norm_2d(large_row, w_ones, 1e-6)) < 1e-5


