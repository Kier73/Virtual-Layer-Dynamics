"""Differential, adversarial, and invariant tests for Frontier Models (`DeepSeek-V4.1` & `GLM-5.3`) and `FP8`/`UE8M0`/`FP4`/`mHC` kernels."""

from __future__ import annotations

import json
import math
import warnings

import pytest

from vld.accel import (
    FP4_E2M1_TABLE,
    NumericEngine,
    apply_interleaved_rope_tail,
    compute_yarn_inv_freq,
    decode_f8_e4m3,
    decode_f8_e8m0,
    dequantize_fp4_e2m1_packed,
    dequantize_fp8_blockwise,
    encode_bf16_le,
    encode_f8_e4m3,
    encode_f8_e8m0,
    encode_f32_le,
    encode_fp4_e2m1_packed,
    get_native_accelerator,
    has_numpy,
    hc_split_sinkhorn,
)
from vld.cli import main as cli_main
from vld.continuous import build_compact_deepseek_v41_fixture, build_compact_glm53_fixture
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.models import (
    DeepSeekSharedAttentionState,
    DeepSeekV41ArchitectureSpec,
    DeepSeekV41LayerRealizer,
    DeepSeekV41ReferenceOracle,
    Glm53LayerRealizer,
    HostedModelGlance,
    LatentMLAKVCacheGround,
    LazyFrontierWeightGround,
    ModelArtifactError,
    build_safetensors_bytes,
    glance_model_structure,
    route_moe_noaux_tc,
    verify_mla_matrix_absorption,
)


def test_f8_e4m3_and_ue8m0_codecs_and_nan_rejection() -> None:
    # Exact boundary & subnormal values in F8_E4M3 (fp8_e4m3fn)
    samples = [0.0, -0.0, 2**-9, -7.0 * (2**-9), 2**-6, 1.0, -1.5, 12.0, -240.0, 448.0, -448.0]
    raw = encode_f8_e4m3(samples)
    dec_std = decode_f8_e4m3(raw, use_numpy=False)
    assert dec_std == [0.0 if v == 0.0 else v for v in samples]
    if has_numpy():
        dec_np = decode_f8_e4m3(raw, use_numpy=True)
        assert dec_np == dec_std

    # Out-of-range finite or NaN inputs rejected by encoder
    with pytest.raises(ValueError):
        encode_f8_e4m3([500.0])
    with pytest.raises(ValueError):
        encode_f8_e4m3([float("nan")])

    # Both 0x7F (+NaN) and 0xFF (-NaN) bytes deterministically refused
    for nan_byte in (b"\x7f", b"\xff"):
        with pytest.raises(ValueError, match="NaN"):
            decode_f8_e4m3(nan_byte, use_numpy=False)
        if has_numpy():
            with pytest.raises(ValueError, match="NaN"):
                decode_f8_e4m3(nan_byte, use_numpy=True)

    # F8_E8M0 (UE8M0): exact powers of two 2^(u - 127) for u in [0, 254], 0xFF is NaN
    scales = [2**-127, 2**-16, 0.125, 1.0, 8.0, 2**30]
    s_raw = encode_f8_e8m0(scales)
    assert decode_f8_e8m0(s_raw, use_numpy=False) == scales
    if has_numpy():
        assert decode_f8_e8m0(s_raw, use_numpy=True) == scales

    with pytest.raises(ValueError, match="NaN"):
        decode_f8_e8m0(b"\xff", use_numpy=False)
    if has_numpy():
        with pytest.raises(ValueError, match="NaN"):
            decode_f8_e8m0(b"\xff", use_numpy=True)
    with pytest.raises(ValueError):
        encode_f8_e8m0([-1.0])


def test_fp8_blockwise_three_way_parity_and_corruption_guards() -> None:
    rows, cols = 64, 96
    sr, sc = 2, 3  # 32x32 blocks
    vals = [float(((i * 13 + 7) % 15 - 7) * 0.5) for i in range(rows * cols)]
    w_raw = encode_f8_e4m3(vals)
    s_raw = encode_f8_e8m0([0.25, 0.5, 1.0, 2.0, 0.125, 4.0])

    out_std = dequantize_fp8_blockwise(w_raw, (rows, cols), s_raw, (sr, sc), use_numpy=False)
    if has_numpy():
        out_np = dequantize_fp8_blockwise(w_raw, (rows, cols), s_raw, (sr, sc), use_numpy=True)
        assert out_np == out_std

        # Also check F32 and BF16 scale dtypes
        s_f32 = encode_f32_le([0.25, 0.5, 1.0, 2.0, 0.125, 4.0])
        s_bf16 = encode_bf16_le([0.25, 0.5, 1.0, 2.0, 0.125, 4.0])
        assert dequantize_fp8_blockwise(w_raw, (rows, cols), s_f32, (sr, sc), scale_dtype="F32", use_numpy=True) == out_std
        assert dequantize_fp8_blockwise(w_raw, (rows, cols), s_bf16, (sr, sc), scale_dtype="BF16", use_numpy=False) == out_std

    rust = get_native_accelerator()
    if rust.available:
        out_rs = rust.dequant_f8_e4m3_ue8m0_block_f32(w_raw, rows, cols, s_raw, sc, block_r=32, block_c=32)
        assert out_rs == out_std

        # NaN in weight or scale must fail in Rust CDLL too
        bad_w = bytearray(w_raw)
        bad_w[17] = 0x7F
        with pytest.raises(ValueError):
            rust.dequant_f8_e4m3_ue8m0_block_f32(bytes(bad_w), rows, cols, s_raw, sc)
        bad_s = bytearray(s_raw)
        bad_s[2] = 0xFF
        with pytest.raises(ValueError):
            rust.dequant_f8_e4m3_ue8m0_block_f32(w_raw, rows, cols, bytes(bad_s), sc)

    # Non-multiple tail dimension (48x64 with 32x32 dyadic blocks -> sr=2, sc=2): rows 0..31 MUST use row-block 0!
    tail_w = encode_f8_e4m3([1.0] * (48 * 64))
    tail_s = encode_f8_e8m0([1.0, 1.0, 4.0, 4.0])
    tail_std = dequantize_fp8_blockwise(tail_w, (48, 64), tail_s, (2, 2), use_numpy=False)
    assert tail_std[25][0] == 1.0 and tail_std[35][0] == 4.0
    if has_numpy():
        assert dequantize_fp8_blockwise(tail_w, (48, 64), tail_s, (2, 2), use_numpy=True) == tail_std
    if rust.available:
        assert rust.dequant_f8_e4m3_ue8m0_block_f32(tail_w, 48, 64, tail_s, 2, block_r=32, block_c=32) == tail_std


def test_fp4_e2m1_packed_roundtrip_and_parity() -> None:
    assert len(FP4_E2M1_TABLE) == 16
    row0 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0] * 2
    row1 = [v * 0.25 for v in reversed(row0)]
    packed, scales = encode_fp4_e2m1_packed([row0, row1], block_k=32)
    assert len(packed) == 2 * 16 and len(scales) == 2

    dec_std = dequantize_fp4_e2m1_packed(packed, (2, 32), scales, (2, 1), use_numpy=False)
    expected0 = [0.0 if v == 0.0 else v for v in row0]
    expected1 = [0.0 if v == 0.0 else v for v in row1]
    assert dec_std == [expected0, expected1]
    if has_numpy():
        dec_np = dequantize_fp4_e2m1_packed(packed, (2, 32), scales, (2, 1), use_numpy=True)
        assert dec_np == dec_std


def test_yarn_interleaved_rope_inverse_identity_and_birkhoff_sinkhorn() -> None:
    inv_f = compute_yarn_inv_freq(8, rope_theta=10000.0, factor=4.0, original_seq_len=4096)
    assert len(inv_f) == 4 and all(f > 0.0 for f in inv_f)

    # 3D [seq, heads, nope + rope] conjugate inverse RoPE identity (DeepSeek-V4.1 o-tail inverse RoPE)
    x_3d = [[[(t + 1) * 0.1 + h * 0.2 + d * 0.03 for d in range(16)] for h in range(3)] for t in range(4)]
    rot = apply_interleaved_rope_tail(x_3d, start_position=7, rope_dim=8, factor=4.0, original_seq_len=4096, inverse=False)
    unrot = apply_interleaved_rope_tail(rot, start_position=7, rope_dim=8, factor=4.0, original_seq_len=4096, inverse=True)
    for t in range(4):
        for h in range(3):
            # Prefix nope dims [0..8) untouched
            assert rot[t][h][:8] == x_3d[t][h][:8]
            for d in range(16):
                assert abs(unrot[t][h][d] - x_3d[t][h][d]) < 1e-6

    # 3-way parity on Manifold-Constrained Hyper-Connections (mHC) Birkhoff Sinkhorn
    mixes = [[0.25 * math.sin(t * 1.7 + j * 0.4) for j in range(24)] for t in range(3)]
    scale = [0.3, 0.4, 0.5]
    base = [0.1 * math.cos(j * 0.5) for j in range(24)]
    pre_s, post_s, comb_s = hc_split_sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, use_numpy=False)
    if has_numpy():
        pre_n, post_n, comb_n = hc_split_sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, use_numpy=True)
        assert NumericEngine.max_abs_diff(pre_s, pre_n) < 1e-6
        assert NumericEngine.max_abs_diff(post_s, post_n) < 1e-6
        for t in range(3):
            assert NumericEngine.max_abs_diff(comb_s[t], comb_n[t]) < 1e-6

    rust = get_native_accelerator()
    if rust.available:
        pre_r, post_r, comb_r = rust.sinkhorn_birkhoff_f32(mixes, scale, base, hc_mult=4, sinkhorn_iters=20)
        assert NumericEngine.max_abs_diff(pre_s, pre_r) < 1e-6
        assert NumericEngine.max_abs_diff(post_s, post_r) < 1e-6
        for t in range(3):
            assert NumericEngine.max_abs_diff(comb_s[t], comb_r[t]) < 1e-6

    for t in range(3):
        assert all(v > 0.0 for v in pre_s[t])
        assert all(0.0 < v < 2.0 for v in post_s[t])
        for c in range(4):
            col_sum = sum(comb_s[t][r][c] for r in range(4))
            assert 1.0 - 5e-5 < col_sum <= 1.0 + 1e-6

    assert hc_split_sinkhorn([], scale, base, hc_mult=4) == ([], [], [])
    with pytest.raises(ValueError):
        hc_split_sinkhorn([[float("inf")] * 24], scale, base, hc_mult=4)

    # Extreme finite logits (+-800.0) must emit zero RuntimeWarnings under NumPy and match Rust
    if has_numpy():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pre_ext, _, _ = hc_split_sinkhorn([[800.0, -800.0] * 12], [1.0, 1.0, 1.0], [0.0] * 24, use_numpy=True)
            sw_ext = NumericEngine("numpy").swiglu_2d([[800.0, -800.0]], [[1.0, 1.0]])
            assert sw_ext == [[800.0, 0.0]]
            if rust.available:
                pre_r_ext, _, _ = rust.sinkhorn_birkhoff_f32([[800.0, -800.0] * 12], [1.0, 1.0, 1.0], [0.0] * 24)
                assert NumericEngine.max_abs_diff(pre_ext, pre_r_ext) < 1e-6


def test_hosted_model_glance_profiles_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    ds = HostedModelGlance.from_published_profile("deepseek-v4.1-flash")
    assert ds.kv_compression_factor == 128.0 and ds.model_type == "deepseek_v41_text"
    assert ds.active_experts_per_tok == 6 and ds.routed_experts == 384 and ds.hyper_connection_mult == 4
    assert len(ds.to_virtual_ground_spec().fingerprint) == 64

    ds_pro = HostedModelGlance.from_published_profile("deepseek-v4-pro")
    assert ds_pro.kv_compression_factor == 256.0 and ds_pro.num_attention_heads == 128
    assert ds_pro.shard_count == 66 and ds_pro.declared_checkpoint_bytes == 892_727_580_904

    glm = HostedModelGlance.from_published_profile("glm-5.3-bf16")
    assert round(glm.kv_compression_factor, 2) == 56.89
    assert glm.dense_layer_count == 3 and glm.moe_layer_count == 75 and glm.scoring_func == "sigmoid"

    glm_fl = HostedModelGlance.from_published_profile("glm-5.3-flash-bf16")
    assert glm_fl.kv_compression_factor == 64.0 and glm_fl.hyper_connection_mult == 4
    assert glm_fl.attention_mechanism == "glm5_hybrid_kda_mla"

    with pytest.raises(ModelArtifactError):
        HostedModelGlance.from_published_profile("nonexistent-model")
    with pytest.raises(ModelArtifactError):
        glance_model_structure(repo_id="bad_repo", revision="abc", config_raw=b"{}", index_raw=b"{}")

    assert cli_main(["glance", "--profile", "deepseek-v4.1-flash"]) == 0
    gl_json = json.loads(capsys.readouterr().out)
    assert gl_json["repo_id"] == "deepseek-ai/DeepSeek-V4.1-Flash"
    assert gl_json["kv_compression_factor"] == 128.0


def test_noaux_tc_moe_routing_bias_decoupling() -> None:
    # Bias flips top-2 selection from (0, 1) to (2, 3), while gate weights come strictly from unbiased scores!
    logits = [2.0, 1.5, 0.2, 0.1]
    bias = [0.0, 0.0, 5.0, 5.0]
    for sf in ("sqrtsoftplus", "sigmoid", "softmax"):
        rcpt = route_moe_noaux_tc(
            model_fingerprint="a" * 64,
            layer_index=0,
            token_position=0,
            router_logits=logits,
            correction_bias=bias,
            top_k=2,
            scoring_func=sf,
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            expert_prefix="layers.0.ffn.experts",
        )
        assert set(rcpt.selected_experts) == {2, 3}
        assert math.isclose(sum(rcpt.normalized_weights), 1.5, rel_tol=1e-6)


def test_deepseek_v41_realizer_oracle_parity_and_sparse_expert_loading() -> None:
    store = PagedObjectStore(InMemoryPageBacking(4096))
    arch, raw_wg = build_compact_deepseek_v41_fixture(seed=19)
    # Wrap with M1 cache_store to also test cached zero-re-read behavior
    shard_bytes = raw_wg._shard_sources["dsv41_compact.safetensors"][0](
        0, raw_wg._shard_sources["dsv41_compact.safetensors"][1]()
    )
    wg = LazyFrontierWeightGround.from_safetensors_bytes(shard_bytes, shard_name="dsv41.safetensors", cache_store=store)

    toks = [3, 14, 27]
    oracle_logits = DeepSeekV41ReferenceOracle.forward_full(arch, wg, toks)

    # Pure-stdlib engine vs FP64 reference oracle
    std_runner = DeepSeekV41LayerRealizer(arch, wg, engine=NumericEngine("stdlib"))
    hc_s, pm_s = std_runner.initial_hc_stream(toks)
    l0 = std_runner.forward_layer(0, hc_s, pm_s)
    # With 3 tokens and top_k=2 out of 8 experts, at most 6 experts can be touched; verify sparse loading
    loaded_set = set(l0.loaded_tensor_names)
    selected_experts_l0 = {e for r in l0.moe_receipts for e in r.selected_experts}
    for e_idx in range(arch.n_routed_experts):
        w1_name = f"layers.0.ffn.experts.{e_idx}.w1.weight"
        if e_idx in selected_experts_l0:
            assert w1_name in loaded_set
        else:
            assert w1_name not in loaded_set

    l1 = std_runner.forward_layer(1, l0.hc_stream, l0.next_pre_mix)
    std_logits = std_runner.collapse_and_logits(l1.hc_stream, l1.next_pre_mix)
    assert NumericEngine.max_abs_diff(std_logits, oracle_logits) < 1e-4

    if has_numpy():
        mat_before = wg.tensors_materialized
        np_runner = DeepSeekV41LayerRealizer(arch, wg, engine=NumericEngine("numpy"))
        hc_n, pm_n = np_runner.initial_hc_stream(toks)
        for li in range(arch.num_hidden_layers):
            step = np_runner.forward_layer(li, hc_n, pm_n)
            hc_n, pm_n = step.hc_stream, step.next_pre_mix
        np_logits = np_runner.collapse_and_logits(hc_n, pm_n)
        assert NumericEngine.max_abs_diff(np_logits, oracle_logits) < 1e-4
        # Because identical tokens selected identical experts, M1 cache_store served every tensor with 0 new shard reads!
        assert wg.tensors_materialized == mat_before


def test_glm53_mla_absorption_and_incremental_latent_kv_decode() -> None:
    arch, wg = build_compact_glm53_fixture(seed=23)
    toks = [4, 9, 15, 22]

    for mode in ("stdlib", "numpy") if has_numpy() else ("stdlib",):
        eng = NumericEngine(mode)
        runner_abs = Glm53LayerRealizer(arch, wg, mla_mode="latent_absorbed", engine=eng)
        runner_mat = Glm53LayerRealizer(arch, wg, mla_mode="materialized", engine=eng)

        emb_all = runner_abs.embed_tokens(toks)
        rep = verify_mla_matrix_absorption(arch, wg, emb_all, layer_index=0)
        assert rep["max_abs_diff"] < 1e-5

        # Full 4-token forward across Layer 0 (dense) + Layer 1 (sigmoid noaux_tc MoE)
        h_abs = emb_all
        h_mat = emb_all
        for li in range(arch.num_hidden_layers):
            h_abs = runner_abs.forward_layer(li, h_abs).hidden_states
            h_mat = runner_mat.forward_layer(li, h_mat).hidden_states
        logits_full_abs = runner_abs.final_logits(h_abs)
        logits_full_mat = runner_mat.final_logits(h_mat)
        assert NumericEngine.max_abs_diff(logits_full_abs, logits_full_mat) < 1e-5

        # Incremental prefill (first 3 tokens) + cached decode (4th token) via LatentMLAKVCacheGround!
        kv_ground = LatentMLAKVCacheGround(PagedObjectStore(InMemoryPageBacking(2048)))
        h_pre = emb_all[:3]
        pages = []
        for li in range(arch.num_hidden_layers):
            out_pre = runner_abs.forward_layer(li, h_pre)
            pages.append(kv_ground.store_latent_kv(arch, layer_index=li, c_kv_2d=out_pre.c_kv_full, k_pe_2d=out_pre.k_pe_full))
            h_pre = out_pre.hidden_states

        h_dec = emb_all[3:]
        for li in range(arch.num_hidden_layers):
            past_c, past_r = kv_ground.load_latent_kv(pages[li])
            out_dec = runner_abs.forward_layer(li, h_dec, past_c_kv=past_c, past_k_pe=past_r)
            h_dec = out_dec.hidden_states
        logits_dec = runner_abs.final_logits(h_dec)
        assert NumericEngine.max_abs_diff(logits_dec[0], logits_full_abs[-1]) < 1e-5


def test_deepseek_v41_compressed_kv_indexer_candidate_blocks_and_incremental_decode() -> None:
    arch = DeepSeekV41ArchitectureSpec(
        source_model="deepseek-ai/DeepSeek-V41-Compressed",
        revision="314113f8a7dadb388215396ba5fbee2c610a04d6",
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        moe_intermediate_size=64,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        sliding_window=2,
        compress_rope_theta=160_000.0,
        original_seq_len=65536,
        rope_factor=16.0,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4),
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        candidate_source_layer=3,
        candidate_topk_blocks=2,
        candidate_block_size=2,
    )
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def _det(n: int, salt: int, amp: float = 0.12) -> list[float]:
        return [amp * math.sin((i + 1) * 0.31 + salt * 0.97) for i in range(n)]

    def _fp8(name: str, r: int, c: int, salt: int) -> None:
        tensors[name + ".weight"] = ("F8_E4M3", (r, c), encode_f8_e4m3(_det(r * c, salt, 0.4)))
        sr, sc = math.ceil(r / 32), math.ceil(c / 32)
        tensors[name + ".scale"] = ("F8_E8M0", (sr, sc), encode_f8_e8m0([0.125] * (sr * sc)))

    def _fp4(name: str, r: int, c: int, salt: int) -> None:
        vals = _det(r * c, salt, 0.2)
        p_raw, s_raw = encode_fp4_e2m1_packed([vals[i * c : (i + 1) * c] for i in range(r)], block_k=32)
        tensors[name + ".weight"] = ("I8", (r, c // 2), p_raw)
        tensors[name + ".scale"] = ("F8_E8M0", (r, c // 32), s_raw)

    tensors["embed.weight"] = ("BF16", (32, 64), encode_bf16_le(_det(32 * 64, 1, 0.2)))
    tensors["norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
    tensors["head.weight"] = ("BF16", (32, 64), encode_bf16_le(_det(32 * 64, 2, 0.2)))
    mix_hc = 24
    for li in range(5):
        p = f"layers.{li}."
        s = 100 * (li + 1)
        tensors[p + "attn_norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
        tensors[p + "ffn_norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
        tensors[p + "attn.q_norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
        tensors[p + "attn.kv_norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
        tensors[p + "attn.attn_sink"] = ("F32", (4,), encode_f32_le([-1.2, -0.8, -1.5, -0.4]))
        _fp8(p + "attn.wq_a", 32, 64, s + 1)
        _fp8(p + "attn.wq_b", 128, 32, s + 2)
        _fp8(p + "attn.wkv", 32, 64, s + 3)
        _fp8(p + "attn.wo_a", 64, 64, s + 4)
        _fp8(p + "attn.wo_b", 64, 64, s + 5)
        if li in arch.kv_source_layers:
            tensors[p + "attn.compressor.norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
            tensors[p + "attn.compressor.wkv.weight"] = ("F32", (32, 64), encode_f32_le(_det(32 * 64, s + 30)))
            if arch.compress_ratios[li] > 1:
                tensors[p + "attn.compressor.wgate.weight"] = ("F32", (32, 64), encode_f32_le(_det(32 * 64, s + 31)))
        if li in arch.index_source_layers:
            _fp8(p + "attn.indexer.wq_b", 64, 32, s + 32)
            tensors[p + "attn.indexer.weights_proj.weight"] = ("BF16", (2, 64), encode_bf16_le(_det(2 * 64, s + 33)))
            if li in arch.kv_source_layers:
                tensors[p + "attn.indexer.wk.weight"] = ("BF16", (32, 32), encode_bf16_le(_det(32 * 32, s + 34)))
                tensors[p + "attn.indexer.k_norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
        tensors[p + "hc_attn_fn"] = ("F32", (mix_hc, 256), encode_f32_le(_det(mix_hc * 256, s + 6, 0.05)))
        tensors[p + "hc_attn_base"] = ("F32", (mix_hc,), encode_f32_le(_det(mix_hc, s + 7, 0.1)))
        tensors[p + "hc_attn_scale"] = ("F32", (3,), encode_f32_le([0.2, 0.2, 0.2]))
        tensors[p + "hc_ffn_fn"] = ("F32", (mix_hc, 256), encode_f32_le(_det(mix_hc * 256, s + 8, 0.05)))
        tensors[p + "hc_ffn_base"] = ("F32", (mix_hc,), encode_f32_le(_det(mix_hc, s + 9, 0.1)))
        tensors[p + "hc_ffn_scale"] = ("F32", (3,), encode_f32_le([0.2, 0.2, 0.2]))
        tensors[p + "ffn.gate.weight"] = ("BF16", (4, 64), encode_bf16_le(_det(4 * 64, s + 10, 0.3)))
        tensors[p + "ffn.gate.bias"] = ("F32", (4,), encode_f32_le(_det(4, s + 11, 0.05)))
        _fp8(p + "ffn.shared_experts.w1", 64, 64, s + 12)
        _fp8(p + "ffn.shared_experts.w2", 64, 64, s + 13)
        _fp8(p + "ffn.shared_experts.w3", 64, 64, s + 14)
        for ei in range(4):
            _fp4(f"{p}ffn.experts.{ei}.w1", 64, 64, s + 20 + 3 * ei)
            _fp4(f"{p}ffn.experts.{ei}.w2", 64, 64, s + 21 + 3 * ei)
            _fp4(f"{p}ffn.experts.{ei}.w3", 64, 64, s + 22 + 3 * ei)

    wg = LazyFrontierWeightGround.from_safetensors_bytes(build_safetensors_bytes(tensors))
    toks = [2, 7, 13, 19, 25, 30]
    oracle_logits = DeepSeekV41ReferenceOracle.forward_full(arch, wg, toks)

    runner = DeepSeekV41LayerRealizer(arch, wg, engine=NumericEngine("numpy" if has_numpy() else "stdlib"))
    hc_s, pm_s = runner.initial_hc_stream(toks)
    shared = DeepSeekSharedAttentionState()
    for li in range(5):
        out = runner.forward_layer(li, hc_s, pm_s, shared_state=shared)
        hc_s, pm_s = out.hc_stream, out.next_pre_mix
        if li == 2:
            # Layer 2 reuses Layer 1's ratio=2 compressed KV & topk_idxs without owning compressor weights
            assert len(out.compressed_kv_selected) == 6
    full_logits = runner.collapse_and_logits(hc_s, pm_s)
    assert NumericEngine.max_abs_diff(full_logits, oracle_logits) < 1e-4

    # Incremental 5-token prefill (leaving 1 trailing partial token in ratio=2 compressor!) + 6th-token decode
    hc_pre, pm_pre = runner.initial_hc_stream(toks[:5])
    shared_inc = DeepSeekSharedAttentionState()
    past_kvs = []
    for li in range(5):
        o_p = runner.forward_layer(li, hc_pre, pm_pre, shared_state=shared_inc)
        past_kvs.append(o_p.kv_cache_full)
        hc_pre, pm_pre = o_p.hc_stream, o_p.next_pre_mix
    hc_dec, pm_dec = runner.initial_hc_stream(toks[5:])
    for li in range(5):
        o_d = runner.forward_layer(li, hc_dec, pm_dec, past_kv=past_kvs[li], shared_state=shared_inc)
        hc_dec, pm_dec = o_d.hc_stream, o_d.next_pre_mix
    dec_logits = runner.collapse_and_logits(hc_dec, pm_dec)
    assert NumericEngine.max_abs_diff(dec_logits[0], full_logits[-1]) < 1e-4
