r"""Frontier Model Scaling, Hosted Structural Glancing, DeepSeek-V4.1 (`mHC` + `FP8`/`FP4` + Sink-MLA) & GLM-5.3 (`MLA` Matrix Absorption + Hybrid `DSA` MoE) Grounds (`vld.models.frontier`).

Governing Mathematical Formulations:
1. **Hosted Model Structural Glancing ($\Lambda_{\text{glance}}$):**
   Lifts remote `config.json` + `model.safetensors.index.json` (+ bounded `[0..8+H)` shard header
   observations) into a versioned `HostedModelGlance` and `VirtualGroundSpec` in $O(|\text{index}|)$
   metadata bytes without transferring weight payloads ($\rho_{\text{bytes}} < 10^{-5}$ on TB-scale models).

2. **Manifold-Constrained Hyper-Connections (`mHC` on Birkhoff Polytope $\mathcal{B}_m$, DeepSeek-V4.1):**
   Carries residual state $H \in \mathbb{R}^{S \times m \times D}$. Each sublayer projects normalized stream
   $\tilde{h}_t \in \mathbb{R}^{mD}$ onto $(\alpha_{\text{pre}}, \alpha_{\text{post}}, C)$ where $C \in \mathcal{B}_m$
   is projected via $K_{\text{sinkhorn}}$ alternating Sinkhorn-Knopp normalizations:
     $$C \mathbf{1} \approx \mathbf{1}, \quad \mathbf{1}^\top C \approx \mathbf{1} \implies \rho(C) \le \|C\|_\infty \le 1,$$
   guaranteeing spectral non-expansiveness of the multi-branch residual recurrence across depth.

3. **Auxiliary-Loss-Free (`noaux_tc`) Biased Top-$k$ MoE Routing (`sqrtsoftplus` & `sigmoid`):**
   Given gate logits $\ell \in \mathbb{R}^E$ and correction bias $b \in \mathbb{R}^E$:
     $$p_e = \psi(\ell_e), \qquad \mathcal{S} = \operatorname{arg\,top}_k(p_e + b_e), \qquad w_e = \gamma \frac{p_e}{\sum_{j \in \mathcal{S}} p_j + 10^{-20}} \quad (e \in \mathcal{S}),$$
   decoupling load-balancing expert selection ($p + b$) from unbiased gradient-preserving combination weights ($p$).

4. **Multi-Head Latent Attention (`MLA`) Matrix Absorption Theorem (GLM-5.3 / DeepSeek MLA):**
   Let $[W_{k,\text{nope}}^{(h)};\; W_v^{(h)}] = W_{kv,b}^{(h)} \in \mathbb{R}^{(d_{\text{nope}} + d_v) \times r_{kv}}$.
   By adjoint bilinearity of the inner product and linearity of expectation over causal weights $\alpha_{t,s}^{(h)}$:
     $$\langle q_{\text{nope}}^{(h)},\; W_{k,\text{nope}}^{(h)} \hat{c}_{kv,s} \rangle = \langle (W_{k,\text{nope}}^{(h)})^\top q_{\text{nope}}^{(h)},\; \hat{c}_{kv,s} \rangle, \qquad \sum_{s \le t} \alpha_{t,s}^{(h)} (W_v^{(h)} \hat{c}_{kv,s}) = W_v^{(h)} \!\left(\sum_{s \le t} \alpha_{t,s}^{(h)} \hat{c}_{kv,s}\right).$$
   Caching only $(\hat{c}_{kv,s}, k_{\text{pe},s}) \in \mathbb{R}^{r_{kv} + d_{\text{rope}}}$ compresses KV memory by
   $\frac{2 n_h d_v}{r_{kv} + d_{\text{rope}}}$ (`56.89x` on `GLM-5.3`) with exact mathematical equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import urllib.request

from vld.accel import (
    NumericEngine,
    apply_interleaved_rope_tail,
    decode_bf16_le,
    decode_f16_le,
    decode_f32_le,
    decode_tensor_ndarray,
    dequantize_fp4_e2m1_packed,
    dequantize_fp8_blockwise,
    encode_f32_le,
    hc_split_sinkhorn,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.ground import VirtualGroundSpec
from vld.memory import PagedByteManifest, PagedObjectStore
from vld.models.artifacts import (
    HttpRangeSource,
    ModelArtifactError,
    SAFETENSORS_DTYPE_WIDTHS,
    SafetensorsShardObservation,
    observe_safetensors_header,
)
from vld.models.realizers import MoERoutingReceipt, RealizerError

_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")


# ============================================================================
# 1. Published Pinned 2026 Frontier Profiles & Hosted Structural Glancer
# ============================================================================

PUBLISHED_FRONTIER_PROFILES: dict[str, dict[str, Any]] = {
    "deepseek-v4.1-flash": {
        "repo_id": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": "314113f8a7dadb388215396ba5fbee2c610a04d6",
        "family": "deepseek_v41",
        "declared_checkpoint_bytes": 510_286_023_000,
        "shard_count": 48,
        "vocab_size": 129280,
        "hidden_size": 5120,
        "num_hidden_layers": 40,
        "num_mtp_layers": 3,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "moe_intermediate_size": 2304,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-20,
        "sliding_window": 128,
        "rope_theta": 10000.0,
        "weight_quant": "fp8_e4m3_32x32+fp4_e2m1_experts",
    },
    "deepseek-v4-pro": {
        "repo_id": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "revision": "72e1d3230f6c080a530b0a1d46f8eb4602340597",
        "family": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_mtp_layers": 1,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "o_groups": 16,
        "moe_intermediate_size": 3072,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 2.5,
        "swiglu_limit": 10.0,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rms_norm_eps": 1e-6,
        "sliding_window": 128,
        "rope_theta": 10000.0,
        "weight_quant": "fp8_e4m3_128x128+fp4_e2m1_experts",
    },
    "glm-5.3-bf16": {
        "repo_id": "zai-org/GLM-5.3-BF16",
        "revision": "abf42e6c0e53bc29a875511301e7b7fcd220315b",
        "family": "glm_moe_dsa",
        "declared_checkpoint_bytes": 1_506_659_919_872,
        "shard_count": 282,
        "vocab_size": 154880,
        "hidden_size": 6144,
        "num_hidden_layers": 78,
        "num_mtp_layers": 1,
        "first_k_dense_replace": 3,
        "intermediate_size": 12288,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "rms_norm_eps": 1e-5,
        "rope_theta": 8_000_000.0,
        "weight_quant": "bf16",
    },
    "glm-5.3-flash-bf16": {
        "repo_id": "zai-org/GLM-5.3-Flash-BF16",
        "revision": "f9fed3bbc82291073373d67452e9f25317ef7577",
        "family": "glm5_next_text",
        "declared_checkpoint_bytes": 642_646_653_816,
        "shard_count": 120,
        "vocab_size": 154880,
        "hidden_size": 4096,
        "num_hidden_layers": 45,
        "num_mtp_layers": 1,
        "first_k_dense_replace": 3,
        "intermediate_size": 12288,
        "num_attention_heads": 64,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "v_head_dim": 256,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 288,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 2.5,
        "hc_mult": 4,
        "rms_norm_eps": 1e-5,
        "weight_quant": "bf16",
    },
}


@dataclass(frozen=True, slots=True)
class HostedModelGlance:
    """Zero-weight-download structural lift of a hosted LLM/VLM checkpoint into VLD."""

    repo_id: str
    revision: str
    model_type: str
    architectures: tuple[str, ...]
    backbone_layers: int
    mtp_layers: int
    dense_layer_count: int
    moe_layer_count: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    attention_mechanism: str
    kv_cache_floats_per_token_layer: int
    naive_mha_kv_floats_per_token_layer: int
    kv_compression_factor: float
    routed_experts: int
    shared_experts: int
    active_experts_per_tok: int
    scoring_func: str
    hyper_connection_mult: int
    quantization_summary: str
    shard_count: int
    total_tensor_count: int
    declared_checkpoint_bytes: int
    metadata_bytes_read: int
    sampled_shard_headers: tuple[str, ...]
    config_sha256: str
    index_sha256: str

    @property
    def expert_sparsity_ratio(self) -> float:
        if self.routed_experts <= 0:
            return 1.0
        return float(self.active_experts_per_tok) / float(self.routed_experts)

    @property
    def glance_digest(self) -> str:
        return domain_digest("vld.hosted-model-glance.v1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_experts_per_tok": self.active_experts_per_tok,
            "architectures": list(self.architectures),
            "attention_mechanism": self.attention_mechanism,
            "backbone_layers": self.backbone_layers,
            "config_sha256": self.config_sha256,
            "declared_checkpoint_bytes": self.declared_checkpoint_bytes,
            "dense_layer_count": self.dense_layer_count,
            "hidden_size": self.hidden_size,
            "hyper_connection_mult": self.hyper_connection_mult,
            "index_sha256": self.index_sha256,
            "kv_cache_floats_per_token_layer": self.kv_cache_floats_per_token_layer,
            "kv_compression_factor": round(self.kv_compression_factor, 6),
            "metadata_bytes_read": self.metadata_bytes_read,
            "model_type": self.model_type,
            "moe_layer_count": self.moe_layer_count,
            "mtp_layers": self.mtp_layers,
            "naive_mha_kv_floats_per_token_layer": self.naive_mha_kv_floats_per_token_layer,
            "num_attention_heads": self.num_attention_heads,
            "quantization_summary": self.quantization_summary,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "routed_experts": self.routed_experts,
            "sampled_shard_headers": list(self.sampled_shard_headers),
            "scoring_func": self.scoring_func,
            "shard_count": self.shard_count,
            "shared_experts": self.shared_experts,
            "total_tensor_count": self.total_tensor_count,
            "vocab_size": self.vocab_size,
        }

    def to_virtual_ground_spec(self) -> VirtualGroundSpec:
        return VirtualGroundSpec(
            ground_id=f"model.glance.{self.repo_id.replace('/', '.')}",
            version="1.0",
            language=f"frontier-llm:{self.model_type}:{self.attention_mechanism}",
            state_domain=f"layers={self.backbone_layers}+mtp={self.mtp_layers}:hc={self.hyper_connection_mult}",
            transitions=("embed_slice", "forward_layer", "route_moe_noaux_tc", "decode_logits"),
            invariants=(
                f"config_sha256={self.config_sha256}",
                f"index_sha256={self.index_sha256}",
                f"kv_compression={self.kv_compression_factor:.2f}x",
                f"moe_active={self.active_experts_per_tok}/{self.routed_experts}+{self.shared_experts}shared",
            ),
            observers=("hidden_state_digest", "moe_routing_receipt", "logits_top1"),
            evidence_scope="hosted-safetensors-header-and-config-glance",
        )

    @classmethod
    def from_published_profile(cls, name: str = "deepseek-v4.1-flash") -> HostedModelGlance:
        """Construct a deterministic offline `HostedModelGlance` from `PUBLISHED_FRONTIER_PROFILES`."""
        key = name.strip().lower()
        if key not in PUBLISHED_FRONTIER_PROFILES:
            raise ModelArtifactError(
                f"Unknown frontier profile {name!r}; expected one of {sorted(PUBLISHED_FRONTIER_PROFILES)}."
            )
        p = dict(PUBLISHED_FRONTIER_PROFILES[key])
        repo_id = str(p["repo_id"])
        rev = str(p["revision"])
        cfg = {
            **p,
            "architectures": [
                "DeepseekV41ForCausalLM"
                if p["family"].startswith("deepseek")
                else ("GlmMoeDsaForCausalLM" if p["family"] == "glm_moe_dsa" else "Glm5NextForConditionalGeneration")
            ],
            "dtype": p.get("weight_quant", "bfloat16"),
            "model_type": p["family"],
            "num_nextn_predict_layers": int(p.get("num_mtp_layers", 0)),
        }
        shard_cnt = int(p.get("shard_count", 1))
        wmap = {f"shard.{i}.placeholder": f"model-{i + 1:05d}-of-{shard_cnt:05d}.safetensors" for i in range(shard_cnt)}
        idx = {
            "metadata": {"total_size": int(p.get("declared_checkpoint_bytes", 0))},
            "weight_map": wmap,
        }
        return glance_model_structure(
            repo_id=repo_id,
            revision=rev,
            config_raw=canonical_bytes(cfg),
            index_raw=canonical_bytes(idx),
        )


def glance_model_structure(
    *,
    repo_id: str,
    revision: str,
    config_raw: bytes,
    index_raw: bytes,
    sampled_observations: Sequence[SafetensorsShardObservation] = (),
) -> HostedModelGlance:
    """Deterministically extract a `HostedModelGlance` from raw `config.json` and `index.json` bytes."""
    if not _REPO_RE.fullmatch(repo_id) or not _HEX40_RE.fullmatch(revision):
        raise ModelArtifactError("glance_model_structure requires 'owner/name' and 40-char Git SHA.")
    cfg_root = parse_strict_json(config_raw)
    idx_root = parse_strict_json(index_raw)
    if not isinstance(cfg_root, dict) or not isinstance(idx_root, dict):
        raise ModelArtifactError("config.json and index.json must be JSON objects.")

    archs = tuple(str(x) for x in cfg_root.get("architectures", ()))
    tcfg = cfg_root["text_config"] if isinstance(cfg_root.get("text_config"), dict) else cfg_root
    model_type = str(tcfg.get("model_type") or cfg_root.get("model_type") or "unknown")
    qcfg = cfg_root.get("quantization_config") or tcfg.get("quantization_config") or {}

    layers = int(tcfg["num_hidden_layers"])
    mtp_layers = int(tcfg.get("num_nextn_predict_layers", 0))
    hidden_size = int(tcfg["hidden_size"])
    vocab_size = int(tcfg["vocab_size"])
    n_heads = int(tcfg["num_attention_heads"])
    routed_e = int(tcfg.get("n_routed_experts") or tcfg.get("num_experts") or 0)
    shared_e = int(tcfg.get("n_shared_experts") or 0)
    top_k = int(tcfg.get("num_experts_per_tok") or 0)
    scoring = str(tcfg.get("scoring_func", "softmax" if routed_e > 0 else "none"))
    hc_mult = int(tcfg.get("hc_mult", 1))

    first_dense = int(tcfg.get("first_k_dense_replace", 0 if routed_e > 0 else layers))
    if "mlp_layer_types" in tcfg and isinstance(tcfg["mlp_layer_types"], list):
        dense_cnt = sum(1 for x in tcfg["mlp_layer_types"] if x == "dense")
        moe_cnt = layers - dense_cnt
    elif routed_e > 0:
        dense_cnt = min(layers, first_dense)
        moe_cnt = layers - dense_cnt
    else:
        dense_cnt = layers
        moe_cnt = 0

    # Attention & KV cache mathematical shape classification
    if model_type in ("deepseek_v41_text", "deepseek_v41", "deepseek_v4"):
        attn_mech = "deepseek_v4_latent_sink_mla"
        hd = int(tcfg["head_dim"])
        kv_floats = hd  # single shared KV head of width head_dim!
        naive_kv = 2 * n_heads * hd
    elif "kv_lora_rank" in tcfg:
        kv_lora = int(tcfg["kv_lora_rank"])
        rope_d = int(tcfg.get("qk_rope_head_dim", 0))
        v_hd = int(tcfg.get("v_head_dim") or tcfg.get("head_dim") or (hidden_size // n_heads))
        attn_mech = "glm5_hybrid_kda_mla" if "linear_attn_config" in tcfg else "multi_head_latent_attention_dsa"
        kv_floats = kv_lora + rope_d
        naive_kv = 2 * n_heads * v_hd
    else:
        attn_mech = "causal_grouped_query_attention"
        hd = int(tcfg.get("head_dim") or (hidden_size // n_heads))
        n_kv = int(tcfg.get("num_key_value_heads", n_heads))
        kv_floats = 2 * n_kv * hd
        naive_kv = 2 * n_heads * hd

    kv_comp = float(naive_kv) / float(max(1, kv_floats))

    if isinstance(qcfg, dict) and qcfg:
        q_method = str(qcfg.get("quant_method", "quant"))
        blk = qcfg.get("weight_block_size")
        exp_dt = qcfg.get("expert_dtype") or cfg_root.get("expert_dtype")
        quant_str = f"{q_method}:block={blk}:scale={qcfg.get('scale_fmt', 'f32')}" + (f":experts={exp_dt}" if exp_dt else "")
    else:
        quant_str = str(tcfg.get("dtype") or cfg_root.get("dtype") or cfg_root.get("torch_dtype") or "bfloat16")

    wmap = idx_root.get("weight_map", {})
    if not isinstance(wmap, dict) or not wmap:
        raise ModelArtifactError("index.json missing non-empty weight_map.")
    shards = sorted({str(v) for v in wmap.values()})
    meta = idx_root.get("metadata") or {}
    total_bytes = int(meta.get("total_size", 0)) if isinstance(meta, dict) else 0
    sampled_hdrs = tuple(f"{o.shard_name}:{o.header_sha256[:16]}" for o in sampled_observations)
    meta_read = len(config_raw) + len(index_raw) + sum(8 + o.header_length_bytes for o in sampled_observations)

    return HostedModelGlance(
        repo_id=repo_id,
        revision=revision,
        model_type=model_type,
        architectures=archs,
        backbone_layers=layers,
        mtp_layers=mtp_layers,
        dense_layer_count=dense_cnt,
        moe_layer_count=moe_cnt,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=n_heads,
        attention_mechanism=attn_mech,
        kv_cache_floats_per_token_layer=kv_floats,
        naive_mha_kv_floats_per_token_layer=naive_kv,
        kv_compression_factor=kv_comp,
        routed_experts=routed_e,
        shared_experts=shared_e,
        active_experts_per_tok=top_k,
        scoring_func=scoring,
        hyper_connection_mult=hc_mult,
        quantization_summary=quant_str,
        shard_count=len(shards),
        total_tensor_count=len(wmap),
        declared_checkpoint_bytes=total_bytes,
        metadata_bytes_read=meta_read,
        sampled_shard_headers=sampled_hdrs,
        config_sha256=hashlib.sha256(config_raw).hexdigest(),
        index_sha256=hashlib.sha256(index_raw).hexdigest(),
    )


def glance_huggingface_model(
    repo_id: str,
    *,
    revision: str,
    sample_first_shard_header: bool = True,
    timeout_sec: float = 25.0,
) -> HostedModelGlance:
    """Glance at a live Hugging Face Hub checkpoint over HTTPS without downloading tensor weights."""
    if not _REPO_RE.fullmatch(repo_id) or not _HEX40_RE.fullmatch(revision):
        raise ModelArtifactError("glance_huggingface_model requires 'owner/name' and 40-char Git SHA.")
    base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
    headers = {"User-Agent": "vld-structural-glancer/0.5"}

    with urllib.request.urlopen(urllib.request.Request(f"{base}/config.json", headers=headers), timeout=timeout_sec) as rc:
        cfg_raw = rc.read()

    sampled_obs: list[SafetensorsShardObservation] = []
    try:
        with urllib.request.urlopen(
            urllib.request.Request(f"{base}/model.safetensors.index.json", headers=headers),
            timeout=timeout_sec,
        ) as ri:
            idx_raw = ri.read()
        if sample_first_shard_header:
            wmap = json.loads(idx_raw.decode("utf-8"))["weight_map"]
            first_shard = sorted({str(v) for v in wmap.values()})[0]
            src = HttpRangeSource(f"{base}/{first_shard}", timeout_sec=timeout_sec)
            try:
                sampled_obs.append(observe_safetensors_header(src.read_range, src.total_size_bytes, shard_name=first_shard))
            finally:
                src.close()
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise ModelArtifactError(f"Failed glancing {repo_id}: {exc}") from exc
        src = HttpRangeSource(f"{base}/model.safetensors", timeout_sec=timeout_sec)
        try:
            obs = observe_safetensors_header(src.read_range, src.total_size_bytes, shard_name="model.safetensors")
            sampled_obs.append(obs)
            synth_idx = {
                "metadata": {"total_size": obs.file_size_bytes},
                "weight_map": {t.name: "model.safetensors" for t in obs.tensors},
            }
            idx_raw = canonical_json(synth_idx).encode("utf-8")
        finally:
            src.close()

    return glance_model_structure(
        repo_id=repo_id,
        revision=revision,
        config_raw=cfg_raw,
        index_raw=idx_raw,
        sampled_observations=sampled_obs,
    )


# ============================================================================
# 2. Lazy Sharded Weight Ground Supporting 50,000..100,000+ Tensor Frontier Models
# ============================================================================


class LazyFrontierWeightGround:
    """Demand-driven multi-shard Safetensors weight reader supporting BF16, F8_E4M3+UE8M0 & packed FP4 E2M1.

    Unlike eager parameter-space builders, `LazyFrontierWeightGround` indexes `weight_map` strings in $O(1)$
    and observes shard JSON headers ONLY for the shards touched by active layers/experts.
    """

    def __init__(
        self,
        *,
        source_pin: str,
        shard_sources: Mapping[str, tuple[Callable[[int, int], bytes], Callable[[], int]]],
        tensor_to_shard: Mapping[str, str],
        cache_store: PagedObjectStore | None = None,
    ) -> None:
        self.source_pin = source_pin
        self._shard_sources = dict(shard_sources)
        self._tensor_to_shard = dict(tensor_to_shard)
        self.cache_store = cache_store
        self._lock = threading.RLock()
        self._observed_shards: dict[str, SafetensorsShardObservation] = {}
        self._cached_manifests: dict[str, PagedByteManifest] = {}
        self.tensors_materialized = 0
        self.bytes_materialized = 0

    @property
    def identity(self) -> str:
        return domain_digest(
            "vld.lazy-frontier-weight-ground.v1",
            {
                "shard_count": len(self._shard_sources),
                "source_pin": self.source_pin,
                "tensor_count": len(self._tensor_to_shard),
            },
        )

    def has_tensor(self, name: str) -> bool:
        return name in self._tensor_to_shard

    def observe_shard(self, shard_name: str) -> SafetensorsShardObservation:
        with self._lock:
            obs = self._observed_shards.get(shard_name)
            if obs is not None:
                return obs
            if shard_name not in self._shard_sources:
                raise ModelArtifactError(f"Unknown Safetensors shard {shard_name!r}.")
            reader_fn, size_fn = self._shard_sources[shard_name]
            obs = observe_safetensors_header(reader_fn, size_fn(), shard_name=shard_name)
            self._observed_shards[shard_name] = obs
            return obs

    def read_raw_entry(self, name: str) -> tuple[str, tuple[int, ...], bytes]:
        """Return `(dtype, shape, raw_bytes)` for `name`, caching in M1 `cache_store` if configured."""
        with self._lock:
            shard_name = self._tensor_to_shard.get(name)
            if shard_name is None:
                raise ModelArtifactError(f"Tensor {name!r} not present in frontier weight_map.")
            obs = self.observe_shard(shard_name)
            entry = obs.tensor_entry(name)
            if self.cache_store is not None and name in self._cached_manifests:
                return entry.dtype, entry.shape, self.cache_store.read_bytes(self._cached_manifests[name])
            abs_off, byte_len = obs.absolute_byte_range(name)
            reader_fn, _ = self._shard_sources[shard_name]
            raw = reader_fn(abs_off, byte_len)
            if len(raw) != byte_len:
                raise ModelArtifactError(f"Short read for {name!r}: expected {byte_len}, got {len(raw)}.")
            self.tensors_materialized += 1
            self.bytes_materialized += byte_len
            if self.cache_store is not None:
                self._cached_manifests[name] = self.cache_store.put_bytes(raw)
            return entry.dtype, entry.shape, raw

    def read_row_slice_f32(self, name: str, row_indices: Sequence[int], *, use_numpy: bool = True) -> list[list[float]]:
        """Read ONLY specific vocabulary rows `[tid * row_bytes : (tid+1) * row_bytes)` from a huge embedding matrix."""
        if not row_indices:
            return []
        with self._lock:
            shard_name = self._tensor_to_shard.get(name)
            if shard_name is None:
                raise ModelArtifactError(f"Embedding tensor {name!r} not found.")
            obs = self.observe_shard(shard_name)
            entry = obs.tensor_entry(name)
            if len(entry.shape) != 2:
                raise ModelArtifactError(f"Expected 2D embedding tensor for {name!r}, got {entry.shape}.")
            rows, cols = entry.shape
            elem_w = SAFETENSORS_DTYPE_WIDTHS[entry.dtype]
            row_bytes = cols * elem_w
            abs_base, _ = obs.absolute_byte_range(name)
            reader_fn, _ = self._shard_sources[shard_name]
            cache: dict[int, list[float]] = {}
            for r_id in sorted(set(int(i) for i in row_indices)):
                if not (0 <= r_id < rows):
                    raise RealizerError(f"Token ID {r_id} out of embedding bounds [0, {rows}).")
                raw = reader_fn(abs_base + r_id * row_bytes, row_bytes)
                if len(raw) != row_bytes:
                    raise ModelArtifactError(f"Short embedding slice read on {name!r} row {r_id}.")
                self.bytes_materialized += row_bytes
                if entry.dtype == "BF16":
                    cache[r_id] = decode_bf16_le(raw, use_numpy=use_numpy)
                elif entry.dtype == "F16":
                    cache[r_id] = decode_f16_le(raw, use_numpy=use_numpy)
                elif entry.dtype == "F32":
                    cache[r_id] = decode_f32_le(raw, use_numpy=use_numpy)
                else:
                    raise RealizerError(f"Unsupported embedding dtype {entry.dtype!r}.")
            return [list(cache[int(i)]) for i in row_indices]

    def load_float_tensor(
        self,
        weight_name: str,
        loaded_log: list[str],
        *,
        logical_shape: tuple[int, ...] | None = None,
        use_numpy: bool = True,
        as_ndarray: bool = False,
    ) -> Any:
        """Load BF16/F16/F32, block-scaled `F8_E4M3` (+ `.scale` / `.weight_scale_inv`), or packed `FP4` (`I8` + `.scale`)."""
        dtype, shape, raw = self.read_raw_entry(weight_name)
        loaded_log.append(weight_name)

        scale_candidates = ()
        if weight_name.endswith(".weight"):
            stem = weight_name[: -len(".weight")]
            scale_candidates = (stem + ".scale", stem + ".weight_scale_inv")

        scale_name = next((sc for sc in scale_candidates if self.has_tensor(sc)), None)

        if dtype in ("F8_E4M3", "F8_E4M3FN"):
            if len(shape) != 2 or scale_name is None:
                raise RealizerError(f"FP8 tensor {weight_name!r} requires a 2D shape and companion scale tensor.")
            s_dtype, s_shape, s_raw = self.read_raw_entry(scale_name)
            loaded_log.append(scale_name)
            if len(s_shape) != 2:
                raise RealizerError(f"Expected 2D scale tensor for {scale_name!r}, got {s_shape}.")
            return dequantize_fp8_blockwise(
                raw,
                (shape[0], shape[1]),
                s_raw,
                (s_shape[0], s_shape[1]),
                scale_dtype=s_dtype,
                use_numpy=use_numpy,
                as_ndarray=as_ndarray,
            )

        if dtype == "I8" and scale_name is not None:
            # DeepSeek-V4.1 packed MXFP4 (`float4_e2m1fn_x2` stored as I8 [out, in//2] + F8_E8M0 [out, in//32])
            if len(shape) != 2:
                raise RealizerError(f"Packed FP4 tensor {weight_name!r} must be 2D.")
            log_shape = (shape[0], shape[1] * 2) if logical_shape is None else (int(logical_shape[0]), int(logical_shape[1]))
            s_dtype, s_shape, s_raw = self.read_raw_entry(scale_name)
            loaded_log.append(scale_name)
            return dequantize_fp4_e2m1_packed(
                raw,
                log_shape,
                s_raw,
                (s_shape[0], s_shape[1]),
                scale_dtype=s_dtype,
                use_numpy=use_numpy,
                as_ndarray=as_ndarray,
            )

        if as_ndarray:
            return decode_tensor_ndarray(raw, dtype, shape)

        if dtype == "BF16":
            flat = decode_bf16_le(raw, use_numpy=use_numpy)
        elif dtype == "F16":
            flat = decode_f16_le(raw, use_numpy=use_numpy)
        elif dtype == "F32":
            flat = decode_f32_le(raw, use_numpy=use_numpy)
        else:
            raise RealizerError(f"Unsupported tensor dtype {dtype!r} for {weight_name!r}.")

        if len(shape) == 1:
            return flat
        if len(shape) == 2:
            r, c = shape
            return [flat[i * c : (i + 1) * c] for i in range(r)]
        raise RealizerError(f"Unsupported non-ndarray rank {len(shape)} for {weight_name!r}.")

    @classmethod
    def from_safetensors_bytes(
        cls,
        shard_bytes: bytes,
        *,
        shard_name: str = "model.safetensors",
        cache_store: PagedObjectStore | None = None,
    ) -> LazyFrontierWeightGround:
        obs = observe_safetensors_header(lambda o, n: shard_bytes[o : o + n], len(shard_bytes), shard_name=shard_name)
        g = cls(
            source_pin=f"mem://{shard_name}:{obs.header_sha256}",
            shard_sources={shard_name: ((lambda o, n, b=shard_bytes: b[o : o + n]), (lambda b=shard_bytes: len(b)))},
            tensor_to_shard={t.name: shard_name for t in obs.tensors},
            cache_store=cache_store,
        )
        g._observed_shards[shard_name] = obs
        return g

    @classmethod
    def from_huggingface_hub(
        cls,
        repo_id: str,
        *,
        revision: str,
        cache_store: PagedObjectStore | None = None,
        timeout_sec: float = 30.0,
    ) -> tuple[LazyFrontierWeightGround, dict[str, HttpRangeSource]]:
        if not _REPO_RE.fullmatch(repo_id) or not _HEX40_RE.fullmatch(revision):
            raise ModelArtifactError("from_huggingface_hub requires 'owner/name' and 40-char Git SHA.")
        base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
        req = urllib.request.Request(f"{base}/model.safetensors.index.json", headers={"User-Agent": "vld-frontier/0.5"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            idx_raw = resp.read()
        idx_sha = hashlib.sha256(idx_raw).hexdigest()
        wmap = json.loads(idx_raw.decode("utf-8"))["weight_map"]
        shard_names = sorted({str(v) for v in wmap.values()})
        sources = {sn: HttpRangeSource(f"{base}/{sn}", timeout_sec=timeout_sec) for sn in shard_names}
        adapters = {sn: (src.read_range, (lambda s=src: s.total_size_bytes)) for sn, src in sources.items()}
        ground = cls(
            source_pin=f"hf://{repo_id}@{revision}:{idx_sha}",
            shard_sources=adapters,
            tensor_to_shard={str(k): str(v) for k, v in wmap.items()},
            cache_store=cache_store,
        )
        return ground, sources


# ============================================================================
# 3. Unified Auxiliary-Loss-Free (`noaux_tc`) MoE Router (`sqrtsoftplus`, `sigmoid`, `softmax`)
# ============================================================================


def route_moe_noaux_tc(
    *,
    model_fingerprint: str,
    layer_index: int,
    token_position: int,
    router_logits: Sequence[float],
    correction_bias: Sequence[float] | None,
    top_k: int,
    scoring_func: str = "sqrtsoftplus",
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
    expert_prefix: str = "",
    expert_weight_suffixes: tuple[str, str, str] = ("w1.weight", "w2.weight", "w3.weight"),
) -> MoERoutingReceipt:
    r"""Deterministic `noaux_tc` MoE router supporting `sqrtsoftplus` (DeepSeek-V4.1), `sigmoid` (GLM-5.3), and `softmax`."""
    n_exp = len(router_logits)
    if n_exp <= 0 or not (1 <= top_k <= n_exp):
        raise RealizerError("Invalid top_k or empty router_logits.")
    logits = [float(x) for x in router_logits]
    if any(not math.isfinite(x) for x in logits):
        raise RealizerError("Non-finite router_logits.")

    if scoring_func == "sqrtsoftplus":
        # softplus(x) = log(1 + exp(-|x|)) + max(x, 0) for overflow-free numerical stability
        scores = [math.sqrt(math.log1p(math.exp(-abs(x))) + max(x, 0.0)) for x in logits]
    elif scoring_func == "sigmoid":
        scores = [1.0 / (1.0 + math.exp(-x)) if x >= 0.0 else math.exp(x) / (1.0 + math.exp(x)) for x in logits]
    elif scoring_func == "softmax":
        m = max(logits)
        ex = [math.exp(x - m) for x in logits]
        sm = math.fsum(ex)
        scores = [v / sm for v in ex]
    else:
        raise RealizerError(f"Unsupported MoE scoring_func {scoring_func!r}.")

    if correction_bias is not None:
        if len(correction_bias) != n_exp:
            raise RealizerError("correction_bias length mismatch.")
        biased = [s + float(b) for s, b in zip(scores, correction_bias)]
        if any(not math.isfinite(v) for v in biased):
            raise RealizerError("Non-finite MoE correction_bias.")
    else:
        biased = list(scores)

    chosen = sorted(range(n_exp), key=lambda idx: (-biased[idx], idx))[:top_k]
    raw_picked = [scores[idx] for idx in chosen]
    if norm_topk_prob and top_k > 1:
        denom = math.fsum(raw_picked) + 1e-20
        weights = [(v / denom) * float(routed_scaling_factor) for v in raw_picked]
    else:
        weights = [v * float(routed_scaling_factor) for v in raw_picked]

    active_names: list[str] = []
    for eid in chosen:
        base = f"{expert_prefix}{eid}."
        active_names.extend(base + sfx for sfx in expert_weight_suffixes)

    return MoERoutingReceipt(
        model_fingerprint=model_fingerprint,
        layer_index=layer_index,
        token_position=token_position,
        top_k=top_k,
        selected_experts=tuple(chosen),
        normalized_weights=tuple(weights),
        active_tensor_names=tuple(active_names),
        router_logits_digest=domain_digest("vld.router-logits-noaux.v1", {"bias": list(correction_bias or ()), "logits": logits}),
    )


# ============================================================================
# 4. DeepSeek-V4.1 Architecture Spec, Realizer, Runner & Scalar FP64 Oracle
# ============================================================================


@dataclass(frozen=True, slots=True)
class DeepSeekV41ArchitectureSpec:
    """Architectural specification for `DeepseekV41ForCausalLM` (`mHC` + Sink-MLA + `FP8`/`FP4` MoE)."""

    source_model: str
    revision: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-20
    sliding_window: int = 128
    rope_theta: float = 10000.0
    compress_ratios: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not _REPO_RE.fullmatch(self.source_model) or not _HEX40_RE.fullmatch(self.revision):
            raise ModelArtifactError("DeepSeekV41ArchitectureSpec requires valid repo_id and 40-char SHA.")
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0 or self.num_attention_heads <= 0:
            raise ModelArtifactError("Dimensions must be positive.")
        if self.num_attention_heads % self.o_groups != 0 or self.qk_rope_head_dim % 2 != 0:
            raise ModelArtifactError("num_attention_heads must be divisible by o_groups and qk_rope_head_dim even.")
        if not (1 <= self.num_experts_per_tok <= self.n_routed_experts):
            raise ModelArtifactError("Invalid num_experts_per_tok.")
        if not self.compress_ratios:
            object.__setattr__(self, "compress_ratios", tuple(0 for _ in range(self.num_hidden_layers)))

    @classmethod
    def from_published_profile(cls, name: str = "deepseek-v4.1-flash") -> DeepSeekV41ArchitectureSpec:
        p = PUBLISHED_FRONTIER_PROFILES[name.lower()]
        return cls(
            source_model=p["repo_id"],
            revision=p["revision"],
            vocab_size=p["vocab_size"],
            hidden_size=p["hidden_size"],
            num_hidden_layers=p["num_hidden_layers"],
            num_attention_heads=p["num_attention_heads"],
            head_dim=p["head_dim"],
            qk_rope_head_dim=p["qk_rope_head_dim"],
            q_lora_rank=p["q_lora_rank"],
            o_lora_rank=p["o_lora_rank"],
            o_groups=p["o_groups"],
            moe_intermediate_size=p["moe_intermediate_size"],
            n_routed_experts=p["n_routed_experts"],
            n_shared_experts=p["n_shared_experts"],
            num_experts_per_tok=p["num_experts_per_tok"],
            scoring_func=p["scoring_func"],
            routed_scaling_factor=p["routed_scaling_factor"],
            swiglu_limit=p["swiglu_limit"],
            hc_mult=p["hc_mult"],
            hc_sinkhorn_iters=p["hc_sinkhorn_iters"],
            hc_eps=p["hc_eps"],
            rms_norm_eps=p["rms_norm_eps"],
            sliding_window=p["sliding_window"],
            rope_theta=p["rope_theta"],
        )

    @property
    def fingerprint(self) -> str:
        return domain_digest(
            "vld.deepseek-v41-arch.v1",
            {
                "hc_mult": self.hc_mult,
                "head_dim": self.head_dim,
                "hidden_size": self.hidden_size,
                "moe_intermediate_size": self.moe_intermediate_size,
                "n_routed_experts": self.n_routed_experts,
                "num_attention_heads": self.num_attention_heads,
                "num_experts_per_tok": self.num_experts_per_tok,
                "num_hidden_layers": self.num_hidden_layers,
                "o_groups": self.o_groups,
                "o_lora_rank": self.o_lora_rank,
                "q_lora_rank": self.q_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "revision": self.revision,
                "source_model": self.source_model,
                "vocab_size": self.vocab_size,
            },
        )


@dataclass(frozen=True, slots=True)
class DeepSeekV41LayerOutput:
    """Verified output of one DeepSeek-V4.1 `mHC` block including Birkhoff doubly-stochastic telemetry."""

    hc_stream: list[list[list[float]]]  # [seq_len, hc_mult, hidden_size]
    next_pre_mix: list[list[float]]  # [seq_len, hc_mult]
    kv_cache_full: list[list[float]]  # [total_seq_len, head_dim] (single shared KV latent!)
    moe_receipts: tuple[MoERoutingReceipt, ...]
    loaded_tensor_names: tuple[str, ...]
    birkhoff_max_row_col_defect: float


class DeepSeekV41LayerRealizer:
    """Lazy numerical realizer for `DeepseekV41ForCausalLM` (`mHC` + Sink-MLA + `FP8`/`FP4` clamped SwiGLU MoE)."""

    def __init__(
        self,
        arch: DeepSeekV41ArchitectureSpec,
        weights: LazyFrontierWeightGround,
        *,
        engine: NumericEngine | None = None,
    ) -> None:
        self.arch = arch
        self.weights = weights
        self.engine = engine if engine is not None else NumericEngine("auto")

    def initial_hc_stream(self, token_ids: Sequence[int]) -> tuple[list[list[list[float]]], list[list[float]]]:
        """Embed `token_ids` via row-slice locality and initialize `(H_0, identity_pre_mix)`."""
        emb_rows = self.weights.read_row_slice_f32("embed.weight", token_ids, use_numpy=self.engine.uses_numpy)
        m = self.arch.hc_mult
        stream = [[[float(v) for v in row] for _ in range(m)] for row in emb_rows]
        pre_mix = [[1.0 if j == 0 else 0.0 for j in range(m)] for _ in range(len(token_ids))]
        return stream, pre_mix

    def collapse_and_logits(
        self,
        hc_stream: Sequence[Sequence[Sequence[float]]],
        pre_mix: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Collapse final `mHC` stream via `hc_pre`, apply `norm.weight`, and project `head.weight`."""
        loaded: list[str] = []
        collapsed = self._hc_pre(hc_stream, pre_mix)
        norm_w = self.weights.load_float_tensor("norm.weight", loaded, use_numpy=self.engine.uses_numpy)
        normed = self.engine.rms_norm_2d(collapsed, norm_w, self.arch.rms_norm_eps)
        if self.engine.uses_numpy:
            head_nd = self.weights.load_float_tensor("head.weight", loaded, use_numpy=True, as_ndarray=True)
            out_nd = self.engine.linear_x_wt_nd(self.engine.asarray_f32(normed), head_nd)
            return [[float(v) for v in row] for row in out_nd.tolist()]
        head_w = self.weights.load_float_tensor("head.weight", loaded, use_numpy=False)
        return self.engine.linear_x_wt(normed, head_w)

    def _hc_mixes(
        self,
        stream: Sequence[Sequence[Sequence[float]]],
        fn_name: str,
        scale_name: str,
        base_name: str,
        loaded: list[str],
    ) -> tuple[list[list[float]], list[list[float]], list[list[list[float]]], float]:
        seq_len = len(stream)
        m = self.arch.hc_mult
        d = self.arch.hidden_size
        flat_2d = [[float(v) for copy_vec in stream[t] for v in copy_vec] for t in range(seq_len)]
        hc_scale = self.weights.load_float_tensor(scale_name, loaded, use_numpy=self.engine.uses_numpy)
        hc_base = self.weights.load_float_tensor(base_name, loaded, use_numpy=self.engine.uses_numpy)

        if self.engine.uses_numpy:
            import numpy as np

            x64 = np.asarray(flat_2d, dtype=np.float64)
            rsqrt = np.reciprocal(np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + float(self.arch.rms_norm_eps)))
            fn_nd = self.weights.load_float_tensor(fn_name, loaded, use_numpy=True, as_ndarray=True).astype(np.float64)
            mixes = ((x64 @ fn_nd.T) * rsqrt).astype(np.float32).tolist()
        else:
            fn_w = self.weights.load_float_tensor(fn_name, loaded, use_numpy=False)
            mixes = []
            for row in flat_2d:
                inv = 1.0 / math.sqrt(math.fsum(v * v for v in row) / float(m * d) + float(self.arch.rms_norm_eps))
                mixes.append([math.fsum(a * b for a, b in zip(row, w_r)) * inv for w_r in fn_w])

        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult=m,
            sinkhorn_iters=self.arch.hc_sinkhorn_iters,
            eps=self.arch.hc_eps,
            use_numpy=self.engine.uses_numpy,
        )
        max_defect = 0.0
        for mat in comb:
            for r in range(m):
                max_defect = max(max_defect, abs(math.fsum(mat[r]) - 1.0))
            for c in range(m):
                max_defect = max(max_defect, abs(math.fsum(mat[r][c] for r in range(m)) - 1.0))
        return pre, post, comb, max_defect

    @staticmethod
    def _hc_pre(stream: Sequence[Sequence[Sequence[float]]], pre_mix: Sequence[Sequence[float]]) -> list[list[float]]:
        out: list[list[float]] = []
        for copies, weights in zip(stream, pre_mix):
            d = len(copies[0])
            m = len(weights)
            out.append([math.fsum(float(weights[j]) * float(copies[j][c]) for j in range(m)) for c in range(d)])
        return out

    @staticmethod
    def _hc_post(
        sub_out: Sequence[Sequence[float]],
        residual: Sequence[Sequence[Sequence[float]]],
        post: Sequence[Sequence[float]],
        comb: Sequence[Sequence[Sequence[float]]],
    ) -> list[list[list[float]]]:
        out: list[list[list[float]]] = []
        for x_t, res_t, p_t, c_t in zip(sub_out, residual, post, comb):
            m = len(p_t)
            d = len(x_t)
            copies_out: list[list[float]] = []
            for k in range(m):
                pk = float(p_t[k])
                col_c = [float(c_t[j][k]) for j in range(m)]
                copies_out.append([
                    pk * float(x_t[idx]) + math.fsum(col_c[j] * float(res_t[j][idx]) for j in range(m))
                    for idx in range(d)
                ])
            out.append(copies_out)
        return out

    def _expert_swiglu(
        self,
        x_2d: Sequence[Sequence[float]],
        prefix: str,
        loaded: list[str],
        *,
        is_routed_fp4: bool,
    ) -> list[list[float]]:
        d = self.arch.hidden_size
        mid = self.arch.moe_intermediate_size
        lim = self.arch.swiglu_limit
        use_np = self.engine.uses_numpy
        if use_np:
            import numpy as np

            x_nd = self.engine.asarray_f32(x_2d)
            w1 = self.weights.load_float_tensor(
                prefix + "w1.weight",
                loaded,
                logical_shape=(mid, d) if is_routed_fp4 else None,
                use_numpy=True,
                as_ndarray=True,
            )
            w3 = self.weights.load_float_tensor(
                prefix + "w3.weight",
                loaded,
                logical_shape=(mid, d) if is_routed_fp4 else None,
                use_numpy=True,
                as_ndarray=True,
            )
            w2 = self.weights.load_float_tensor(
                prefix + "w2.weight",
                loaded,
                logical_shape=(d, mid) if is_routed_fp4 else None,
                use_numpy=True,
                as_ndarray=True,
            )
            gate = (x_nd @ w1.T).astype(np.float64)
            up = (x_nd @ w3.T).astype(np.float64)
            if lim > 0.0:
                gate = np.minimum(gate, float(lim))
                up = np.clip(up, -float(lim), float(lim))
            sig = np.where(gate >= 0.0, 1.0 / (1.0 + np.exp(-gate)), np.exp(gate) / (1.0 + np.exp(gate)))
            act = ((gate * sig) * up).astype(np.float32)
            out_nd = self.engine.linear_x_wt_nd(act, w2)
            return [[float(v) for v in r] for r in out_nd.tolist()]

        w1_l = self.weights.load_float_tensor(prefix + "w1.weight", loaded, logical_shape=(mid, d) if is_routed_fp4 else None, use_numpy=False)
        w3_l = self.weights.load_float_tensor(prefix + "w3.weight", loaded, logical_shape=(mid, d) if is_routed_fp4 else None, use_numpy=False)
        w2_l = self.weights.load_float_tensor(prefix + "w2.weight", loaded, logical_shape=(d, mid) if is_routed_fp4 else None, use_numpy=False)
        g_rows = self.engine.linear_x_wt(x_2d, w1_l)
        u_rows = self.engine.linear_x_wt(x_2d, w3_l)
        act_rows: list[list[float]] = []
        for gr, ur in zip(g_rows, u_rows):
            ar: list[float] = []
            for gv, uv in zip(gr, ur):
                g_c = min(float(gv), float(lim)) if lim > 0.0 else float(gv)
                u_c = max(-float(lim), min(float(lim), float(uv))) if lim > 0.0 else float(uv)
                sig = 1.0 / (1.0 + math.exp(-g_c)) if g_c >= 0.0 else math.exp(g_c) / (1.0 + math.exp(g_c))
                ar.append((g_c * sig) * u_c)
            act_rows.append(ar)
        return self.engine.linear_x_wt(act_rows, w2_l)

    def forward_layer(
        self,
        layer_index: int,
        hc_stream: Sequence[Sequence[Sequence[float]]],
        pre_mix: Sequence[Sequence[float]],
        *,
        past_kv: Sequence[Sequence[float]] = (),
    ) -> DeepSeekV41LayerOutput:
        """Execute one complete DeepSeek-V4.1 layer (`mHC` + Sink-MLA + `sqrtsoftplus` `FP8`/`FP4` MoE)."""
        if not (0 <= layer_index < self.arch.num_hidden_layers):
            raise RealizerError("layer_index out of bounds.")
        seq_len = len(hc_stream)
        if seq_len == 0 or len(pre_mix) != seq_len:
            raise RealizerError("Non-empty matching hc_stream and pre_mix required.")
        start_pos = len(past_kv)
        loaded: list[str] = []
        p = f"layers.{layer_index}."
        nh = self.arch.num_attention_heads
        hd = self.arch.head_dim
        rd = self.arch.qk_rope_head_dim
        ng = self.arch.o_groups
        ro = self.arch.o_lora_rank
        eps = self.arch.rms_norm_eps

        # 1. Attention mHC mix + pre-collapse + attn_norm
        attn_pre, attn_post, attn_comb, def_attn = self._hc_mixes(
            hc_stream, p + "hc_attn_fn", p + "hc_attn_scale", p + "hc_attn_base", loaded
        )
        x_attn_in = self._hc_pre(hc_stream, pre_mix)
        attn_norm_w = self.weights.load_float_tensor(p + "attn_norm.weight", loaded, use_numpy=self.engine.uses_numpy)
        x_norm = self.engine.rms_norm_2d(x_attn_in, attn_norm_w, eps)

        # 2. Latent Q (`wq_a -> q_norm -> wq_b`) & Single-Head Latent KV (`wkv -> kv_norm`) + Tail RoPE
        if self.engine.uses_numpy:
            import numpy as np

            xn_nd = self.engine.asarray_f32(x_norm)
            wqa = self.weights.load_float_tensor(p + "attn.wq_a.weight", loaded, use_numpy=True, as_ndarray=True)
            qn_w = self.weights.load_float_tensor(p + "attn.q_norm.weight", loaded, use_numpy=True, as_ndarray=True)
            wqb = self.weights.load_float_tensor(p + "attn.wq_b.weight", loaded, use_numpy=True, as_ndarray=True)
            wkv = self.weights.load_float_tensor(p + "attn.wkv.weight", loaded, use_numpy=True, as_ndarray=True)
            kvn_w = self.weights.load_float_tensor(p + "attn.kv_norm.weight", loaded, use_numpy=True, as_ndarray=True)

            qr_nd = self.engine.rms_norm_nd(self.engine.linear_x_wt_nd(xn_nd, wqa), qn_w, eps)
            q_flat_nd = self.engine.linear_x_wt_nd(qr_nd, wqb).reshape(seq_len, nh, hd)
            kv_nd = self.engine.rms_norm_nd(self.engine.linear_x_wt_nd(xn_nd, wkv), kvn_w, eps)
            q_heads = [[[float(v) for v in hv] for hv in tok] for tok in q_flat_nd.tolist()]
            kv_new = [[float(v) for v in tok] for tok in kv_nd.tolist()]
        else:
            wqa_l = self.weights.load_float_tensor(p + "attn.wq_a.weight", loaded, use_numpy=False)
            qn_l = self.weights.load_float_tensor(p + "attn.q_norm.weight", loaded, use_numpy=False)
            wqb_l = self.weights.load_float_tensor(p + "attn.wq_b.weight", loaded, use_numpy=False)
            wkv_l = self.weights.load_float_tensor(p + "attn.wkv.weight", loaded, use_numpy=False)
            kvn_l = self.weights.load_float_tensor(p + "attn.kv_norm.weight", loaded, use_numpy=False)

            qr_l = self.engine.rms_norm_2d(self.engine.linear_x_wt(x_norm, wqa_l), qn_l, eps)
            qf_l = self.engine.linear_x_wt(qr_l, wqb_l)
            q_heads = [[qf_l[t][h * hd : (h + 1) * hd] for h in range(nh)] for t in range(seq_len)]
            kv_new = self.engine.rms_norm_2d(self.engine.linear_x_wt(x_norm, wkv_l), kvn_l, eps)

        q_rot = apply_interleaved_rope_tail(
            q_heads, start_position=start_pos, rope_dim=rd, rope_theta=self.arch.rope_theta
        )
        kv_rot = apply_interleaved_rope_tail(
            kv_new, start_position=start_pos, rope_dim=rd, rope_theta=self.arch.rope_theta
        )
        kv_full = [*([list(r) for r in past_kv]), *kv_rot]

        # 3. Sliding-Window Attention with Learnable Per-Head `attn_sink` + Inverse Tail RoPE
        attn_sink = self.weights.load_float_tensor(p + "attn.attn_sink", loaded, use_numpy=self.engine.uses_numpy)
        scale = 1.0 / math.sqrt(float(hd))
        win = self.arch.sliding_window
        o_heads_rot: list[list[list[float]]] = []

        if self.engine.uses_numpy:
            import numpy as np

            kv_arr = np.asarray(kv_full, dtype=np.float64)  # [total_seq, hd]
            sink_arr = np.asarray(attn_sink, dtype=np.float64)  # [nh]
            for t_rel in range(seq_len):
                abs_pos = start_pos + t_rel
                win_start = max(0, abs_pos - win + 1)
                kv_win = kv_arr[win_start : abs_pos + 1]  # [W, hd]
                q_t = np.asarray(q_rot[t_rel], dtype=np.float64)  # [nh, hd]
                scores = (q_t @ kv_win.T) * scale  # [nh, W]
                s_max = np.maximum(np.max(scores, axis=-1), -1e30)  # [nh]
                ex = np.exp(scores - s_max[:, None])
                denom = np.sum(ex, axis=-1) + np.exp(sink_arr - s_max)  # [nh]
                ctx = (ex @ kv_win) / denom[:, None]  # [nh, hd]
                o_heads_rot.append([[float(v) for v in hv] for hv in ctx.tolist()])
        else:
            for t_rel in range(seq_len):
                abs_pos = start_pos + t_rel
                win_start = max(0, abs_pos - win + 1)
                t_out: list[list[float]] = []
                for h_i in range(nh):
                    qv = q_rot[t_rel][h_i]
                    sc_list = [
                        math.fsum(qv[d_i] * kv_full[s_i][d_i] for d_i in range(hd)) * scale
                        for s_i in range(win_start, abs_pos + 1)
                    ]
                    s_max = max(max(sc_list), -1e30)
                    ex = [math.exp(v - s_max) for v in sc_list]
                    denom = math.fsum(ex) + math.exp(float(attn_sink[h_i]) - s_max)
                    t_out.append([
                        math.fsum(ex[idx] * kv_full[win_start + idx][d_i] for idx in range(len(ex))) / denom
                        for d_i in range(hd)
                    ])
                o_heads_rot.append(t_out)

        o_unrot = apply_interleaved_rope_tail(
            o_heads_rot, start_position=start_pos, rope_dim=rd, rope_theta=self.arch.rope_theta, inverse=True
        )

        # 4. Block-Diagonal Grouped Output LoRA (`wo_a` [G * r_o, (nh * hd) // G] -> `wo_b` [D, G * r_o])
        group_hd = (nh * hd) // ng
        if self.engine.uses_numpy:
            import numpy as np

            o_nd = np.asarray(o_unrot, dtype=np.float32).reshape(seq_len, ng, group_hd)
            wo_a_nd = self.weights.load_float_tensor(p + "attn.wo_a.weight", loaded, use_numpy=True, as_ndarray=True).reshape(
                ng, ro, group_hd
            )
            wo_b_nd = self.weights.load_float_tensor(p + "attn.wo_b.weight", loaded, use_numpy=True, as_ndarray=True)
            mid_o = np.einsum("sgd,grd->sgr", o_nd.astype(np.float64), wo_a_nd.astype(np.float64)).reshape(
                seq_len, ng * ro
            ).astype(np.float32)
            attn_out = [[float(v) for v in r] for r in self.engine.linear_x_wt_nd(mid_o, wo_b_nd).tolist()]
        else:
            wo_a_l = self.weights.load_float_tensor(p + "attn.wo_a.weight", loaded, use_numpy=False)
            wo_b_l = self.weights.load_float_tensor(p + "attn.wo_b.weight", loaded, use_numpy=False)
            mid_rows: list[list[float]] = []
            for t_rel in range(seq_len):
                flat_o = [v for hv in o_unrot[t_rel] for v in hv]
                row_g: list[float] = []
                for g_i in range(ng):
                    g_slice = flat_o[g_i * group_hd : (g_i + 1) * group_hd]
                    for r_i in range(ro):
                        w_row = wo_a_l[g_i * ro + r_i]
                        row_g.append(math.fsum(a * b for a, b in zip(g_slice, w_row)))
                mid_rows.append(row_g)
            attn_out = self.engine.linear_x_wt(mid_rows, wo_b_l)

        h_mid = self._hc_post(attn_out, hc_stream, attn_post, attn_comb)

        # 5. FFN mHC mix + pre-collapse (using `attn_pre`!) + `ffn_norm` + Sparse `FP4` MoE + Shared Expert
        ffn_pre, ffn_post, ffn_comb, def_ffn = self._hc_mixes(
            h_mid, p + "hc_ffn_fn", p + "hc_ffn_scale", p + "hc_ffn_base", loaded
        )
        x_ffn_in = self._hc_pre(h_mid, attn_pre)
        ffn_norm_w = self.weights.load_float_tensor(p + "ffn_norm.weight", loaded, use_numpy=self.engine.uses_numpy)
        x_ffn = self.engine.rms_norm_2d(x_ffn_in, ffn_norm_w, eps)

        gate_w = self.weights.load_float_tensor(p + "ffn.gate.weight", loaded, use_numpy=self.engine.uses_numpy)
        gate_b = self.weights.load_float_tensor(p + "ffn.gate.bias", loaded, use_numpy=self.engine.uses_numpy)
        router_logits_2d = self.engine.linear_x_wt(x_ffn, gate_w)

        moe_receipts: list[MoERoutingReceipt] = []
        routed_accum = [[0.0] * self.arch.hidden_size for _ in range(seq_len)]
        by_expert: dict[int, list[tuple[int, float]]] = {}
        for t_rel in range(seq_len):
            rec = route_moe_noaux_tc(
                model_fingerprint=self.arch.fingerprint,
                layer_index=layer_index,
                token_position=start_pos + t_rel,
                router_logits=router_logits_2d[t_rel],
                correction_bias=gate_b,
                top_k=self.arch.num_experts_per_tok,
                scoring_func=self.arch.scoring_func,
                norm_topk_prob=self.arch.norm_topk_prob,
                routed_scaling_factor=self.arch.routed_scaling_factor,
                expert_prefix=f"{p}ffn.experts.",
            )
            moe_receipts.append(rec)
            for eid, ew in zip(rec.selected_experts, rec.normalized_weights):
                by_expert.setdefault(eid, []).append((t_rel, ew))

        # Materialize ONLY active experts across the batch (`6 / 384` = `1.56%` per token!)
        for eid in sorted(by_expert.keys()):
            items = by_expert[eid]
            tok_batch = [x_ffn[t_i] for t_i, _ in items]
            exp_outs = self._expert_swiglu(tok_batch, f"{p}ffn.experts.{eid}.", loaded, is_routed_fp4=True)
            for (t_i, ew), e_vec in zip(items, exp_outs):
                for c_i in range(self.arch.hidden_size):
                    routed_accum[t_i][c_i] += ew * e_vec[c_i]

        shared_outs = self._expert_swiglu(x_ffn, f"{p}ffn.shared_experts.", loaded, is_routed_fp4=False)
        ffn_total = [
            [routed_accum[t_i][c_i] + shared_outs[t_i][c_i] for c_i in range(self.arch.hidden_size)]
            for t_i in range(seq_len)
        ]

        h_out = self._hc_post(ffn_total, h_mid, ffn_post, ffn_comb)
        if any(not math.isfinite(v) for tok in h_out for cp in tok for v in cp):
            raise RealizerError("Non-finite activation in DeepSeekV41LayerRealizer.")

        return DeepSeekV41LayerOutput(
            hc_stream=h_out,
            next_pre_mix=ffn_pre,
            kv_cache_full=kv_full,
            moe_receipts=tuple(moe_receipts),
            loaded_tensor_names=tuple(loaded),
            birkhoff_max_row_col_defect=max(def_attn, def_ffn),
        )


class DeepSeekV41ReferenceOracle:
    """Independent pure-Python scalar FP64 oracle for `DeepSeekV41ArchitectureSpec` verification."""

    @classmethod
    def forward_full(
        cls,
        arch: DeepSeekV41ArchitectureSpec,
        weights: LazyFrontierWeightGround,
        token_ids: Sequence[int],
    ) -> list[list[float]]:
        realizer_std = DeepSeekV41LayerRealizer(arch, weights, engine=NumericEngine("stdlib"))
        stream, pre_mix = realizer_std.initial_hc_stream(token_ids)
        for l_idx in range(arch.num_hidden_layers):
            out = realizer_std.forward_layer(l_idx, stream, pre_mix)
            stream, pre_mix = out.hc_stream, out.next_pre_mix
        return realizer_std.collapse_and_logits(stream, pre_mix)


# ============================================================================
# 5. GLM-5.3 (`GlmMoeDsaForCausalLM`) Spec, Latent MLA Ground, Matrix Absorption & Realizer
# ============================================================================


@dataclass(frozen=True, slots=True)
class Glm53ArchitectureSpec:
    """Architectural specification for `GlmMoeDsaForCausalLM` (`GLM-5.3`: `MLA` + Hybrid Dense/Sigmoid-MoE)."""

    source_model: str
    revision: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    first_k_dense_replace: int
    intermediate_size: int
    num_attention_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    rms_norm_eps: float = 1e-5
    rope_theta: float = 8_000_000.0

    def __post_init__(self) -> None:
        if not _REPO_RE.fullmatch(self.source_model) or not _HEX40_RE.fullmatch(self.revision):
            raise ModelArtifactError("Glm53ArchitectureSpec requires valid repo_id and 40-char SHA.")
        if self.qk_rope_head_dim % 2 != 0:
            raise ModelArtifactError("qk_rope_head_dim must be even.")

    @classmethod
    def from_published_profile(cls, name: str = "glm-5.3-bf16") -> Glm53ArchitectureSpec:
        p = PUBLISHED_FRONTIER_PROFILES[name.lower()]
        return cls(
            source_model=p["repo_id"],
            revision=p["revision"],
            vocab_size=p["vocab_size"],
            hidden_size=p["hidden_size"],
            num_hidden_layers=p["num_hidden_layers"],
            first_k_dense_replace=p["first_k_dense_replace"],
            intermediate_size=p["intermediate_size"],
            num_attention_heads=p["num_attention_heads"],
            q_lora_rank=p["q_lora_rank"],
            kv_lora_rank=p["kv_lora_rank"],
            qk_nope_head_dim=p["qk_nope_head_dim"],
            qk_rope_head_dim=p["qk_rope_head_dim"],
            v_head_dim=p["v_head_dim"],
            moe_intermediate_size=p["moe_intermediate_size"],
            n_routed_experts=p["n_routed_experts"],
            n_shared_experts=p["n_shared_experts"],
            num_experts_per_tok=p["num_experts_per_tok"],
            scoring_func=p["scoring_func"],
            routed_scaling_factor=p["routed_scaling_factor"],
            rms_norm_eps=p["rms_norm_eps"],
            rope_theta=p["rope_theta"],
        )

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_kv_width(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def kv_compression_ratio(self) -> float:
        return float(2 * self.num_attention_heads * self.v_head_dim) / float(self.latent_kv_width)

    @property
    def fingerprint(self) -> str:
        return domain_digest(
            "vld.glm53-arch.v1",
            {
                "first_k_dense_replace": self.first_k_dense_replace,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "kv_lora_rank": self.kv_lora_rank,
                "moe_intermediate_size": self.moe_intermediate_size,
                "n_routed_experts": self.n_routed_experts,
                "num_attention_heads": self.num_attention_heads,
                "num_experts_per_tok": self.num_experts_per_tok,
                "num_hidden_layers": self.num_hidden_layers,
                "q_lora_rank": self.q_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "revision": self.revision,
                "source_model": self.source_model,
                "v_head_dim": self.v_head_dim,
                "vocab_size": self.vocab_size,
            },
        )


@dataclass(frozen=True, slots=True)
class LatentMLAKVPage:
    """Content-addressed M1 page storing compressed MLA latents `(c_kv, k_pe)` (`56.89x` smaller than MHA KV)."""

    model_fingerprint: str
    layer_index: int
    seq_len: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    latent_manifest: PagedByteManifest


class LatentMLAKVCacheGround:
    """M1-backed compressed latent KV cache for Multi-Head Latent Attention (`MLA`)."""

    def __init__(self, store: PagedObjectStore) -> None:
        self.store = store

    def store_latent_kv(
        self,
        arch: Glm53ArchitectureSpec,
        *,
        layer_index: int,
        c_kv_2d: Sequence[Sequence[float]],
        k_pe_2d: Sequence[Sequence[float]],
    ) -> LatentMLAKVPage:
        seq_len = len(c_kv_2d)
        if len(k_pe_2d) != seq_len:
            raise RealizerError("c_kv_2d and k_pe_2d sequence lengths must match.")
        flat: list[float] = []
        for c_row, r_row in zip(c_kv_2d, k_pe_2d):
            if len(c_row) != arch.kv_lora_rank or len(r_row) != arch.qk_rope_head_dim:
                raise RealizerError("Latent MLA row width mismatch.")
            flat.extend(float(v) for v in c_row)
            flat.extend(float(v) for v in r_row)
        manifest = self.store.put_bytes(encode_f32_le(flat))
        return LatentMLAKVPage(
            model_fingerprint=arch.fingerprint,
            layer_index=layer_index,
            seq_len=seq_len,
            kv_lora_rank=arch.kv_lora_rank,
            qk_rope_head_dim=arch.qk_rope_head_dim,
            latent_manifest=manifest,
        )

    def load_latent_kv(self, page: LatentMLAKVPage) -> tuple[list[list[float]], list[list[float]]]:
        vals = decode_f32_le(self.store.read_bytes(page.latent_manifest))
        w = page.kv_lora_rank + page.qk_rope_head_dim
        if len(vals) != page.seq_len * w:
            raise RealizerError("Corrupted LatentMLAKVPage float count.")
        c_kv: list[list[float]] = []
        k_pe: list[list[float]] = []
        for t in range(page.seq_len):
            row = vals[t * w : (t + 1) * w]
            c_kv.append(row[: page.kv_lora_rank])
            k_pe.append(row[page.kv_lora_rank :])
        return c_kv, k_pe


@dataclass(frozen=True, slots=True)
class Glm53LayerOutput:
    hidden_states: list[list[float]]
    c_kv_full: list[list[float]]  # [total_seq, kv_lora_rank]
    k_pe_full: list[list[float]]  # [total_seq, qk_rope_head_dim]
    moe_receipts: tuple[MoERoutingReceipt, ...]
    loaded_tensor_names: tuple[str, ...]
    mla_mode: str


class Glm53LayerRealizer:
    """Numerical realizer for `GlmMoeDsaForCausalLM` supporting both `latent_absorbed` and `materialized` MLA."""

    def __init__(
        self,
        arch: Glm53ArchitectureSpec,
        weights: LazyFrontierWeightGround,
        *,
        mla_mode: str = "latent_absorbed",
        engine: NumericEngine | None = None,
    ) -> None:
        if mla_mode not in ("latent_absorbed", "materialized"):
            raise ValueError(f"Unsupported mla_mode: {mla_mode!r}")
        self.arch = arch
        self.weights = weights
        self.mla_mode = mla_mode
        self.engine = engine if engine is not None else NumericEngine("auto")

    def embed_tokens(self, token_ids: Sequence[int]) -> list[list[float]]:
        return self.weights.read_row_slice_f32("model.embed_tokens.weight", token_ids, use_numpy=self.engine.uses_numpy)

    def final_logits(self, hidden_states: Sequence[Sequence[float]]) -> list[list[float]]:
        loaded: list[str] = []
        norm_w = self.weights.load_float_tensor("model.norm.weight", loaded, use_numpy=self.engine.uses_numpy)
        normed = self.engine.rms_norm_2d(hidden_states, norm_w, self.arch.rms_norm_eps)
        if self.engine.uses_numpy:
            lm_nd = self.weights.load_float_tensor("lm_head.weight", loaded, use_numpy=True, as_ndarray=True)
            out_nd = self.engine.linear_x_wt_nd(self.engine.asarray_f32(normed), lm_nd)
            return [[float(v) for v in row] for row in out_nd.tolist()]
        lm_w = self.weights.load_float_tensor("lm_head.weight", loaded, use_numpy=False)
        return self.engine.linear_x_wt(normed, lm_w)

    def _swiglu_mlp(self, x_2d: Sequence[Sequence[float]], prefix: str, loaded: list[str]) -> list[list[float]]:
        if self.engine.uses_numpy:
            x_nd = self.engine.asarray_f32(x_2d)
            gw = self.weights.load_float_tensor(prefix + "gate_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            uw = self.weights.load_float_tensor(prefix + "up_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            dw = self.weights.load_float_tensor(prefix + "down_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            act = self.engine.swiglu_nd(self.engine.linear_x_wt_nd(x_nd, gw), self.engine.linear_x_wt_nd(x_nd, uw))
            return [[float(v) for v in r] for r in self.engine.linear_x_wt_nd(act, dw).tolist()]
        gw_l = self.weights.load_float_tensor(prefix + "gate_proj.weight", loaded, use_numpy=False)
        uw_l = self.weights.load_float_tensor(prefix + "up_proj.weight", loaded, use_numpy=False)
        dw_l = self.weights.load_float_tensor(prefix + "down_proj.weight", loaded, use_numpy=False)
        act_l = self.engine.swiglu_2d(self.engine.linear_x_wt(x_2d, gw_l), self.engine.linear_x_wt(x_2d, uw_l))
        return self.engine.linear_x_wt(act_l, dw_l)

    def forward_layer(
        self,
        layer_index: int,
        hidden_states: Sequence[Sequence[float]],
        *,
        past_c_kv: Sequence[Sequence[float]] = (),
        past_k_pe: Sequence[Sequence[float]] = (),
    ) -> Glm53LayerOutput:
        if not (0 <= layer_index < self.arch.num_hidden_layers):
            raise RealizerError("layer_index out of bounds.")
        if len(past_c_kv) != len(past_k_pe):
            raise RealizerError("past_c_kv and past_k_pe lengths must match.")
        seq_len = len(hidden_states)
        start_pos = len(past_c_kv)
        loaded: list[str] = []
        p = f"model.layers.{layer_index}."
        nh = self.arch.num_attention_heads
        dn = self.arch.qk_nope_head_dim
        dr = self.arch.qk_rope_head_dim
        dv = self.arch.v_head_dim
        rkv = self.arch.kv_lora_rank
        eps = self.arch.rms_norm_eps

        in_norm = self.weights.load_float_tensor(p + "input_layernorm.weight", loaded, use_numpy=self.engine.uses_numpy)
        x_norm = self.engine.rms_norm_2d(hidden_states, in_norm, eps)

        # 1. Query LoRA (`q_a_proj -> q_a_layernorm -> q_b_proj`) + Joint KV Compression (`kv_a_proj_with_mqa`)
        if self.engine.uses_numpy:
            xn_nd = self.engine.asarray_f32(x_norm)
            q_a_w = self.weights.load_float_tensor(p + "self_attn.q_a_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            q_an_w = self.weights.load_float_tensor(p + "self_attn.q_a_layernorm.weight", loaded, use_numpy=True, as_ndarray=True)
            q_b_w = self.weights.load_float_tensor(p + "self_attn.q_b_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            kv_a_w = self.weights.load_float_tensor(
                p + "self_attn.kv_a_proj_with_mqa.weight", loaded, use_numpy=True, as_ndarray=True
            )
            kv_an_w = self.weights.load_float_tensor(
                p + "self_attn.kv_a_layernorm.weight", loaded, use_numpy=True, as_ndarray=True
            )

            qa_nd = self.engine.rms_norm_nd(self.engine.linear_x_wt_nd(xn_nd, q_a_w), q_an_w, eps)
            qb_nd = self.engine.linear_x_wt_nd(qa_nd, q_b_w).reshape(seq_len, nh, dn + dr)
            q_heads_raw = [[[float(v) for v in hv] for hv in tok] for tok in qb_nd.tolist()]

            kva_nd = self.engine.linear_x_wt_nd(xn_nd, kv_a_w)
            ckv_nd = self.engine.rms_norm_nd(kva_nd[:, :rkv], kv_an_w, eps)
            c_kv_new = [[float(v) for v in r] for r in ckv_nd.tolist()]
            k_pe_raw = [[float(v) for v in r] for r in kva_nd[:, rkv:].tolist()]
        else:
            q_a_l = self.weights.load_float_tensor(p + "self_attn.q_a_proj.weight", loaded, use_numpy=False)
            q_an_l = self.weights.load_float_tensor(p + "self_attn.q_a_layernorm.weight", loaded, use_numpy=False)
            q_b_l = self.weights.load_float_tensor(p + "self_attn.q_b_proj.weight", loaded, use_numpy=False)
            kv_a_l = self.weights.load_float_tensor(p + "self_attn.kv_a_proj_with_mqa.weight", loaded, use_numpy=False)
            kv_an_l = self.weights.load_float_tensor(p + "self_attn.kv_a_layernorm.weight", loaded, use_numpy=False)

            qa_l = self.engine.rms_norm_2d(self.engine.linear_x_wt(x_norm, q_a_l), q_an_l, eps)
            qb_l = self.engine.linear_x_wt(qa_l, q_b_l)
            q_heads_raw = [[qb_l[t][h * (dn + dr) : (h + 1) * (dn + dr)] for h in range(nh)] for t in range(seq_len)]
            kva_l = self.engine.linear_x_wt(x_norm, kv_a_l)
            c_kv_new = self.engine.rms_norm_2d([r[:rkv] for r in kva_l], kv_an_l, eps)
            k_pe_raw = [r[rkv:] for r in kva_l]

        q_rot = apply_interleaved_rope_tail(
            q_heads_raw, start_position=start_pos, rope_dim=dr, rope_theta=self.arch.rope_theta
        )
        k_pe_new = apply_interleaved_rope_tail(
            k_pe_raw, start_position=start_pos, rope_dim=dr, rope_theta=self.arch.rope_theta
        )

        c_kv_full = [*([list(r) for r in past_c_kv]), *c_kv_new]
        k_pe_full = [*([list(r) for r in past_k_pe]), *k_pe_new]
        total_seq = len(c_kv_full)
        inv_sqrt_dk = 1.0 / math.sqrt(float(dn + dr))

        # 2. Multi-Head Latent Attention (`latent_absorbed` vs `materialized`)
        if self.engine.uses_numpy:
            import numpy as np

            kv_b_nd = self.weights.load_float_tensor(
                p + "self_attn.kv_b_proj.weight", loaded, use_numpy=True, as_ndarray=True
            ).astype(np.float64)
            # kv_b_proj shape: [nh * (dn + dv), rkv] -> [nh, dn + dv, rkv]
            kv_b_3d = kv_b_nd.reshape(nh, dn + dv, rkv)
            w_k_nope = kv_b_3d[:, :dn, :]  # [nh, dn, rkv]
            w_v = kv_b_3d[:, dn:, :]  # [nh, dv, rkv]

            q64 = np.asarray(q_rot, dtype=np.float64)  # [seq_len, nh, dn + dr]
            q_nope = q64[:, :, :dn]  # [seq_len, nh, dn]
            q_pe = q64[:, :, dn:]  # [seq_len, nh, dr]
            c64 = np.asarray(c_kv_full, dtype=np.float64)  # [total_seq, rkv]
            kpe64 = np.asarray(k_pe_full, dtype=np.float64)  # [total_seq, dr]

            if self.mla_mode == "latent_absorbed":
                # Absorb W_k_nope into query: q_latent = einsum("thd,hdr->thr", q_nope, w_k_nope)
                q_latent = np.einsum("thd,hdr->thr", q_nope, w_k_nope)
                scores_nope = np.einsum("thr,sr->hts", q_latent, c64)
                scores_pe = np.einsum("thd,sd->hts", q_pe, kpe64) if dr > 0 else 0.0
                scores = (scores_nope + scores_pe) * inv_sqrt_dk  # [nh, seq_len, total_seq]
                mask = np.arange(total_seq)[None, :] <= (np.arange(seq_len) + start_pos)[:, None]
                masked = np.where(mask[None, :, :], scores, -np.inf)
                ex = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
                probs = ex / np.sum(ex, axis=-1, keepdims=True)  # [nh, seq_len, total_seq]
                # Aggregate in rkv-dim latent space first, then apply W_v once!
                latent_ctx = np.einsum("hts,sr->thr", probs, c64)  # [seq_len, nh, rkv]
                v_ctx = np.einsum("thr,hvr->thv", latent_ctx, w_v)  # [seq_len, nh, dv]
            else:
                # Materialize full K_nope [total_seq, nh, dn] & V [total_seq, nh, dv]
                k_nope_full = np.einsum("sr,hdr->shd", c64, w_k_nope)
                v_full = np.einsum("sr,hvr->shv", c64, w_v)
                scores_nope = np.einsum("thd,shd->hts", q_nope, k_nope_full)
                scores_pe = np.einsum("thd,sd->hts", q_pe, kpe64) if dr > 0 else 0.0
                scores = (scores_nope + scores_pe) * inv_sqrt_dk
                mask = np.arange(total_seq)[None, :] <= (np.arange(seq_len) + start_pos)[:, None]
                masked = np.where(mask[None, :, :], scores, -np.inf)
                ex = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
                probs = ex / np.sum(ex, axis=-1, keepdims=True)
                v_ctx = np.einsum("hts,shv->thv", probs, v_full)

            o_w_nd = self.weights.load_float_tensor(p + "self_attn.o_proj.weight", loaded, use_numpy=True, as_ndarray=True)
            attn_proj_nd = self.engine.linear_x_wt_nd(v_ctx.reshape(seq_len, nh * dv).astype(np.float32), o_w_nd)
            attn_proj = [[float(v) for v in r] for r in attn_proj_nd.tolist()]
        else:
            kv_b_l = self.weights.load_float_tensor(p + "self_attn.kv_b_proj.weight", loaded, use_numpy=False)
            o_w_l = self.weights.load_float_tensor(p + "self_attn.o_proj.weight", loaded, use_numpy=False)
            v_concat_rows: list[list[float]] = []
            for t_rel in range(seq_len):
                causal_len = start_pos + t_rel + 1
                tok_v: list[float] = []
                for h_i in range(nh):
                    hb_off = h_i * (dn + dv)
                    wk_h = kv_b_l[hb_off : hb_off + dn]
                    wv_h = kv_b_l[hb_off + dn : hb_off + dn + dv]
                    qn_vec = q_rot[t_rel][h_i][:dn]
                    qp_vec = q_rot[t_rel][h_i][dn:]
                    if self.mla_mode == "latent_absorbed":
                        q_lat = [math.fsum(qn_vec[d_i] * wk_h[d_i][r_i] for d_i in range(dn)) for r_i in range(rkv)]
                        sc = [
                            (
                                math.fsum(q_lat[r_i] * c_kv_full[s_i][r_i] for r_i in range(rkv))
                                + math.fsum(qp_vec[p_i] * k_pe_full[s_i][p_i] for p_i in range(dr))
                            )
                            * inv_sqrt_dk
                            for s_i in range(causal_len)
                        ]
                        pr = self.engine.softmax_1d(sc)
                        lat_u = [math.fsum(pr[s_i] * c_kv_full[s_i][r_i] for s_i in range(causal_len)) for r_i in range(rkv)]
                        tok_v.extend(math.fsum(wv_h[v_i][r_i] * lat_u[r_i] for r_i in range(rkv)) for v_i in range(dv))
                    else:
                        sc = []
                        v_mat = []
                        for s_i in range(causal_len):
                            kn_s = [math.fsum(wk_h[d_i][r_i] * c_kv_full[s_i][r_i] for r_i in range(rkv)) for d_i in range(dn)]
                            vv_s = [math.fsum(wv_h[v_i][r_i] * c_kv_full[s_i][r_i] for r_i in range(rkv)) for v_i in range(dv)]
                            v_mat.append(vv_s)
                            sc.append(
                                (
                                    math.fsum(qn_vec[d_i] * kn_s[d_i] for d_i in range(dn))
                                    + math.fsum(qp_vec[p_i] * k_pe_full[s_i][p_i] for p_i in range(dr))
                                )
                                * inv_sqrt_dk
                            )
                        pr = self.engine.softmax_1d(sc)
                        tok_v.extend(math.fsum(pr[s_i] * v_mat[s_i][v_i] for s_i in range(causal_len)) for v_i in range(dv))
                v_concat_rows.append(tok_v)
            attn_proj = self.engine.linear_x_wt(v_concat_rows, o_w_l)

        post_attn = [[float(a) + float(b) for a, b in zip(r0, dr_v)] for r0, dr_v in zip(hidden_states, attn_proj)]
        post_norm = self.weights.load_float_tensor(
            p + "post_attention_layernorm.weight", loaded, use_numpy=self.engine.uses_numpy
        )
        x_ffn = self.engine.rms_norm_2d(post_attn, post_norm, eps)

        # 3. Hybrid Dense Prefix (`layer_index < first_k_dense_replace`) vs Sigmoid `noaux_tc` MoE + Shared Expert
        moe_receipts: list[MoERoutingReceipt] = []
        if layer_index < self.arch.first_k_dense_replace:
            ffn_out = self._swiglu_mlp(x_ffn, p + "mlp.", loaded)
        else:
            gate_w = self.weights.load_float_tensor(p + "mlp.gate.weight", loaded, use_numpy=self.engine.uses_numpy)
            corr_b = self.weights.load_float_tensor(
                p + "mlp.gate.e_score_correction_bias", loaded, use_numpy=self.engine.uses_numpy
            )
            r_logits = self.engine.linear_x_wt(x_ffn, gate_w)
            routed_acc = [[0.0] * self.arch.hidden_size for _ in range(seq_len)]
            by_exp: dict[int, list[tuple[int, float]]] = {}
            for t_rel in range(seq_len):
                rec = route_moe_noaux_tc(
                    model_fingerprint=self.arch.fingerprint,
                    layer_index=layer_index,
                    token_position=start_pos + t_rel,
                    router_logits=r_logits[t_rel],
                    correction_bias=corr_b,
                    top_k=self.arch.num_experts_per_tok,
                    scoring_func=self.arch.scoring_func,
                    norm_topk_prob=self.arch.norm_topk_prob,
                    routed_scaling_factor=self.arch.routed_scaling_factor,
                    expert_prefix=f"{p}mlp.experts.",
                    expert_weight_suffixes=("gate_proj.weight", "up_proj.weight", "down_proj.weight"),
                )
                moe_receipts.append(rec)
                for eid, ew in zip(rec.selected_experts, rec.normalized_weights):
                    by_exp.setdefault(eid, []).append((t_rel, ew))

            for eid in sorted(by_exp.keys()):
                items = by_exp[eid]
                e_outs = self._swiglu_mlp([x_ffn[t_i] for t_i, _ in items], f"{p}mlp.experts.{eid}.", loaded)
                for (t_i, ew), ev in zip(items, e_outs):
                    for c_i in range(self.arch.hidden_size):
                        routed_acc[t_i][c_i] += ew * ev[c_i]

            shared_outs = self._swiglu_mlp(x_ffn, f"{p}mlp.shared_experts.", loaded)
            ffn_out = [
                [routed_acc[t_i][c_i] + shared_outs[t_i][c_i] for c_i in range(self.arch.hidden_size)]
                for t_i in range(seq_len)
            ]

        final_h = [[float(a) + float(b) for a, b in zip(r0, df)] for r0, df in zip(post_attn, ffn_out)]
        if any(not math.isfinite(v) for row in final_h for v in row):
            raise RealizerError("Non-finite activation in Glm53LayerRealizer.")

        return Glm53LayerOutput(
            hidden_states=final_h,
            c_kv_full=c_kv_full,
            k_pe_full=k_pe_full,
            moe_receipts=tuple(moe_receipts),
            loaded_tensor_names=tuple(loaded),
            mla_mode=self.mla_mode,
        )


def verify_mla_matrix_absorption(
    arch: Glm53ArchitectureSpec,
    weights: LazyFrontierWeightGround,
    hidden_states: Sequence[Sequence[float]],
    *,
    layer_index: int = 0,
    past_c_kv: Sequence[Sequence[float]] = (),
    past_k_pe: Sequence[Sequence[float]] = (),
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Verify exact equivalence between Materialized-KV MLA and Matrix-Absorbed Latent-KV MLA."""
    r_mat = Glm53LayerRealizer(arch, weights, mla_mode="materialized")
    r_abs = Glm53LayerRealizer(arch, weights, mla_mode="latent_absorbed")
    out_mat = r_mat.forward_layer(layer_index, hidden_states, past_c_kv=past_c_kv, past_k_pe=past_k_pe)
    out_abs = r_abs.forward_layer(layer_index, hidden_states, past_c_kv=past_c_kv, past_k_pe=past_k_pe)
    diff = NumericEngine.max_abs_diff(out_mat.hidden_states, out_abs.hidden_states)
    if diff > atol:
        raise RealizerError(f"MLA Matrix Absorption discrepancy {diff} > atol {atol}.")
    return {
        "compression_ratio": arch.kv_compression_ratio,
        "latent_floats_per_token": arch.latent_kv_width,
        "max_abs_diff": diff,
        "mha_floats_per_token": 2 * arch.num_attention_heads * arch.v_head_dim,
        "status": "PASS",
    }


__all__ = [
    "DeepSeekV41ArchitectureSpec",
    "DeepSeekV41LayerOutput",
    "DeepSeekV41LayerRealizer",
    "DeepSeekV41ReferenceOracle",
    "Glm53ArchitectureSpec",
    "Glm53LayerOutput",
    "Glm53LayerRealizer",
    "HostedModelGlance",
    "LatentMLAKVCacheGround",
    "LatentMLAKVPage",
    "LazyFrontierWeightGround",
    "PUBLISHED_FRONTIER_PROFILES",
    "glance_huggingface_model",
    "glance_model_structure",
    "route_moe_noaux_tc",
    "verify_mla_matrix_absorption",
]
