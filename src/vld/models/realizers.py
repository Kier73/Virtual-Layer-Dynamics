r"""Phase 6 — Verified Transformer & Sparse MoE Numerical Realizers + Independent FP64 Reference Oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from vld.accel import NumericEngine, decode_bf16_le, decode_f16_le, decode_f32_le, decode_tensor_ndarray, encode_f32_le
from vld.canonical import domain_digest
from vld.memory import PagedObjectStore
from vld.models.artifacts import (
    LazySafetensorsWeightGround,
    ModelArtifactError,
    ModelParameterSpace,
    ModelWeightSnapshot,
    SAFETENSORS_DTYPE_WIDTHS,
    TransformerArchitectureSpec,
)


class RealizerError(ValueError):
    """Raised when tensor shapes, non-finite activations, or reference tolerances fail."""


def decode_shaped_tensor(
    raw: bytes,
    dtype: str,
    shape: tuple[int, ...],
    *,
    use_numpy: bool | None = None,
) -> Any:
    """Decode raw little-endian BF16/F16/F32 bytes into nested finite float lists matching `shape`."""
    if dtype == "BF16":
        flat = decode_bf16_le(raw, use_numpy=use_numpy)
    elif dtype == "F16":
        flat = decode_f16_le(raw, use_numpy=use_numpy)
    elif dtype == "F32":
        flat = decode_f32_le(raw, use_numpy=use_numpy)
    else:
        raise RealizerError(f"Unsupported numerical tensor dtype {dtype!r}.")

    expected = math.prod(shape)
    if len(flat) != expected:
        raise RealizerError(f"Decoded {len(flat)} elements, expected {expected} for shape {shape}.")

    if len(shape) == 1:
        return flat
    if len(shape) == 2:
        rows, cols = shape
        return [flat[r * cols : (r + 1) * cols] for r in range(rows)]
    if len(shape) == 3:
        d0, d1, d2 = shape
        stride0 = d1 * d2
        return [
            [flat[i * stride0 + j * d2 : i * stride0 + (j + 1) * d2] for j in range(d1)]
            for i in range(d0)
        ]
    raise RealizerError(f"Unsupported tensor rank {len(shape)} in decode_shaped_tensor.")


def hidden_state_digest(hidden_2d: Sequence[Sequence[float]]) -> str:
    flat = [float(v) for row in hidden_2d for v in row]
    return hashlib.sha256(encode_f32_le(flat)).hexdigest()


@dataclass(frozen=True, slots=True)
class MoERoutingReceipt:
    """Deterministic top-k sparse MoE routing decision with active tensor addresses."""

    model_fingerprint: str
    layer_index: int
    token_position: int
    top_k: int
    selected_experts: tuple[int, ...]
    normalized_weights: tuple[float, ...]
    active_tensor_names: tuple[str, ...]
    router_logits_digest: str

    @property
    def receipt_digest(self) -> str:
        return domain_digest(
            "vld.moe-routing-receipt.v1",
            {
                "active_tensor_names": list(self.active_tensor_names),
                "layer_index": self.layer_index,
                "model_fingerprint": self.model_fingerprint,
                "normalized_weights": list(self.normalized_weights),
                "router_logits_digest": self.router_logits_digest,
                "selected_experts": list(self.selected_experts),
                "token_position": self.token_position,
                "top_k": self.top_k,
            },
        )


def route_moe_topk(
    arch: TransformerArchitectureSpec,
    *,
    layer_index: int,
    token_position: int,
    router_logits: Sequence[float],
) -> MoERoutingReceipt:
    r"""Select deterministic top-$k$ active experts with tie-breaking by ascending expert index."""
    if not arch.is_moe or arch.num_experts is None or arch.num_experts_per_tok is None:
        raise RealizerError("route_moe_topk requires a sparse MoE TransformerArchitectureSpec.")
    if not (0 <= layer_index < arch.num_hidden_layers):
        raise RealizerError("layer_index out of bounds.")
    if not (0 <= token_position < arch.max_position_embeddings):
        raise RealizerError("token_position exceeds max_position_embeddings.")
    if len(router_logits) != arch.num_experts:
        raise RealizerError(f"Expected {arch.num_experts} router logits, got {len(router_logits)}.")

    vals = [float(x) for x in router_logits]
    if any(not math.isfinite(x) for x in vals):
        raise RealizerError("Router logits must be finite.")

    max_l = max(vals)
    exps = [math.exp(x - max_l) for x in vals]
    denom = math.fsum(exps)
    probs = [e / denom for e in exps]

    chosen = sorted(range(arch.num_experts), key=lambda idx: (-probs[idx], idx))[: arch.num_experts_per_tok]
    if arch.norm_topk_prob:
        sum_top = math.fsum(probs[idx] for idx in chosen)
        weights = [probs[idx] / sum_top for idx in chosen]
    else:
        weights = [probs[idx] for idx in chosen]

    active_tensors: list[str] = []
    for exp_id in chosen:
        base = f"model.layers.{layer_index}.mlp.experts.{exp_id}."
        active_tensors.extend([base + "gate_proj.weight", base + "up_proj.weight", base + "down_proj.weight"])

    return MoERoutingReceipt(
        model_fingerprint=arch.fingerprint,
        layer_index=layer_index,
        token_position=token_position,
        top_k=arch.num_experts_per_tok,
        selected_experts=tuple(chosen),
        normalized_weights=tuple(weights),
        active_tensor_names=tuple(active_tensors),
        router_logits_digest=domain_digest("vld.router-logits.v1", vals),
    )


def _apply_rope_3d(
    heads_3d: list[list[list[float]]],  # [seq_len, n_heads, head_dim]
    *,
    start_position: int,
    rope_theta: float,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> list[list[list[float]]]:
    seq_len = len(heads_3d)
    if seq_len == 0:
        return []
    head_dim = len(heads_3d[0][0])
    rdim = head_dim if rotary_dim is None else int(rotary_dim)
    half = rdim // 2
    inv_freq = [rope_theta ** (-2.0 * i / rdim) for i in range(half)]
    out: list[list[list[float]]] = []
    for t in range(seq_len):
        pos = start_position + t
        cos_half = [math.cos(pos * f) for f in inv_freq]
        sin_half = [math.sin(pos * f) for f in inv_freq]
        token_heads: list[list[float]] = []
        for h_vec in heads_3d[t]:
            tail = h_vec[rdim:] if rdim < head_dim else []
            if interleaved:
                rot_part: list[float] = []
                for i in range(half):
                    x0 = h_vec[2 * i]
                    x1 = h_vec[2 * i + 1]
                    c, s = cos_half[i], sin_half[i]
                    rot_part.append(x0 * c - x1 * s)
                    rot_part.append(x1 * c + x0 * s)
                token_heads.append(rot_part + tail)
            else:
                first = h_vec[:half]
                second = h_vec[half:rdim]
                rot_first = [x1 * c - x2 * s for x1, x2, c, s in zip(first, second, cos_half, sin_half)]
                rot_second = [x2 * c + x1 * s for x1, x2, c, s in zip(first, second, cos_half, sin_half)]
                token_heads.append(rot_first + rot_second + tail)
        out.append(token_heads)
    return out


def _apply_rope_3d_nd(
    heads_nd: Any,  # [seq_len, n_heads, head_dim]
    *,
    start_position: int,
    rope_theta: float,
    rotary_dim: int,
    interleaved: bool,
    engine: NumericEngine,
) -> Any:
    import numpy as np

    head_dim = int(heads_nd.shape[2])
    if not interleaved and rotary_dim == head_dim:
        return engine.rope_3d_nd(heads_nd, start_position=start_position, rope_theta=rope_theta)
    seq_len = int(heads_nd.shape[0])
    half = rotary_dim // 2
    inv_freq = np.power(float(rope_theta), -2.0 * np.arange(half, dtype=np.float64) / float(rotary_dim))
    positions = np.arange(start_position, start_position + seq_len, dtype=np.float64)[:, None]
    angles = positions * inv_freq[None, :]
    cos_a = np.cos(angles)[:, None, :]  # [seq_len, 1, half]
    sin_a = np.sin(angles)[:, None, :]
    rot_slice = heads_nd[:, :, :rotary_dim].astype(np.float64)
    if interleaved:
        x0 = rot_slice[:, :, 0::2]
        x1 = rot_slice[:, :, 1::2]
        y0 = x0 * cos_a - x1 * sin_a
        y1 = x1 * cos_a + x0 * sin_a
        rot_out = np.empty_like(rot_slice)
        rot_out[:, :, 0::2] = y0
        rot_out[:, :, 1::2] = y1
    else:
        x0 = rot_slice[:, :, :half]
        x1 = rot_slice[:, :, half:]
        rot_out = np.concatenate((x0 * cos_a - x1 * sin_a, x1 * cos_a + x0 * sin_a), axis=-1)
    if rotary_dim < head_dim:
        full = np.concatenate((rot_out.astype(np.float32), heads_nd[:, :, rotary_dim:].astype(np.float32)), axis=-1)
        return np.ascontiguousarray(full, dtype=np.float32)
    return np.ascontiguousarray(rot_out, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class LayerExecutionOutput:
    """Verified output of a single Transformer/MoE layer step including updated KV cache and telemetry."""

    hidden_states: list[list[float]]
    k_cache_full: list[list[list[float]]]  # [total_seq_len, n_kv_heads, head_dim]
    v_cache_full: list[list[list[float]]]  # [total_seq_len, n_kv_heads, head_dim]
    moe_receipts: tuple[MoERoutingReceipt, ...]
    loaded_tensor_names: tuple[str, ...]


class TransformerLayerRealizer:
    """Lazy M1/Vantage-backed numerical realizer for Qwen3 Dense and Sparse MoE decoder layers."""

    def __init__(
        self,
        arch: TransformerArchitectureSpec,
        snapshot: ModelWeightSnapshot | LazySafetensorsWeightGround,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
        engine: NumericEngine | None = None,
        cache_decoded_weights: bool = False,
    ) -> None:
        self.arch = arch
        self.snapshot = snapshot
        self.store = store
        self.remote_range_fetcher = remote_range_fetcher
        self.engine = engine if engine is not None else NumericEngine("auto")
        self.cache_decoded_weights = bool(cache_decoded_weights)
        self._decoded_nd_cache: dict[str, Any] = {}
        self._decoded_py_cache: dict[str, Any] = {}

    def _load_tensor(self, name: str, loaded_log: list[str]) -> Any:
        if self.cache_decoded_weights and name in self._decoded_py_cache:
            loaded_log.append(name)
            return self._decoded_py_cache[name]
        t_space = self.snapshot.space.tensor(name)
        raw = self.snapshot.read_tensor_bytes(
            name,
            store=self.store,
            remote_range_fetcher=self.remote_range_fetcher,
        )
        loaded_log.append(name)
        dec = decode_shaped_tensor(raw, t_space.dtype, t_space.shape, use_numpy=self.engine.uses_numpy)
        if self.cache_decoded_weights:
            self._decoded_py_cache[name] = dec
        return dec

    def _load_tensor_nd(self, name: str, loaded_log: list[str]) -> Any:
        if self.cache_decoded_weights and name in self._decoded_nd_cache:
            loaded_log.append(name)
            return self._decoded_nd_cache[name]
        t_space = self.snapshot.space.tensor(name)
        raw = self.snapshot.read_tensor_bytes(
            name,
            store=self.store,
            remote_range_fetcher=self.remote_range_fetcher,
        )
        loaded_log.append(name)
        try:
            arr = decode_tensor_ndarray(raw, t_space.dtype, t_space.shape)
        except ValueError as exc:
            raise RealizerError(str(exc)) from exc
        if self.cache_decoded_weights:
            arr.setflags(write=False)
            self._decoded_nd_cache[name] = arr
        return arr

    def embed_tokens(self, token_ids: Sequence[int]) -> list[list[float]]:
        for tid in token_ids:
            if not (0 <= tid < self.arch.vocab_size):
                raise RealizerError(f"Token ID {tid} outside vocab_size {self.arch.vocab_size}.")
        if not token_ids:
            return []
        unique_ids = sorted(set(int(t) for t in token_ids))
        # Row-slice locality: when touching a sparse subset of a large vocabulary (e.g., 10 of 151,936 tokens),
        # read ONLY the exact row byte ranges [tid * row_bytes, (tid + 1) * row_bytes) instead of 300MB..1.2GB!
        if hasattr(self.snapshot, "read_tensor_slice_bytes") and len(unique_ids) * 8 < self.arch.vocab_size:
            t_space = self.snapshot.space.tensor("model.embed_tokens.weight")
            row_bytes = self.arch.hidden_size * SAFETENSORS_DTYPE_WIDTHS[t_space.dtype]
            row_cache: dict[int, list[float]] = {}
            for uid in unique_ids:
                raw_row = self.snapshot.read_tensor_slice_bytes(
                    "model.embed_tokens.weight",
                    uid * row_bytes,
                    row_bytes,
                    store=self.store,
                    remote_range_fetcher=self.remote_range_fetcher,
                )
                row_cache[uid] = decode_shaped_tensor(
                    raw_row,
                    t_space.dtype,
                    (self.arch.hidden_size,),
                    use_numpy=self.engine.uses_numpy,
                )
            return [list(row_cache[int(tid)]) for tid in token_ids]

        loaded: list[str] = []
        if self.engine.uses_numpy:
            emb_nd = self._load_tensor_nd("model.embed_tokens.weight", loaded)
            return [[float(v) for v in row] for row in emb_nd[list(token_ids)].tolist()]
        emb_table = self._load_tensor("model.embed_tokens.weight", loaded)
        return [list(emb_table[tid]) for tid in token_ids]

    def final_logits(self, hidden_states: Sequence[Sequence[float]]) -> list[list[float]]:
        if not hidden_states or any(len(row) != self.arch.hidden_size for row in hidden_states):
            raise RealizerError("Invalid hidden_states shape in final_logits.")
        loaded: list[str] = []
        head_name = "model.embed_tokens.weight" if self.arch.tie_word_embeddings else "lm_head.weight"
        if self.engine.uses_numpy:
            import numpy as np

            x_nd = self.engine.asarray_f32(hidden_states)
            if not bool(np.isfinite(x_nd).all()):
                raise RealizerError("Non-finite hidden_states supplied to final_logits.")
            norm_w_nd = self._load_tensor_nd("model.norm.weight", loaded)
            normed_nd = self.engine.rms_norm_nd(x_nd, norm_w_nd, self.arch.rms_norm_eps)
            lm_w_nd = self._load_tensor_nd(head_name, loaded)
            logits_nd = self.engine.linear_x_wt_nd(normed_nd, lm_w_nd)
            if not bool(np.isfinite(logits_nd).all()):
                raise RealizerError("Non-finite logits produced in final_logits.")
            return [[float(v) for v in row] for row in logits_nd.tolist()]
        if any(not math.isfinite(float(v)) for row in hidden_states for v in row):
            raise RealizerError("Non-finite hidden_states supplied to final_logits.")
        norm_w = self._load_tensor("model.norm.weight", loaded)
        normed = self.engine.rms_norm_2d(hidden_states, norm_w, self.arch.rms_norm_eps)
        lm_w = self._load_tensor(head_name, loaded)
        out = self.engine.linear_x_wt(normed, lm_w)
        if any(not math.isfinite(float(v)) for row in out for v in row):
            raise RealizerError("Non-finite logits produced in final_logits.")
        return out

    def _forward_layer_numpy(
        self,
        layer_index: int,
        hidden_states: Sequence[Sequence[float]],
        *,
        past_k: Sequence[Sequence[Sequence[float]]],
        past_v: Sequence[Sequence[Sequence[float]]],
    ) -> LayerExecutionOutput:
        """Contiguous `ndarray`-resident layer realizer avoiding intermediate Python list marshaling."""
        import numpy as np

        seq_len = len(hidden_states)
        start_pos = len(past_k)
        loaded_log: list[str] = []
        p = f"model.layers.{layer_index}."
        nq = self.arch.num_attention_heads
        nkv = self.arch.num_key_value_heads
        hd = self.arch.head_dim
        eps = self.arch.rms_norm_eps

        x_nd = self.engine.asarray_f32(hidden_states)  # [seq_len, hidden_size]
        in_norm_w = self._load_tensor_nd(p + "input_layernorm.weight", loaded_log)
        normed_x = self.engine.rms_norm_nd(x_nd, in_norm_w, eps)

        q_flat = self.engine.linear_x_wt_nd(normed_x, self._load_tensor_nd(p + "self_attn.q_proj.weight", loaded_log))
        k_flat = self.engine.linear_x_wt_nd(normed_x, self._load_tensor_nd(p + "self_attn.k_proj.weight", loaded_log))
        v_flat = self.engine.linear_x_wt_nd(normed_x, self._load_tensor_nd(p + "self_attn.v_proj.weight", loaded_log))
        if self.arch.has_qkv_bias:
            q_flat = np.ascontiguousarray(q_flat + self._load_tensor_nd(p + "self_attn.q_proj.bias", loaded_log)[None, :], dtype=np.float32)
            k_flat = np.ascontiguousarray(k_flat + self._load_tensor_nd(p + "self_attn.k_proj.bias", loaded_log)[None, :], dtype=np.float32)
            v_flat = np.ascontiguousarray(v_flat + self._load_tensor_nd(p + "self_attn.v_proj.bias", loaded_log)[None, :], dtype=np.float32)

        if self.arch.has_qk_norm:
            q_norm_w = self._load_tensor_nd(p + "self_attn.q_norm.weight", loaded_log)
            k_norm_w = self._load_tensor_nd(p + "self_attn.k_norm.weight", loaded_log)
            q_heads = self.engine.rms_norm_nd(q_flat.reshape(seq_len, nq, hd), q_norm_w, eps)
            k_heads = self.engine.rms_norm_nd(k_flat.reshape(seq_len, nkv, hd), k_norm_w, eps)
        else:
            q_heads = np.ascontiguousarray(q_flat.reshape(seq_len, nq, hd), dtype=np.float32)
            k_heads = np.ascontiguousarray(k_flat.reshape(seq_len, nkv, hd), dtype=np.float32)
        v_heads = np.ascontiguousarray(v_flat.reshape(seq_len, nkv, hd), dtype=np.float32)

        q_rot = _apply_rope_3d_nd(
            q_heads,
            start_position=start_pos,
            rope_theta=self.arch.rope_theta,
            rotary_dim=self.arch.rotary_dim,
            interleaved=self.arch.has_interleaved_rope,
            engine=self.engine,
        )
        k_rot = _apply_rope_3d_nd(
            k_heads,
            start_position=start_pos,
            rope_theta=self.arch.rope_theta,
            rotary_dim=self.arch.rotary_dim,
            interleaved=self.arch.has_interleaved_rope,
            engine=self.engine,
        )

        if start_pos > 0:
            try:
                past_k_nd = self.engine.asarray_f32(past_k)
                past_v_nd = self.engine.asarray_f32(past_v)
            except ValueError as exc:
                raise RealizerError(f"Malformed past KV cache: {exc}") from exc
            if past_k_nd.shape != (start_pos, nkv, hd) or past_v_nd.shape != (start_pos, nkv, hd):
                raise RealizerError("past_k / past_v head dimensions disagree with architecture.")
            if not bool(np.isfinite(past_k_nd).all()) or not bool(np.isfinite(past_v_nd).all()):
                raise RealizerError("Non-finite values in past KV cache.")
            k_full_nd = np.ascontiguousarray(np.concatenate((past_k_nd, k_rot), axis=0), dtype=np.float32)
            v_full_nd = np.ascontiguousarray(np.concatenate((past_v_nd, v_heads), axis=0), dtype=np.float32)
        else:
            k_full_nd = k_rot
            v_full_nd = v_heads

        attn_concat = self.engine.causal_gqa_nd(
            q_rot,
            k_full_nd,
            v_full_nd,
            start_position=start_pos,
            sliding_window=self.arch.sliding_window_for_layer(layer_index),
        )
        o_proj_w = self._load_tensor_nd(p + "self_attn.o_proj.weight", loaded_log)
        attn_proj = self.engine.linear_x_wt_nd(attn_concat, o_proj_w)
        if self.arch.has_sandwich_norms:
            attn_proj = self.engine.rms_norm_nd(
                attn_proj,
                self._load_tensor_nd(p + "post_self_attn_layernorm.weight", loaded_log),
                eps,
            )
        post_attn = np.ascontiguousarray(x_nd + attn_proj, dtype=np.float32)

        post_norm_w = self._load_tensor_nd(p + "post_attention_layernorm.weight", loaded_log)
        normed_ffn = self.engine.rms_norm_nd(post_attn, post_norm_w, eps)

        moe_receipts: list[MoERoutingReceipt] = []
        if not self.arch.is_moe:
            if self.arch.has_fused_gate_up:
                f_sz = int(self.arch.intermediate_size or 0)
                gu = self.engine.linear_x_wt_nd(normed_ffn, self._load_tensor_nd(p + "mlp.gate_up_proj.weight", loaded_log))
                gate = np.ascontiguousarray(gu[:, :f_sz], dtype=np.float32)
                up = np.ascontiguousarray(gu[:, f_sz:], dtype=np.float32)
            else:
                gate = self.engine.linear_x_wt_nd(normed_ffn, self._load_tensor_nd(p + "mlp.gate_proj.weight", loaded_log))
                up = self.engine.linear_x_wt_nd(normed_ffn, self._load_tensor_nd(p + "mlp.up_proj.weight", loaded_log))
            act = self.engine.swiglu_nd(gate, up)
            ffn_out = self.engine.linear_x_wt_nd(act, self._load_tensor_nd(p + "mlp.down_proj.weight", loaded_log))
        else:
            router_w = self._load_tensor_nd(p + "mlp.gate.weight", loaded_log)
            router_logits_nd = self.engine.linear_x_wt_nd(normed_ffn, router_w)
            router_logits_2d = router_logits_nd.tolist()
            expert_cache: dict[int, tuple[Any, Any, Any]] = {}
            ffn_out = np.zeros((seq_len, self.arch.hidden_size), dtype=np.float64)
            for t_rel in range(seq_len):
                rec = route_moe_topk(
                    self.arch,
                    layer_index=layer_index,
                    token_position=start_pos + t_rel,
                    router_logits=router_logits_2d[t_rel],
                )
                moe_receipts.append(rec)
                tok_nd = normed_ffn[t_rel : t_rel + 1]
                for exp_id, exp_w in zip(rec.selected_experts, rec.normalized_weights):
                    if exp_id not in expert_cache:
                        eb = f"{p}mlp.experts.{exp_id}."
                        expert_cache[exp_id] = (
                            self._load_tensor_nd(eb + "gate_proj.weight", loaded_log),
                            self._load_tensor_nd(eb + "up_proj.weight", loaded_log),
                            self._load_tensor_nd(eb + "down_proj.weight", loaded_log),
                        )
                    gw, uw, dw = expert_cache[exp_id]
                    g_val = self.engine.linear_x_wt_nd(tok_nd, gw)
                    u_val = self.engine.linear_x_wt_nd(tok_nd, uw)
                    swi = self.engine.swiglu_nd(g_val, u_val)
                    e_out = self.engine.linear_x_wt_nd(swi, dw)[0]
                    ffn_out[t_rel] += float(exp_w) * e_out.astype(np.float64)
            ffn_out = ffn_out.astype(np.float32)

        if self.arch.has_sandwich_norms:
            ffn_out = self.engine.rms_norm_nd(
                ffn_out,
                self._load_tensor_nd(p + "post_mlp_layernorm.weight", loaded_log),
                eps,
            )

        final_nd = np.ascontiguousarray(post_attn.astype(np.float64) + ffn_out.astype(np.float64), dtype=np.float64)
        if not bool(np.isfinite(final_nd).all()):
            raise RealizerError("Non-finite activation produced during layer realization.")

        return LayerExecutionOutput(
            hidden_states=[[float(v) for v in row] for row in final_nd.tolist()],
            k_cache_full=[[[float(v) for v in hv] for hv in tok] for tok in k_full_nd.tolist()],
            v_cache_full=[[[float(v) for v in hv] for hv in tok] for tok in v_full_nd.tolist()],
            moe_receipts=tuple(moe_receipts),
            loaded_tensor_names=tuple(loaded_log),
        )

    def forward_layer(
        self,
        layer_index: int,
        hidden_states: Sequence[Sequence[float]],
        *,
        past_k: Sequence[Sequence[Sequence[float]]] = (),
        past_v: Sequence[Sequence[Sequence[float]]] = (),
    ) -> LayerExecutionOutput:
        if not (0 <= layer_index < self.arch.num_hidden_layers):
            raise RealizerError("layer_index out of bounds.")
        seq_len = len(hidden_states)
        if seq_len == 0 or any(len(row) != self.arch.hidden_size for row in hidden_states):
            raise RealizerError("Invalid hidden_states shape.")
        if len(past_k) != len(past_v):
            raise RealizerError("past_k and past_v sequence lengths must match.")
        start_pos = len(past_k)
        if start_pos + seq_len > self.arch.max_position_embeddings:
            raise RealizerError("Sequence position exceeds max_position_embeddings.")

        if self.engine.uses_numpy:
            return self._forward_layer_numpy(
                layer_index,
                hidden_states,
                past_k=past_k,
                past_v=past_v,
            )

        loaded_log: list[str] = []
        p = f"model.layers.{layer_index}."
        nq = self.arch.num_attention_heads
        nkv = self.arch.num_key_value_heads
        hd = self.arch.head_dim
        eps = self.arch.rms_norm_eps

        # 1. Pre-attention RMSNorm + QKV projections + QK per-head RMSNorm + RoPE
        in_norm_w = self._load_tensor(p + "input_layernorm.weight", loaded_log)
        normed_x = self.engine.rms_norm_2d(hidden_states, in_norm_w, eps)

        q_flat = self.engine.linear_x_wt(normed_x, self._load_tensor(p + "self_attn.q_proj.weight", loaded_log))
        k_flat = self.engine.linear_x_wt(normed_x, self._load_tensor(p + "self_attn.k_proj.weight", loaded_log))
        v_flat = self.engine.linear_x_wt(normed_x, self._load_tensor(p + "self_attn.v_proj.weight", loaded_log))
        if self.arch.has_qkv_bias:
            qb = self._load_tensor(p + "self_attn.q_proj.bias", loaded_log)
            kb = self._load_tensor(p + "self_attn.k_proj.bias", loaded_log)
            vb = self._load_tensor(p + "self_attn.v_proj.bias", loaded_log)
            q_flat = [[float(x) + float(b) for x, b in zip(row, qb)] for row in q_flat]
            k_flat = [[float(x) + float(b) for x, b in zip(row, kb)] for row in k_flat]
            v_flat = [[float(x) + float(b) for x, b in zip(row, vb)] for row in v_flat]

        if self.arch.has_qk_norm:
            q_norm_w = self._load_tensor(p + "self_attn.q_norm.weight", loaded_log)
            k_norm_w = self._load_tensor(p + "self_attn.k_norm.weight", loaded_log)
            q_heads = [
                self.engine.rms_norm_2d([q_flat[t][h * hd : (h + 1) * hd] for h in range(nq)], q_norm_w, eps)
                for t in range(seq_len)
            ]
            k_heads = [
                self.engine.rms_norm_2d([k_flat[t][h * hd : (h + 1) * hd] for h in range(nkv)], k_norm_w, eps)
                for t in range(seq_len)
            ]
        else:
            q_heads = [
                [[float(x) for x in q_flat[t][h * hd : (h + 1) * hd]] for h in range(nq)]
                for t in range(seq_len)
            ]
            k_heads = [
                [[float(x) for x in k_flat[t][h * hd : (h + 1) * hd]] for h in range(nkv)]
                for t in range(seq_len)
            ]
        v_heads = [
            [[float(x) for x in v_flat[t][h * hd : (h + 1) * hd]] for h in range(nkv)]
            for t in range(seq_len)
        ]

        q_rot = _apply_rope_3d(
            q_heads,
            start_position=start_pos,
            rope_theta=self.arch.rope_theta,
            rotary_dim=self.arch.rotary_dim,
            interleaved=self.arch.has_interleaved_rope,
        )
        k_rot = _apply_rope_3d(
            k_heads,
            start_position=start_pos,
            rope_theta=self.arch.rope_theta,
            rotary_dim=self.arch.rotary_dim,
            interleaved=self.arch.has_interleaved_rope,
        )

        k_full = [
            *([[list(hv) for hv in tok] for tok in past_k]),
            *k_rot,
        ]
        v_full = [
            *([[list(hv) for hv in tok] for tok in past_v]),
            *v_heads,
        ]

        # 2. Causal Grouped-Query Attention (GQA) with optional sliding window
        group_size = nq // nkv
        inv_sqrt_d = 1.0 / math.sqrt(hd)
        sw = self.arch.sliding_window_for_layer(layer_index)
        attn_concat: list[list[float]] = []
        for t_rel in range(seq_len):
            causal_len = start_pos + t_rel + 1
            win_start = max(0, causal_len - sw) if sw is not None else 0
            tok_out_flat: list[float] = []
            for qh in range(nq):
                kvh = qh // group_size
                q_vec = q_rot[t_rel][qh]
                scores = [
                    math.fsum(q_vec[d_i] * k_full[s_idx][kvh][d_i] for d_i in range(hd)) * inv_sqrt_d
                    for s_idx in range(win_start, causal_len)
                ]
                probs = self.engine.softmax_1d(scores)
                ctx = [
                    math.fsum(probs[p_i] * v_full[win_start + p_i][kvh][d_i] for p_i in range(len(probs)))
                    for d_i in range(hd)
                ]
                tok_out_flat.extend(ctx)
            attn_concat.append(tok_out_flat)

        o_proj_w = self._load_tensor(p + "self_attn.o_proj.weight", loaded_log)
        attn_proj = self.engine.linear_x_wt(attn_concat, o_proj_w)
        if self.arch.has_sandwich_norms:
            attn_proj = self.engine.rms_norm_2d(
                attn_proj,
                self._load_tensor(p + "post_self_attn_layernorm.weight", loaded_log),
                eps,
            )
        post_attn = [
            [float(x0) + float(dx) for x0, dx in zip(r0, dr)]
            for r0, dr in zip(hidden_states, attn_proj)
        ]

        # 3. Post-attention RMSNorm + Dense SwiGLU OR Active-Expert-Only Sparse MoE FFN
        post_norm_w = self._load_tensor(p + "post_attention_layernorm.weight", loaded_log)
        normed_ffn = self.engine.rms_norm_2d(post_attn, post_norm_w, eps)

        moe_receipts: list[MoERoutingReceipt] = []
        if not self.arch.is_moe:
            if self.arch.has_fused_gate_up:
                f_sz = int(self.arch.intermediate_size or 0)
                gu = self.engine.linear_x_wt(normed_ffn, self._load_tensor(p + "mlp.gate_up_proj.weight", loaded_log))
                gate = [row[:f_sz] for row in gu]
                up = [row[f_sz:] for row in gu]
            else:
                gate = self.engine.linear_x_wt(normed_ffn, self._load_tensor(p + "mlp.gate_proj.weight", loaded_log))
                up = self.engine.linear_x_wt(normed_ffn, self._load_tensor(p + "mlp.up_proj.weight", loaded_log))
            act = self.engine.swiglu_2d(gate, up)
            ffn_out = self.engine.linear_x_wt(act, self._load_tensor(p + "mlp.down_proj.weight", loaded_log))
        else:
            router_w = self._load_tensor(p + "mlp.gate.weight", loaded_log)
            router_logits_2d = self.engine.linear_x_wt(normed_ffn, router_w)
            # Load each active expert's weights AT MOST ONCE across the token batch; NEVER touch inactive experts!
            expert_cache: dict[int, tuple[Any, Any, Any]] = {}
            ffn_out = []
            for t_rel in range(seq_len):
                rec = route_moe_topk(
                    self.arch,
                    layer_index=layer_index,
                    token_position=start_pos + t_rel,
                    router_logits=router_logits_2d[t_rel],
                )
                moe_receipts.append(rec)
                tok_vec = [normed_ffn[t_rel]]
                accum = [0.0] * self.arch.hidden_size
                for exp_id, exp_w in zip(rec.selected_experts, rec.normalized_weights):
                    if exp_id not in expert_cache:
                        eb = f"{p}mlp.experts.{exp_id}."
                        expert_cache[exp_id] = (
                            self._load_tensor(eb + "gate_proj.weight", loaded_log),
                            self._load_tensor(eb + "up_proj.weight", loaded_log),
                            self._load_tensor(eb + "down_proj.weight", loaded_log),
                        )
                    gw, uw, dw = expert_cache[exp_id]
                    g_val = self.engine.linear_x_wt(tok_vec, gw)
                    u_val = self.engine.linear_x_wt(tok_vec, uw)
                    swi = self.engine.swiglu_2d(g_val, u_val)
                    e_out = self.engine.linear_x_wt(swi, dw)[0]
                    for h_i in range(self.arch.hidden_size):
                        accum[h_i] += exp_w * e_out[h_i]
                ffn_out.append(accum)

        if self.arch.has_sandwich_norms:
            ffn_out = self.engine.rms_norm_2d(
                ffn_out,
                self._load_tensor(p + "post_mlp_layernorm.weight", loaded_log),
                eps,
            )

        final_hidden = [
            [float(r0[i]) + float(df[i]) for i in range(self.arch.hidden_size)]
            for r0, df in zip(post_attn, ffn_out)
        ]
        if any(not math.isfinite(v) for row in final_hidden for v in row):
            raise RealizerError("Non-finite activation produced during layer realization.")

        return LayerExecutionOutput(
            hidden_states=final_hidden,
            k_cache_full=k_full,
            v_cache_full=v_full,
            moe_receipts=tuple(moe_receipts),
            loaded_tensor_names=tuple(loaded_log),
        )


# ============================================================================
# Independent Pure-Python FP64 Reference Oracle (Shares ZERO NumericEngine code)
# ============================================================================


class IndependentReferenceOracle:
    """Standalone FP64 scalar reference evaluator for independent numerical verification."""

    @staticmethod
    def _rms(vec: Sequence[float], weight: Sequence[float], eps: float) -> list[float]:
        ms = sum(float(v) * float(v) for v in vec) / len(vec)
        inv = 1.0 / math.sqrt(ms + float(eps))
        return [float(v) * inv * float(w) for v, w in zip(vec, weight)]

    @staticmethod
    def _matvec(mat_out_in: Sequence[Sequence[float]], vec: Sequence[float]) -> list[float]:
        return [sum(float(m) * float(v) for m, v in zip(row, vec)) for row in mat_out_in]

    @classmethod
    def forward_full_sequence(
        cls,
        arch: TransformerArchitectureSpec,
        raw_tensors: Mapping[str, bytes],
        token_ids: Sequence[int],
    ) -> list[list[float]]:
        """Evaluate full multi-layer causal forward logits from scratch in scalar FP64."""
        space = arch.build_parameter_space()

        def W(name: str) -> Any:
            t = space.tensor(name)
            return decode_shaped_tensor(raw_tensors[name], t.dtype, t.shape, use_numpy=False)

        emb = W("model.embed_tokens.weight")
        hidden = [[float(v) for v in emb[tid]] for tid in token_ids]
        seq_len = len(token_ids)
        nq = arch.num_attention_heads
        nkv = arch.num_key_value_heads
        hd = arch.head_dim
        rdim = arch.rotary_dim
        half = rdim // 2
        eps = arch.rms_norm_eps

        for l_idx in range(arch.num_hidden_layers):
            p = f"model.layers.{l_idx}."
            w_in = W(p + "input_layernorm.weight")
            wq = W(p + "self_attn.q_proj.weight")
            wk = W(p + "self_attn.k_proj.weight")
            wv = W(p + "self_attn.v_proj.weight")
            wo = W(p + "self_attn.o_proj.weight")
            wqn = W(p + "self_attn.q_norm.weight") if arch.has_qk_norm else None
            wkn = W(p + "self_attn.k_norm.weight") if arch.has_qk_norm else None
            bq = W(p + "self_attn.q_proj.bias") if arch.has_qkv_bias else None
            bk = W(p + "self_attn.k_proj.bias") if arch.has_qkv_bias else None
            bv = W(p + "self_attn.v_proj.bias") if arch.has_qkv_bias else None
            w_post_sa = W(p + "post_self_attn_layernorm.weight") if arch.has_sandwich_norms else None
            w_post_mlp = W(p + "post_mlp_layernorm.weight") if arch.has_sandwich_norms else None

            q_seq: list[list[list[float]]] = []
            k_seq: list[list[list[float]]] = []
            v_seq: list[list[list[float]]] = []
            for pos in range(seq_len):
                nx = cls._rms(hidden[pos], w_in, eps)
                qf = cls._matvec(wq, nx)
                kf = cls._matvec(wk, nx)
                vf = cls._matvec(wv, nx)
                if bq is not None and bk is not None and bv is not None:
                    qf = [x + float(b) for x, b in zip(qf, bq)]
                    kf = [x + float(b) for x, b in zip(kf, bk)]
                    vf = [x + float(b) for x, b in zip(vf, bv)]
                if wqn is not None and wkn is not None:
                    qh_list = [cls._rms(qf[h * hd : (h + 1) * hd], wqn, eps) for h in range(nq)]
                    kh_list = [cls._rms(kf[h * hd : (h + 1) * hd], wkn, eps) for h in range(nkv)]
                else:
                    qh_list = [[float(x) for x in qf[h * hd : (h + 1) * hd]] for h in range(nq)]
                    kh_list = [[float(x) for x in kf[h * hd : (h + 1) * hd]] for h in range(nkv)]
                vh_list = [vf[h * hd : (h + 1) * hd] for h in range(nkv)]

                def rot_tok(h_vecs: list[list[float]], position: int) -> list[list[float]]:
                    res_h = []
                    for hv in h_vecs:
                        tail = hv[rdim:] if rdim < hd else []
                        if arch.has_interleaved_rope:
                            rp = []
                            for i in range(half):
                                ang = position * (arch.rope_theta ** (-2.0 * i / rdim))
                                c, s = math.cos(ang), math.sin(ang)
                                x0, x1 = hv[2 * i], hv[2 * i + 1]
                                rp.append(x0 * c - x1 * s)
                                rp.append(x1 * c + x0 * s)
                            res_h.append(rp + tail)
                        else:
                            r1, r2 = [], []
                            for i in range(half):
                                ang = position * (arch.rope_theta ** (-2.0 * i / rdim))
                                c, s = math.cos(ang), math.sin(ang)
                                r1.append(hv[i] * c - hv[i + half] * s)
                                r2.append(hv[i + half] * c + hv[i] * s)
                            res_h.append(r1 + r2 + tail)
                    return res_h

                q_seq.append(rot_tok(qh_list, pos))
                k_seq.append(rot_tok(kh_list, pos))
                v_seq.append(vh_list)

            after_attn: list[list[float]] = []
            grp = nq // nkv
            sw = arch.sliding_window_for_layer(l_idx)
            for t in range(seq_len):
                win_start = max(0, (t + 1) - sw) if sw is not None else 0
                cat: list[float] = []
                for h_i in range(nq):
                    kv_i = h_i // grp
                    qv = q_seq[t][h_i]
                    sc = [
                        sum(qv[d] * k_seq[s][kv_i][d] for d in range(hd)) / math.sqrt(hd)
                        for s in range(win_start, t + 1)
                    ]
                    m_sc = max(sc)
                    ex = [math.exp(val - m_sc) for val in sc]
                    sm = sum(ex)
                    pr = [val / sm for val in ex]
                    cat.extend([sum(pr[p_i] * v_seq[win_start + p_i][kv_i][d] for p_i in range(len(pr))) for d in range(hd)])
                proj = cls._matvec(wo, cat)
                if w_post_sa is not None:
                    proj = cls._rms(proj, w_post_sa, eps)
                after_attn.append([a + b for a, b in zip(hidden[t], proj)])

            w_post = W(p + "post_attention_layernorm.weight")
            next_hidden: list[list[float]] = []
            if not arch.is_moe:
                wd = W(p + "mlp.down_proj.weight")
                if arch.has_fused_gate_up:
                    f_sz = int(arch.intermediate_size or 0)
                    wgu = W(p + "mlp.gate_up_proj.weight")
                    wg = wgu[:f_sz]
                    wu = wgu[f_sz:]
                else:
                    wg = W(p + "mlp.gate_proj.weight")
                    wu = W(p + "mlp.up_proj.weight")
                for t in range(seq_len):
                    ny = cls._rms(after_attn[t], w_post, eps)
                    gv = cls._matvec(wg, ny)
                    uv = cls._matvec(wu, ny)
                    act = [(g / (1.0 + math.exp(-g))) * u for g, u in zip(gv, uv)]
                    dv = cls._matvec(wd, act)
                    if w_post_mlp is not None:
                        dv = cls._rms(dv, w_post_mlp, eps)
                    next_hidden.append([a + b for a, b in zip(after_attn[t], dv)])
            else:
                assert arch.num_experts is not None and arch.num_experts_per_tok is not None
                wr = W(p + "mlp.gate.weight")
                for t in range(seq_len):
                    ny = cls._rms(after_attn[t], w_post, eps)
                    r_logits = cls._matvec(wr, ny)
                    ml = max(r_logits)
                    rex = [math.exp(v - ml) for v in r_logits]
                    rsm = sum(rex)
                    rpr = [v / rsm for v in rex]
                    top = sorted(range(arch.num_experts), key=lambda idx: (-rpr[idx], idx))[: arch.num_experts_per_tok]
                    tsum = sum(rpr[idx] for idx in top) if arch.norm_topk_prob else 1.0
                    acc = [0.0] * arch.hidden_size
                    for eid in top:
                        w_scale = rpr[eid] / tsum
                        eb = f"{p}mlp.experts.{eid}."
                        gv = cls._matvec(W(eb + "gate_proj.weight"), ny)
                        uv = cls._matvec(W(eb + "up_proj.weight"), ny)
                        act = [(g / (1.0 + math.exp(-g))) * u for g, u in zip(gv, uv)]
                        dv = cls._matvec(W(eb + "down_proj.weight"), act)
                        for hi in range(arch.hidden_size):
                            acc[hi] += w_scale * dv[hi]
                    next_hidden.append([a + b for a, b in zip(after_attn[t], acc)])
            hidden = next_hidden

        w_final = W("model.norm.weight")
        w_lm = emb if arch.tie_word_embeddings else W("lm_head.weight")
        return [cls._matvec(w_lm, cls._rms(row, w_final, eps)) for row in hidden]


__all__ = [
    "IndependentReferenceOracle",
    "LayerExecutionOutput",
    "MoERoutingReceipt",
    "RealizerError",
    "TransformerLayerRealizer",
    "decode_shaped_tensor",
    "hidden_state_digest",
    "route_moe_topk",
]
