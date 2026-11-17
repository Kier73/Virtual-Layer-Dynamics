r"""Phase 6 — Verified Transformer & Sparse MoE Numerical Realizers + Independent FP64 Reference Oracle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from vld.accel import NumericEngine, decode_bf16_le, decode_f16_le, decode_f32_le, encode_f32_le
from vld.canonical import domain_digest
from vld.memory import PagedObjectStore
from vld.models.artifacts import ModelArtifactError, ModelWeightSnapshot, TransformerArchitectureSpec


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
) -> list[list[list[float]]]:
    seq_len = len(heads_3d)
    if seq_len == 0:
        return []
    head_dim = len(heads_3d[0][0])
    half = head_dim // 2
    inv_freq = [rope_theta ** (-2.0 * i / head_dim) for i in range(half)]
    out: list[list[list[float]]] = []
    for t in range(seq_len):
        pos = start_position + t
        cos_half = [math.cos(pos * f) for f in inv_freq]
        sin_half = [math.sin(pos * f) for f in inv_freq]
        token_heads: list[list[float]] = []
        for h_vec in heads_3d[t]:
            first = h_vec[:half]
            second = h_vec[half:]
            rot_first = [x1 * c - x2 * s for x1, x2, c, s in zip(first, second, cos_half, sin_half)]
            rot_second = [x2 * c + x1 * s for x1, x2, c, s in zip(first, second, cos_half, sin_half)]
            token_heads.append(rot_first + rot_second)
        out.append(token_heads)
    return out


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
        snapshot: ModelWeightSnapshot,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
        engine: NumericEngine | None = None,
    ) -> None:
        self.arch = arch
        self.snapshot = snapshot
        self.store = store
        self.remote_range_fetcher = remote_range_fetcher
        self.engine = engine if engine is not None else NumericEngine("auto")

    def _load_tensor(self, name: str, loaded_log: list[str]) -> Any:
        t_space = self.snapshot.space.tensor(name)
        raw = self.snapshot.read_tensor_bytes(
            name,
            store=self.store,
            remote_range_fetcher=self.remote_range_fetcher,
        )
        loaded_log.append(name)
        return decode_shaped_tensor(raw, t_space.dtype, t_space.shape, use_numpy=self.engine.uses_numpy)

    def embed_tokens(self, token_ids: Sequence[int]) -> list[list[float]]:
        loaded: list[str] = []
        emb_table = self._load_tensor("model.embed_tokens.weight", loaded)
        out: list[list[float]] = []
        for tid in token_ids:
            if not (0 <= tid < self.arch.vocab_size):
                raise RealizerError(f"Token ID {tid} outside vocab_size {self.arch.vocab_size}.")
            out.append(list(emb_table[tid]))
        return out

    def final_logits(self, hidden_states: Sequence[Sequence[float]]) -> list[list[float]]:
        loaded: list[str] = []
        norm_w = self._load_tensor("model.norm.weight", loaded)
        normed = self.engine.rms_norm_2d(hidden_states, norm_w, self.arch.rms_norm_eps)
        head_name = "model.embed_tokens.weight" if self.arch.tie_word_embeddings else "lm_head.weight"
        lm_w = self._load_tensor(head_name, loaded)
        return self.engine.linear_x_wt(normed, lm_w)

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
        v_heads = [
            [[float(x) for x in v_flat[t][h * hd : (h + 1) * hd]] for h in range(nkv)]
            for t in range(seq_len)
        ]

        q_rot = _apply_rope_3d(q_heads, start_position=start_pos, rope_theta=self.arch.rope_theta)
        k_rot = _apply_rope_3d(k_heads, start_position=start_pos, rope_theta=self.arch.rope_theta)

        k_full = [
            *([[list(hv) for hv in tok] for tok in past_k]),
            *k_rot,
        ]
        v_full = [
            *([[list(hv) for hv in tok] for tok in past_v]),
            *v_heads,
        ]

        # 2. Causal Grouped-Query Attention (GQA)
        group_size = nq // nkv
        inv_sqrt_d = 1.0 / math.sqrt(hd)
        attn_concat: list[list[float]] = []
        for t_rel in range(seq_len):
            causal_len = start_pos + t_rel + 1
            tok_out_flat: list[float] = []
            for qh in range(nq):
                kvh = qh // group_size
                q_vec = q_rot[t_rel][qh]
                scores = [
                    math.fsum(q_vec[d_i] * k_full[s_idx][kvh][d_i] for d_i in range(hd)) * inv_sqrt_d
                    for s_idx in range(causal_len)
                ]
                probs = self.engine.softmax_1d(scores)
                ctx = [
                    math.fsum(probs[s_idx] * v_full[s_idx][kvh][d_i] for s_idx in range(causal_len))
                    for d_i in range(hd)
                ]
                tok_out_flat.extend(ctx)
            attn_concat.append(tok_out_flat)

        o_proj_w = self._load_tensor(p + "self_attn.o_proj.weight", loaded_log)
        attn_proj = self.engine.linear_x_wt(attn_concat, o_proj_w)
        post_attn = [
            [float(x0) + float(dx) for x0, dx in zip(r0, dr)]
            for r0, dr in zip(hidden_states, attn_proj)
        ]

        # 3. Post-attention RMSNorm + Dense SwiGLU OR Active-Expert-Only Sparse MoE FFN
        post_norm_w = self._load_tensor(p + "post_attention_layernorm.weight", loaded_log)
        normed_ffn = self.engine.rms_norm_2d(post_attn, post_norm_w, eps)

        moe_receipts: list[MoERoutingReceipt] = []
        if not self.arch.is_moe:
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
        half = hd // 2
        eps = arch.rms_norm_eps

        for l_idx in range(arch.num_hidden_layers):
            p = f"model.layers.{l_idx}."
            w_in = W(p + "input_layernorm.weight")
            wq = W(p + "self_attn.q_proj.weight")
            wk = W(p + "self_attn.k_proj.weight")
            wv = W(p + "self_attn.v_proj.weight")
            wo = W(p + "self_attn.o_proj.weight")
            wqn = W(p + "self_attn.q_norm.weight")
            wkn = W(p + "self_attn.k_norm.weight")

            q_seq: list[list[list[float]]] = []
            k_seq: list[list[list[float]]] = []
            v_seq: list[list[list[float]]] = []
            for pos in range(seq_len):
                nx = cls._rms(hidden[pos], w_in, eps)
                qf = cls._matvec(wq, nx)
                kf = cls._matvec(wk, nx)
                vf = cls._matvec(wv, nx)
                qh_list = [cls._rms(qf[h * hd : (h + 1) * hd], wqn, eps) for h in range(nq)]
                kh_list = [cls._rms(kf[h * hd : (h + 1) * hd], wkn, eps) for h in range(nkv)]
                vh_list = [vf[h * hd : (h + 1) * hd] for h in range(nkv)]

                def rot_tok(h_vecs: list[list[float]], position: int) -> list[list[float]]:
                    res_h = []
                    for hv in h_vecs:
                        r1, r2 = [], []
                        for i in range(half):
                            ang = position * (arch.rope_theta ** (-2.0 * i / hd))
                            c, s = math.cos(ang), math.sin(ang)
                            r1.append(hv[i] * c - hv[i + half] * s)
                            r2.append(hv[i + half] * c + hv[i] * s)
                        res_h.append(r1 + r2)
                    return res_h

                q_seq.append(rot_tok(qh_list, pos))
                k_seq.append(rot_tok(kh_list, pos))
                v_seq.append(vh_list)

            after_attn: list[list[float]] = []
            grp = nq // nkv
            for t in range(seq_len):
                cat: list[float] = []
                for h_i in range(nq):
                    kv_i = h_i // grp
                    qv = q_seq[t][h_i]
                    sc = [
                        sum(qv[d] * k_seq[s][kv_i][d] for d in range(hd)) / math.sqrt(hd)
                        for s in range(t + 1)
                    ]
                    m_sc = max(sc)
                    ex = [math.exp(val - m_sc) for val in sc]
                    sm = sum(ex)
                    pr = [val / sm for val in ex]
                    cat.extend([sum(pr[s] * v_seq[s][kv_i][d] for s in range(t + 1)) for d in range(hd)])
                proj = cls._matvec(wo, cat)
                after_attn.append([a + b for a, b in zip(hidden[t], proj)])

            w_post = W(p + "post_attention_layernorm.weight")
            next_hidden: list[list[float]] = []
            if not arch.is_moe:
                wg = W(p + "mlp.gate_proj.weight")
                wu = W(p + "mlp.up_proj.weight")
                wd = W(p + "mlp.down_proj.weight")
                for t in range(seq_len):
                    ny = cls._rms(after_attn[t], w_post, eps)
                    gv = cls._matvec(wg, ny)
                    uv = cls._matvec(wu, ny)
                    act = [(g / (1.0 + math.exp(-g))) * u for g, u in zip(gv, uv)]
                    dv = cls._matvec(wd, act)
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
