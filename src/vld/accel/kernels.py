r"""Dual-engine numeric backend (`stdlib` pure Python + optional `numpy` C/SIMD vectorization).

Guarantees identical functional contracts when `dependencies = []` (zero third-party packages)
while automatically accelerating linear algebra, IEEE BF16/FP16/FP32 codecs, RMSNorm, RoPE,
and SwiGLU via `numpy` when installed.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Sequence

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised explicitly via mode="stdlib"
    _np = None


def has_numpy() -> bool:
    """Return True iff `numpy` is importable in the active Python runtime."""
    return _np is not None


def encode_f32_le(values: Sequence[float]) -> bytes:
    """Pack finite float values into little-endian IEEE 754 binary32 bytes."""
    for v in values:
        if not math.isfinite(float(v)):
            raise ValueError(f"Non-finite value rejected in encode_f32_le: {v!r}")
    if _np is not None:
        return _np.asarray(values, dtype="<f4").tobytes(order="C")
    return struct.pack(f"<{len(values)}f", *(float(v) for v in values))


def decode_f32_le(raw: bytes, *, use_numpy: bool | None = None) -> list[float]:
    """Unpack little-endian IEEE 754 binary32 bytes into Python floats, rejecting NaN/Inf."""
    if len(raw) % 4 != 0:
        raise ValueError("F32 byte buffer length must be a multiple of 4.")
    n = len(raw) // 4
    if n == 0:
        return []
    if (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None:
        arr = _np.frombuffer(raw, dtype="<f4")
        if not bool(_np.isfinite(arr).all()):
            raise ValueError("Decoded F32 tensor contains non-finite values.")
        return [float(x) for x in arr.tolist()]
    vals = list(struct.unpack(f"<{n}f", raw))
    if any(not math.isfinite(x) for x in vals):
        raise ValueError("Decoded F32 tensor contains non-finite values.")
    return vals


def encode_bf16_le(values: Sequence[float]) -> bytes:
    """Encode finite floats into round-to-nearest-even little-endian bfloat16 bytes."""
    f32_bytes = encode_f32_le(values)
    n = len(values)
    u32s = struct.unpack(f"<{n}I", f32_bytes)
    u16s: list[int] = []
    for bits in u32s:
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        top16 = ((bits + rounding_bias) >> 16) & 0xFFFF
        if (top16 & 0x7F80) == 0x7F80:
            raise ValueError("Finite float value overflows bfloat16 finite dynamic range.")
        u16s.append(top16)
    return struct.pack(f"<{n}H", *u16s)


def decode_bf16_le(raw: bytes, *, use_numpy: bool | None = None) -> list[float]:
    """Decode little-endian bfloat16 bytes into finite floats via exact 16-bit left shift into binary32."""
    if len(raw) % 2 != 0:
        raise ValueError("BF16 byte buffer length must be a multiple of 2.")
    n = len(raw) // 2
    if n == 0:
        return []
    if (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None:
        arr = (_np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
        if not bool(_np.isfinite(arr).all()):
            raise ValueError("Decoded BF16 tensor contains non-finite values.")
        return [float(x) for x in arr.tolist()]
    u16s = struct.unpack(f"<{n}H", raw)
    u32_bytes = struct.pack(f"<{n}I", *((u << 16) for u in u16s))
    vals = list(struct.unpack(f"<{n}f", u32_bytes))
    if any(not math.isfinite(x) for x in vals):
        raise ValueError("Decoded BF16 tensor contains non-finite values.")
    return vals


def decode_f16_le(raw: bytes, *, use_numpy: bool | None = None) -> list[float]:
    """Decode little-endian IEEE 754 binary16 bytes into finite floats."""
    if len(raw) % 2 != 0:
        raise ValueError("F16 byte buffer length must be a multiple of 2.")
    n = len(raw) // 2
    if n == 0:
        return []
    if (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None:
        arr = _np.frombuffer(raw, dtype="<f2").astype("<f4")
        if not bool(_np.isfinite(arr).all()):
            raise ValueError("Decoded F16 tensor contains non-finite values.")
        return [float(x) for x in arr.tolist()]
    vals = list(struct.unpack(f"<{n}e", raw))
    if any(not math.isfinite(x) for x in vals):
        raise ValueError("Decoded F16 tensor contains non-finite values.")
    return vals


def decode_tensor_ndarray(raw: bytes, dtype: str, shape: tuple[int, ...]) -> Any:
    """Decode raw BF16/F16/F32 bytes directly into a contiguous C-order float32 `numpy.ndarray`."""
    if _np is None:
        raise RuntimeError("decode_tensor_ndarray requires numpy.")
    expected = math.prod(shape)
    if dtype == "BF16":
        if len(raw) != expected * 2:
            raise ValueError(f"Expected {expected * 2} BF16 bytes for shape {shape}, got {len(raw)}.")
        arr = (_np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view("<f4")
    elif dtype == "F16":
        if len(raw) != expected * 2:
            raise ValueError(f"Expected {expected * 2} F16 bytes for shape {shape}, got {len(raw)}.")
        arr = _np.frombuffer(raw, dtype="<f2").astype("<f4")
    elif dtype == "F32":
        if len(raw) != expected * 4:
            raise ValueError(f"Expected {expected * 4} F32 bytes for shape {shape}, got {len(raw)}.")
        arr = _np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError(f"Unsupported numerical tensor dtype {dtype!r}.")
    if not bool(_np.isfinite(arr).all()):
        raise ValueError(f"Decoded {dtype} tensor contains non-finite values.")
    return _np.ascontiguousarray(arr.reshape(shape), dtype=_np.float32)


class NumericEngine:
    """Selectable numeric execution engine (`auto`, `numpy`, or `stdlib`)."""

    def __init__(self, mode: str = "auto") -> None:
        if mode not in ("auto", "numpy", "stdlib"):
            raise ValueError(f"Unsupported NumericEngine mode: {mode!r}")
        if mode == "numpy" and _np is None:
            raise RuntimeError("NumericEngine(mode='numpy') requested, but numpy is not installed.")
        self.mode = mode
        self.uses_numpy = (_np is not None) and (mode in ("auto", "numpy"))

    def asarray_f32(self, values: Any) -> Any:
        """Convert nested float sequences or array-likes into a contiguous float32 `ndarray`."""
        if _np is None:
            raise RuntimeError("asarray_f32 requires numpy.")
        return _np.ascontiguousarray(values, dtype=_np.float32)

    def linear_x_wt_nd(self, x_nd: Any, weight_out_in_nd: Any) -> Any:
        r"""Contiguous `ndarray`-resident $Y = X W^\top$ without Python list marshaling."""
        assert _np is not None
        return _np.ascontiguousarray(x_nd @ weight_out_in_nd.T, dtype=_np.float32)

    def rms_norm_nd(self, x_nd: Any, weight_nd: Any, eps: float) -> Any:
        r"""Contiguous `ndarray`-resident row/last-axis RMSNorm across arbitrary leading rank."""
        assert _np is not None
        inv_rms = _np.reciprocal(_np.sqrt(_np.mean(x_nd * x_nd, axis=-1, keepdims=True) + _np.float32(eps)))
        return _np.ascontiguousarray((x_nd * inv_rms) * weight_nd, dtype=_np.float32)

    def rope_3d_nd(self, heads_3d_nd: Any, *, start_position: int, rope_theta: float) -> Any:
        r"""Vectorized `ndarray`-resident Rotary Position Embedding over `[seq_len, n_heads, head_dim]`."""
        assert _np is not None
        seq_len, _, head_dim = heads_3d_nd.shape
        if seq_len == 0:
            return heads_3d_nd.copy()
        half = head_dim // 2
        inv_freq = rope_theta ** (-2.0 * _np.arange(half, dtype=_np.float64) / float(head_dim))
        positions = (_np.arange(seq_len, dtype=_np.float64) + float(start_position))[:, None]
        angles = positions * inv_freq[None, :]  # [seq_len, half]
        cos_a = _np.cos(angles)[:, None, :].astype(_np.float32)
        sin_a = _np.sin(angles)[:, None, :].astype(_np.float32)
        first = heads_3d_nd[:, :, :half]
        second = heads_3d_nd[:, :, half:]
        rot_first = first * cos_a - second * sin_a
        rot_second = second * cos_a + first * sin_a
        return _np.ascontiguousarray(_np.concatenate((rot_first, rot_second), axis=-1), dtype=_np.float32)

    def causal_gqa_nd(
        self,
        q_rot_nd: Any,
        k_full_nd: Any,
        v_full_nd: Any,
        *,
        start_position: int,
    ) -> Any:
        r"""Contiguous `ndarray`-resident Causal Grouped-Query Attention returning `[seq_len, n_heads * head_dim]`."""
        assert _np is not None
        seq_len, nq, hd = q_rot_nd.shape
        total_seq, nkv, _ = k_full_nd.shape
        group_size = nq // nkv
        # Expand KV heads to match Q heads via zero-copy repeat view where possible: [total_seq, nq, hd]
        k_exp = _np.repeat(k_full_nd, group_size, axis=1).astype(_np.float64)
        v_exp = _np.repeat(v_full_nd, group_size, axis=1).astype(_np.float64)
        q64 = q_rot_nd.astype(_np.float64)
        inv_sqrt_d = 1.0 / math.sqrt(float(hd))

        # Scores: [nq, seq_len, total_seq]
        scores = _np.einsum("tqd,sqd->qts", q64, k_exp) * inv_sqrt_d
        q_pos = (_np.arange(seq_len) + start_position)[:, None]
        k_pos = _np.arange(total_seq)[None, :]
        causal_mask = k_pos <= q_pos  # [seq_len, total_seq]
        masked_scores = _np.where(causal_mask[None, :, :], scores, -_np.inf)
        shifted = _np.exp(masked_scores - _np.max(masked_scores, axis=-1, keepdims=True))
        probs = shifted / _np.sum(shifted, axis=-1, keepdims=True)  # [nq, seq_len, total_seq]
        ctx = _np.einsum("qts,sqd->tqd", probs, v_exp)  # [seq_len, nq, hd]
        return _np.ascontiguousarray(ctx.reshape(seq_len, nq * hd), dtype=_np.float32)

    def swiglu_nd(self, gate_nd: Any, up_nd: Any) -> Any:
        r"""Contiguous `ndarray`-resident $\operatorname{SiLU}(\text{gate}) \odot \text{up}$."""
        assert _np is not None
        g64 = gate_nd.astype(_np.float64)
        sig = _np.where(g64 >= 0.0, 1.0 / (1.0 + _np.exp(-g64)), _np.exp(g64) / (1.0 + _np.exp(g64)))
        ya = (g64 * sig).astype(_np.float32) * up_nd
        return _np.ascontiguousarray(ya, dtype=_np.float32)

    def linear_x_wt(self, x: Sequence[Sequence[float]], weight_out_in: Sequence[Sequence[float]]) -> list[list[float]]:
        r"""Compute $Y = X W^\top$ where $X \in \mathbb{R}^{S \times D_{\text{in}}}$ and $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$."""
        if self.uses_numpy and _np is not None:
            xa = _np.asarray(x, dtype=_np.float32)
            wa = _np.asarray(weight_out_in, dtype=_np.float32)
            ya = xa @ wa.T
            return [[float(v) for v in row] for row in ya.tolist()]
        out: list[list[float]] = []
        for row in x:
            out.append([math.fsum(float(a) * float(b) for a, b in zip(row, w_row)) for w_row in weight_out_in])
        return out

    def rms_norm_2d(
        self,
        x: Sequence[Sequence[float]],
        weight: Sequence[float],
        eps: float,
    ) -> list[list[float]]:
        r"""Compute row-wise $\operatorname{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{D}\sum_j x_{i,j}^2 + \epsilon}} \odot w$."""
        if self.uses_numpy and _np is not None:
            xa = _np.asarray(x, dtype=_np.float32)
            wa = _np.asarray(weight, dtype=_np.float32)
            inv_rms = _np.reciprocal(_np.sqrt(_np.mean(xa * xa, axis=-1, keepdims=True) + _np.float32(eps)))
            ya = (xa * inv_rms) * wa
            return [[float(v) for v in row] for row in ya.tolist()]
        d = len(weight)
        out: list[list[float]] = []
        for row in x:
            mean_sq = math.fsum(float(v) * float(v) for v in row) / d
            scale = 1.0 / math.sqrt(mean_sq + float(eps))
            out.append([float(v) * scale * float(w) for v, w in zip(row, weight)])
        return out

    def softmax_1d(self, scores: Sequence[float]) -> list[float]:
        """Numerically stable 1D softmax."""
        if not scores:
            return []
        if self.uses_numpy and _np is not None:
            sa = _np.asarray(scores, dtype=_np.float64)
            shifted = _np.exp(sa - _np.max(sa))
            probs = shifted / _np.sum(shifted)
            return [float(v) for v in probs.tolist()]
        m = max(scores)
        exps = [math.exp(float(s) - m) for s in scores]
        denom = math.fsum(exps)
        return [e / denom for e in exps]

    def swiglu_2d(
        self,
        gate: Sequence[Sequence[float]],
        up: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        r"""Compute $\operatorname{SiLU}(\text{gate}) \odot \text{up}$ where $\operatorname{SiLU}(z) = \frac{z}{1 + e^{-z}}$."""
        if self.uses_numpy and _np is not None:
            ga = _np.asarray(gate, dtype=_np.float32)
            ua = _np.asarray(up, dtype=_np.float32)
            ya = (ga / (_np.float32(1.0) + _np.exp(-ga))) * ua
            return [[float(v) for v in row] for row in ya.tolist()]
        out: list[list[float]] = []
        for g_row, u_row in zip(gate, up):
            row_vals: list[float] = []
            for g, u in zip(g_row, u_row):
                gf = float(g)
                sig = 1.0 / (1.0 + math.exp(-gf)) if gf >= 0.0 else (math.exp(gf) / (1.0 + math.exp(gf)))
                row_vals.append((gf * sig) * float(u))
            out.append(row_vals)
        return out

    @staticmethod
    def max_abs_diff(a: Any, b: Any) -> float:
        """Recursively compute maximum absolute elementwise discrepancy."""
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                raise ValueError("Shape mismatch in max_abs_diff.")
            return max((NumericEngine.max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
        return abs(float(a) - float(b))
