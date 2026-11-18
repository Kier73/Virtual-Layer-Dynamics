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


# ============================================================================
# Microscaling / Frontier Quantization Codecs: F8_E4M3(FN), F8_E8M0 & FP4 E2M1
# ============================================================================

FP4_E2M1_TABLE: tuple[float, ...] = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _build_f8_e4m3_lut() -> tuple[float, ...]:
    lut: list[float] = []
    for b in range(256):
        sign = -1.0 if (b & 0x80) else 1.0
        exp = (b >> 3) & 0x0F
        man = b & 0x07
        if exp == 15 and man == 7:
            lut.append(math.nan)
        elif exp == 0:
            lut.append(sign * math.ldexp(float(man), -9))
        else:
            lut.append(sign * math.ldexp(8.0 + float(man), exp - 10))
    return tuple(lut)


def _build_f8_e8m0_lut() -> tuple[float, ...]:
    return tuple(math.nan if b == 255 else math.ldexp(1.0, b - 127) for b in range(256))


_F8_E4M3_LUT: tuple[float, ...] = _build_f8_e4m3_lut()
_F8_E8M0_LUT: tuple[float, ...] = _build_f8_e8m0_lut()
_NP_F8_E4M3_LUT: Any = _np.asarray(_F8_E4M3_LUT, dtype=_np.float64) if _np is not None else None
_NP_F8_E8M0_LUT: Any = _np.asarray(_F8_E8M0_LUT, dtype=_np.float64) if _np is not None else None
_NP_FP4_LUT: Any = _np.asarray(FP4_E2M1_TABLE, dtype=_np.float64) if _np is not None else None


def decode_f8_e4m3(raw: bytes, *, use_numpy: bool | None = None) -> list[float]:
    r"""Decode OCP `F8_E4M3` / `F8_E4M3FN` bytes (bias 7, max $\pm 448.0$), rejecting `0x7F`/`0xFF` NaNs."""
    if not raw:
        return []
    if (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None:
        u8 = _np.frombuffer(raw, dtype=_np.uint8)
        vals_nd = _NP_F8_E4M3_LUT[u8]
        if not bool(_np.isfinite(vals_nd).all()):
            raise ValueError("Decoded F8_E4M3 tensor contains NaN byte (0x7F or 0xFF).")
        return [float(x) for x in vals_nd.tolist()]
    out: list[float] = []
    for b in raw:
        v = _F8_E4M3_LUT[b]
        if not math.isfinite(v):
            raise ValueError(f"Decoded F8_E4M3 tensor contains NaN byte {b:#04x}.")
        out.append(v)
    return out


def encode_f8_e4m3(values: Sequence[float]) -> bytes:
    """Quantize finite floats in `[-448.0, 448.0]` to nearest OCP `F8_E4M3` byte."""
    finite_candidates = [(b, _F8_E4M3_LUT[b]) for b in range(256) if math.isfinite(_F8_E4M3_LUT[b])]
    out = bytearray(len(values))
    for i, val in enumerate(values):
        fv = float(val)
        if not math.isfinite(fv) or abs(fv) > 448.0:
            raise ValueError(f"Value {val!r} outside finite F8_E4M3 range [-448.0, 448.0].")
        if fv == 0.0:
            out[i] = 0x80 if math.copysign(1.0, fv) < 0.0 else 0x00
            continue
        best_b = min(finite_candidates, key=lambda item: (abs(item[1] - fv), item[0]))[0]
        out[i] = best_b
    return bytes(out)


def decode_f8_e8m0(raw: bytes, *, use_numpy: bool | None = None) -> list[float]:
    r"""Decode `F8_E8M0` (`UE8M0`) unsigned power-of-two scale bytes $2^{u - 127}$, rejecting `0xFF` NaN."""
    if not raw:
        return []
    if (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None:
        u8 = _np.frombuffer(raw, dtype=_np.uint8)
        vals_nd = _NP_F8_E8M0_LUT[u8]
        if not bool(_np.isfinite(vals_nd).all()):
            raise ValueError("Decoded F8_E8M0 scale tensor contains NaN byte (0xFF).")
        return [float(x) for x in vals_nd.tolist()]
    out: list[float] = []
    for b in raw:
        if b == 255:
            raise ValueError("Decoded F8_E8M0 scale tensor contains NaN byte (0xFF).")
        out.append(_F8_E8M0_LUT[b])
    return out


def encode_f8_e8m0(scales: Sequence[float]) -> bytes:
    r"""Encode positive scales into power-of-two ceiling `F8_E8M0` (`UE8M0`) bytes $u = \lceil \log_2 s \rceil + 127$."""
    out = bytearray(len(scales))
    for idx, s in enumerate(scales):
        fs = float(s)
        if not math.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"F8_E8M0 scale must be finite and positive, got {s!r}.")
        exp = math.ceil(math.log2(fs))
        u = exp + 127
        if not (0 <= u <= 254):
            raise ValueError(f"Scale {s!r} exponent {exp} outside UE8M0 range [-127, 127].")
        out[idx] = u
    return bytes(out)


def _decode_scale_2d(
    scale_raw: bytes,
    scale_shape: tuple[int, int],
    scale_dtype: str,
    *,
    use_numpy: bool | None,
) -> Any:
    sr, sc = scale_shape
    if sr <= 0 or sc <= 0:
        raise ValueError("scale_shape dimensions must be positive.")
    if scale_dtype == "F8_E8M0":
        flat = decode_f8_e8m0(scale_raw, use_numpy=use_numpy)
    elif scale_dtype == "F32":
        flat = decode_f32_le(scale_raw, use_numpy=use_numpy)
    elif scale_dtype == "BF16":
        flat = decode_bf16_le(scale_raw, use_numpy=use_numpy)
    else:
        raise ValueError(f"Unsupported scale_dtype {scale_dtype!r}.")
    if len(flat) != sr * sc:
        raise ValueError(f"Scale element count {len(flat)} != {sr * sc} for shape {scale_shape}.")
    return flat


def _resolve_block_axis(dim: int, scale_dim: int, explicit_block: int | None) -> int:
    r"""Resolve block granularity $b$ for dimension $d$ and scale count $s = \lceil d / b \rceil$.

    By the dyadic uniqueness inequality $d/s \le b < d/(s-1)$ ($s \ge 2 \implies \frac{s}{s-1} \le 2$),
    at most one power-of-two block size $b = 2^k$ can ever satisfy $\lceil d / b \rceil = s$.
    """
    if explicit_block is not None:
        if explicit_block <= 0 or math.ceil(dim / explicit_block) != scale_dim:
            raise ValueError(
                f"Explicit block size {explicit_block} incompatible with dim={dim} and scale_dim={scale_dim}."
            )
        return explicit_block
    if scale_dim == 1:
        return dim
    cand = 1 << max(0, math.ceil(math.log2(dim / scale_dim)))
    if math.ceil(dim / cand) == scale_dim:
        return cand
    fallback = math.ceil(dim / scale_dim)
    if math.ceil(dim / fallback) == scale_dim:
        return fallback
    raise ValueError(f"Incompatible dim={dim} and scale_dim={scale_dim}.")


def dequantize_fp8_blockwise(
    weight_raw: bytes,
    weight_shape: tuple[int, int],
    scale_raw: bytes,
    scale_shape: tuple[int, int],
    *,
    block_shape: tuple[int, int] | None = None,
    scale_dtype: str = "F8_E8M0",
    use_numpy: bool | None = None,
    as_ndarray: bool = False,
) -> Any:
    r"""Dequantize 2D `F8_E4M3(FN)` matrix $W \in \mathbb{R}^{R \times C}$ with block scale $S \in \mathbb{R}^{S_R \times S_C}$."""
    rows, cols = weight_shape
    sr, sc = scale_shape
    if rows <= 0 or cols <= 0 or sr <= 0 or sc <= 0:
        raise ValueError("weight_shape and scale_shape must be positive 2D shapes.")
    if len(weight_raw) != rows * cols:
        raise ValueError(f"Expected {rows * cols} FP8 bytes, got {len(weight_raw)}.")
    block_r = _resolve_block_axis(rows, sr, None if block_shape is None else int(block_shape[0]))
    block_c = _resolve_block_axis(cols, sc, None if block_shape is None else int(block_shape[1]))

    want_np = (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None
    if as_ndarray and _np is None:
        raise RuntimeError("as_ndarray=True requires numpy.")

    if want_np:
        assert _np is not None
        u8_w = _np.frombuffer(weight_raw, dtype=_np.uint8).reshape(rows, cols)
        w64 = _NP_F8_E4M3_LUT[u8_w]
        if not bool(_np.isfinite(w64).all()):
            raise ValueError("FP8 weight tensor contains NaN byte (0x7F or 0xFF).")
        if scale_dtype == "F8_E8M0":
            if len(scale_raw) != sr * sc:
                raise ValueError(f"Expected {sr * sc} F8_E8M0 scale bytes, got {len(scale_raw)}.")
            s64 = _NP_F8_E8M0_LUT[_np.frombuffer(scale_raw, dtype=_np.uint8).reshape(sr, sc)]
        elif scale_dtype == "F32":
            s64 = _np.frombuffer(scale_raw, dtype="<f4").reshape(sr, sc).astype(_np.float64)
        elif scale_dtype == "BF16":
            s64 = ((_np.frombuffer(scale_raw, dtype="<u2").astype("<u4") << 16).view("<f4")).reshape(sr, sc).astype(_np.float64)
        else:
            raise ValueError(f"Unsupported scale_dtype {scale_dtype!r}.")
        if not bool(_np.isfinite(s64).all()):
            raise ValueError("FP8 scale tensor contains non-finite values.")
        r_idx = _np.arange(rows) // block_r
        c_idx = _np.arange(cols) // block_c
        out32 = _np.ascontiguousarray(w64 * s64[_np.ix_(r_idx, c_idx)], dtype=_np.float32)
        if not bool(_np.isfinite(out32).all()):
            raise ValueError("Dequantized FP8 tensor overflowed float32.")
        return out32 if as_ndarray else [[float(v) for v in row] for row in out32.tolist()]

    w_flat = decode_f8_e4m3(weight_raw, use_numpy=False)
    s_flat = _decode_scale_2d(scale_raw, scale_shape, scale_dtype, use_numpy=False)
    out_rows: list[list[float]] = []
    for r in range(rows):
        br = r // block_r
        row_vals: list[float] = []
        w_off = r * cols
        s_off = br * sc
        for c in range(cols):
            val = w_flat[w_off + c] * s_flat[s_off + (c // block_c)]
            if not math.isfinite(val) or abs(val) > 3.402823466e38:
                raise ValueError("Dequantized FP8 value overflowed finite float32.")
            row_vals.append(float(struct.unpack("<f", struct.pack("<f", val))[0]))
        out_rows.append(row_vals)
    return out_rows


def dequantize_fp4_e2m1_packed(
    weight_raw: bytes,
    logical_shape: tuple[int, int],
    scale_raw: bytes,
    scale_shape: tuple[int, int],
    *,
    block_shape: tuple[int, int] | None = None,
    scale_dtype: str = "F8_E8M0",
    use_numpy: bool | None = None,
    as_ndarray: bool = False,
) -> Any:
    r"""Dequantize packed MXFP4 (`float4_e2m1fn_x2` stored in `I8` `[R, C//2]`) with block scale `S`."""
    rows, cols = logical_shape
    sr, sc = scale_shape
    if rows <= 0 or cols <= 0 or cols % 2 != 0 or sr <= 0 or sc <= 0:
        raise ValueError("logical_shape must have positive rows and even cols.")
    if len(weight_raw) != rows * (cols // 2):
        raise ValueError(f"Expected {rows * (cols // 2)} packed FP4 bytes, got {len(weight_raw)}.")
    block_r = _resolve_block_axis(rows, sr, None if block_shape is None else int(block_shape[0]))
    block_c = _resolve_block_axis(cols, sc, None if block_shape is None else int(block_shape[1]))

    want_np = (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None
    if as_ndarray and _np is None:
        raise RuntimeError("as_ndarray=True requires numpy.")

    if want_np:
        assert _np is not None
        packed = _np.frombuffer(weight_raw, dtype=_np.uint8).reshape(rows, cols // 2)
        low = _NP_FP4_LUT[packed & 0x0F]
        high = _NP_FP4_LUT[(packed >> 4) & 0x0F]
        w64 = _np.empty((rows, cols), dtype=_np.float64)
        w64[:, 0::2] = low
        w64[:, 1::2] = high
        if scale_dtype == "F8_E8M0":
            if len(scale_raw) != sr * sc:
                raise ValueError(f"Expected {sr * sc} F8_E8M0 scale bytes, got {len(scale_raw)}.")
            s64 = _NP_F8_E8M0_LUT[_np.frombuffer(scale_raw, dtype=_np.uint8).reshape(sr, sc)]
        elif scale_dtype == "F32":
            s64 = _np.frombuffer(scale_raw, dtype="<f4").reshape(sr, sc).astype(_np.float64)
        else:
            raise ValueError(f"Unsupported FP4 scale_dtype {scale_dtype!r}.")
        if not bool(_np.isfinite(s64).all()):
            raise ValueError("FP4 scale tensor contains non-finite values.")
        r_idx = _np.arange(rows) // block_r
        c_idx = _np.arange(cols) // block_c
        out32 = _np.ascontiguousarray(w64 * s64[_np.ix_(r_idx, c_idx)], dtype=_np.float32)
        if not bool(_np.isfinite(out32).all()):
            raise ValueError("Dequantized FP4 tensor overflowed float32.")
        return out32 if as_ndarray else [[float(v) for v in row] for row in out32.tolist()]

    s_flat = _decode_scale_2d(scale_raw, scale_shape, scale_dtype, use_numpy=False)
    out_rows: list[list[float]] = []
    half_c = cols // 2
    for r in range(rows):
        br = r // block_r
        s_off = br * sc
        row_vals = [0.0] * cols
        b_off = r * half_c
        for cb in range(half_c):
            byte_v = weight_raw[b_off + cb]
            c0 = 2 * cb
            c1 = c0 + 1
            v0 = FP4_E2M1_TABLE[byte_v & 0x0F] * s_flat[s_off + (c0 // block_c)]
            v1 = FP4_E2M1_TABLE[(byte_v >> 4) & 0x0F] * s_flat[s_off + (c1 // block_c)]
            row_vals[c0] = float(struct.unpack("<f", struct.pack("<f", v0))[0])
            row_vals[c1] = float(struct.unpack("<f", struct.pack("<f", v1))[0])
        out_rows.append(row_vals)
    return out_rows


def encode_fp4_e2m1_packed(
    values_2d: Sequence[Sequence[float]],
    *,
    block_k: int = 32,
) -> tuple[bytes, bytes]:
    r"""Encode a 2D matrix `[R, C]` into `(packed_i8_bytes, ue8m0_scale_bytes)` with `block_k` scaling."""
    rows = len(values_2d)
    if rows == 0:
        raise ValueError("values_2d cannot be empty.")
    cols = len(values_2d[0])
    if cols == 0 or cols % block_k != 0 or block_k % 2 != 0 or any(len(r) != cols for r in values_2d):
        raise ValueError("cols must be a positive multiple of even block_k.")
    packed = bytearray(rows * (cols // 2))
    scales: list[float] = []
    pos_nibbles = [(idx, FP4_E2M1_TABLE[idx]) for idx in range(16)]
    p_cursor = 0
    for r in range(rows):
        row = values_2d[r]
        for b_start in range(0, cols, block_k):
            blk = [float(x) for x in row[b_start : b_start + block_k]]
            if any(not math.isfinite(x) for x in blk):
                raise ValueError("Non-finite value in encode_fp4_e2m1_packed.")
            amax = max((abs(x) for x in blk), default=0.0)
            raw_scale = max(amax / 6.0, 2.0**-126)
            exp = max(-127, min(127, math.ceil(math.log2(raw_scale))))
            sc_val = math.ldexp(1.0, exp)
            scales.append(sc_val)
            for j in range(0, block_k, 2):
                q0 = blk[j] / sc_val
                q1 = blk[j + 1] / sc_val
                n0 = min(pos_nibbles, key=lambda it: (abs(it[1] - q0), it[0]))[0]
                n1 = min(pos_nibbles, key=lambda it: (abs(it[1] - q1), it[0]))[0]
                packed[p_cursor] = (n0 & 0x0F) | ((n1 & 0x0F) << 4)
                p_cursor += 1
    return bytes(packed), encode_f8_e8m0(scales)


def quantize_dequantize_fp4_rows(
    values_2d: Sequence[Sequence[float]],
    *,
    block_k: int = 32,
    scale_dtype: str = "F8_E8M0",
) -> list[list[float]]:
    r"""Fused MXFP4 quantize + dequantize (`fp4_act_quant(..., inplace=True)`) with `F8_E8M0` or `F8_E4M3` block scales."""
    if not values_2d:
        return []
    cols = len(values_2d[0])
    if cols <= 0 or block_k <= 0 or cols % block_k != 0 or any(len(r) != cols for r in values_2d):
        raise ValueError("Rows must have positive width divisible by block_k.")
    if scale_dtype not in ("F8_E8M0", "F8_E4M3"):
        raise ValueError(f"Unsupported fp4 activation scale_dtype {scale_dtype!r}.")
    finite_e4m3 = [_F8_E4M3_LUT[b] for b in range(127)]  # non-negative finite E4M3 values [0..448]
    pos_fp4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    out_rows: list[list[float]] = []
    for row in values_2d:
        frow = [float(x) for x in row]
        if any(not math.isfinite(x) for x in frow):
            raise ValueError("Non-finite value in quantize_dequantize_fp4_rows.")
        out_r: list[float] = []
        for b_start in range(0, cols, block_k):
            blk = frow[b_start : b_start + block_k]
            amax = max(abs(x) for x in blk)
            if scale_dtype == "F8_E4M3":
                target_s = min(448.0, max(amax, 6.0 * (2.0**-9)) / 6.0)
                sc_val = min(finite_e4m3, key=lambda cand: abs(cand - target_s))
            else:
                target_s = max(amax, 6.0 * (2.0**-126)) / 6.0
                exp = max(-127, min(127, math.ceil(math.log2(target_s))))
                sc_val = math.ldexp(1.0, exp)
            for x in blk:
                clamped = max(-6.0, min(6.0, x / sc_val))
                ax = abs(clamped)
                q_mag = min(pos_fp4, key=lambda cand: abs(cand - ax))
                q_signed = -q_mag if clamped < 0.0 else q_mag
                out_r.append(float(struct.unpack("<f", struct.pack("<f", q_signed * sc_val))[0]))
        out_rows.append(out_r)
    return out_rows


def compute_yarn_inv_freq(
    rope_dim: int,
    *,
    rope_theta: float = 10000.0,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> list[float]:
    r"""Compute rotary inverse frequencies with optional DeepSeek YaRN wavelength interpolation ramp."""
    if rope_dim <= 0 or rope_dim % 2 != 0:
        raise ValueError("rope_dim must be a positive even integer.")
    half = rope_dim // 2
    freqs = [1.0 / (float(rope_theta) ** (2.0 * i / float(rope_dim))) for i in range(half)]
    if original_seq_len > 0 and factor > 1.0:
        def _corr_dim(rotations: float) -> float:
            return float(rope_dim) * math.log(float(original_seq_len) / (rotations * 2.0 * math.pi)) / (2.0 * math.log(float(rope_theta)))

        low = max(math.floor(_corr_dim(beta_fast)), 0)
        high = min(math.ceil(_corr_dim(beta_slow)), rope_dim - 1)
        denom = max(float(high - low), 1e-3)
        smoothed: list[float] = []
        for i, f in enumerate(freqs):
            ramp = min(1.0, max(0.0, (float(i) - float(low)) / denom))
            smooth = 1.0 - ramp
            smoothed.append((f / float(factor)) * (1.0 - smooth) + f * smooth)
        return smoothed
    return freqs


def apply_interleaved_rope_tail(
    x: Sequence[Any],
    *,
    start_position: int,
    rope_dim: int,
    rope_theta: float = 10000.0,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    inverse: bool = False,
) -> list[Any]:
    r"""Apply interleaved complex rotary embedding to the trailing `rope_dim` elements of 2D `[S, D]` or 3D `[S, H, D]`."""
    if not x:
        return []
    if rope_dim == 0:
        if isinstance(x[0][0], (list, tuple)):
            return [[[float(v) for v in hv] for hv in tok] for tok in x]
        return [[float(v) for v in tok] for tok in x]

    inv_freq = compute_yarn_inv_freq(
        rope_dim,
        rope_theta=rope_theta,
        original_seq_len=original_seq_len,
        factor=factor,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
    )
    sign = -1.0 if inverse else 1.0
    is_3d = isinstance(x[0][0], (list, tuple))

    def _rot_vec(vec: Sequence[float], cos_v: list[float], sin_v: list[float]) -> list[float]:
        d = len(vec)
        if rope_dim > d:
            raise ValueError(f"rope_dim {rope_dim} exceeds vector dim {d}.")
        prefix = [float(v) for v in vec[: d - rope_dim]]
        tail = vec[d - rope_dim :]
        rot_tail: list[float] = []
        for idx, (c, s) in enumerate(zip(cos_v, sin_v)):
            re_v = float(tail[2 * idx])
            im_v = float(tail[2 * idx + 1])
            rot_tail.append(re_v * c - im_v * s)
            rot_tail.append(re_v * s + im_v * c)
        return prefix + rot_tail

    out: list[Any] = []
    for t_rel, item in enumerate(x):
        pos = float(start_position + t_rel)
        cos_v = [math.cos(pos * f) for f in inv_freq]
        sin_v = [sign * math.sin(pos * f) for f in inv_freq]
        if is_3d:
            out.append([_rot_vec(hv, cos_v, sin_v) for hv in item])
        else:
            out.append(_rot_vec(item, cos_v, sin_v))
    return out


def hc_split_sinkhorn(
    mixes_2d: Sequence[Sequence[float]],
    hc_scale: Sequence[float],
    hc_base: Sequence[float],
    *,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    use_numpy: bool | None = None,
) -> tuple[list[list[float]], list[list[float]], list[list[list[float]]]]:
    r"""Manifold-Constrained Hyper-Connections (`mHC`) Birkhoff-polytope Sinkhorn-Knopp projection.

    Given `mixes_2d` $\in \mathbb{R}^{N \times ((2+m)m)}$, projects the $m \times m$ residual mixing
    matrix onto the Birkhoff polytope $\mathcal{B}_m$ of doubly-stochastic matrices ($\rho(C) \le 1$).
    """
    mix_hc = (2 + hc_mult) * hc_mult
    if hc_mult <= 0 or sinkhorn_iters <= 0 or eps < 0.0:
        raise ValueError("Invalid mHC Sinkhorn parameters.")
    if len(hc_scale) != 3 or len(hc_base) != mix_hc:
        raise ValueError("hc_scale must have length 3 and hc_base length (2 + hc_mult) * hc_mult.")
    if not mixes_2d:
        return [], [], []
    if any(len(r) != mix_hc for r in mixes_2d):
        raise ValueError("mixes_2d row width mismatch.")

    want_np = (use_numpy is True or (use_numpy is None and _np is not None)) and _np is not None
    if want_np:
        assert _np is not None
        m64 = _np.asarray(mixes_2d, dtype=_np.float64)
        sc64 = _np.asarray(hc_scale, dtype=_np.float64)
        bs64 = _np.asarray(hc_base, dtype=_np.float64)
        if not (bool(_np.isfinite(m64).all()) and bool(_np.isfinite(sc64).all()) and bool(_np.isfinite(bs64).all())):
            raise ValueError("Non-finite input to hc_split_sinkhorn.")

        def _sig(z: Any) -> Any:
            ez = _np.exp(-_np.abs(z))
            return _np.where(z >= 0.0, 1.0 / (1.0 + ez), ez / (1.0 + ez))

        z_pre = m64[:, :hc_mult] * sc64[0] + bs64[:hc_mult]
        pre = (_sig(z_pre) + float(eps)).astype(_np.float32)

        z_post = m64[:, hc_mult : 2 * hc_mult] * sc64[1] + bs64[hc_mult : 2 * hc_mult]
        post = (2.0 * _sig(z_post)).astype(_np.float32)

        z_comb = (m64[:, 2 * hc_mult :] * sc64[2] + bs64[2 * hc_mult :]).reshape(len(mixes_2d), hc_mult, hc_mult)
        rmax = _np.max(z_comb, axis=-1, keepdims=True)
        ex = _np.exp(z_comb - rmax)
        comb = ex / _np.sum(ex, axis=-1, keepdims=True) + float(eps)
        comb = comb / (_np.sum(comb, axis=-2, keepdims=True) + float(eps))
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (_np.sum(comb, axis=-1, keepdims=True) + float(eps))
            comb = comb / (_np.sum(comb, axis=-2, keepdims=True) + float(eps))
        comb32 = comb.astype(_np.float32)
        return (
            [[float(v) for v in r] for r in pre.tolist()],
            [[float(v) for v in r] for r in post.tolist()],
            [[[float(v) for v in row] for row in mat] for mat in comb32.tolist()],
        )

    s0, s1, s2 = (float(v) for v in hc_scale)
    b_list = [float(v) for v in hc_base]
    if any(not math.isfinite(v) for v in (s0, s1, s2, *b_list)):
        raise ValueError("Non-finite scale or base in hc_split_sinkhorn.")

    def _s1d(z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))

    pre_all: list[list[float]] = []
    post_all: list[list[float]] = []
    comb_all: list[list[list[float]]] = []
    feps = float(eps)
    for row in mixes_2d:
        frow = [float(v) for v in row]
        if any(not math.isfinite(v) for v in frow):
            raise ValueError("Non-finite mixes row in hc_split_sinkhorn.")
        pre_all.append([float(struct.unpack("<f", struct.pack("<f", _s1d(frow[j] * s0 + b_list[j]) + feps))[0]) for j in range(hc_mult)])
        post_all.append([
            float(struct.unpack("<f", struct.pack("<f", 2.0 * _s1d(frow[hc_mult + j] * s1 + b_list[hc_mult + j])))[0])
            for j in range(hc_mult)
        ])
        mat: list[list[float]] = []
        for j in range(hc_mult):
            z_r = [frow[2 * hc_mult + j * hc_mult + k] * s2 + b_list[2 * hc_mult + j * hc_mult + k] for k in range(hc_mult)]
            rm = max(z_r)
            ex = [math.exp(v - rm) for v in z_r]
            rs = math.fsum(ex)
            mat.append([v / rs + feps for v in ex])
        for k in range(hc_mult):
            cs = math.fsum(mat[j][k] for j in range(hc_mult)) + feps
            for j in range(hc_mult):
                mat[j][k] /= cs
        for _ in range(sinkhorn_iters - 1):
            for j in range(hc_mult):
                rs = math.fsum(mat[j]) + feps
                mat[j] = [v / rs for v in mat[j]]
            for k in range(hc_mult):
                cs = math.fsum(mat[j][k] for j in range(hc_mult)) + feps
                for j in range(hc_mult):
                    mat[j][k] /= cs
        comb_all.append([[float(struct.unpack("<f", struct.pack("<f", v))[0]) for v in r_m] for r_m in mat])
    return pre_all, post_all, comb_all


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
    elif dtype in ("F8_E4M3", "F8_E4M3FN"):
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} {dtype} bytes for shape {shape}, got {len(raw)}.")
        arr = _NP_F8_E4M3_LUT[_np.frombuffer(raw, dtype=_np.uint8)].astype(_np.float32)
    elif dtype == "F8_E8M0":
        if len(raw) != expected:
            raise ValueError(f"Expected {expected} F8_E8M0 bytes for shape {shape}, got {len(raw)}.")
        arr = _NP_F8_E8M0_LUT[_np.frombuffer(raw, dtype=_np.uint8)].astype(_np.float32)
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
        x64 = x_nd.astype(_np.float64)
        inv_rms = _np.reciprocal(_np.sqrt(_np.mean(x64 * x64, axis=-1, keepdims=True) + float(eps)))
        return _np.ascontiguousarray((x64 * inv_rms) * weight_nd.astype(_np.float64), dtype=_np.float32)

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
        ez = _np.exp(-_np.abs(g64))
        sig = _np.where(g64 >= 0.0, 1.0 / (1.0 + ez), ez / (1.0 + ez))
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
            xa = _np.asarray(x, dtype=_np.float64)
            wa = _np.asarray(weight, dtype=_np.float64)
            inv_rms = _np.reciprocal(_np.sqrt(_np.mean(xa * xa, axis=-1, keepdims=True) + float(eps)))
            ya = ((xa * inv_rms) * wa).astype(_np.float32)
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
            ga = _np.asarray(gate, dtype=_np.float64)
            ua = _np.asarray(up, dtype=_np.float32)
            ez = _np.exp(-_np.abs(ga))
            sig = _np.where(ga >= 0.0, 1.0 / (1.0 + ez), ez / (1.0 + ez))
            ya = (ga * sig).astype(_np.float32) * ua
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
