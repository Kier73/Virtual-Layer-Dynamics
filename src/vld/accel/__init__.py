r"""Multi-Tier Numeric, Native Rust CDLL, and Structural Acceleration (`vld.accel`)."""

from __future__ import annotations

from vld.accel.kernels import (
    FP4_E2M1_TABLE,
    NumericEngine,
    apply_interleaved_rope_tail,
    compute_yarn_inv_freq,
    decode_bf16_le,
    decode_f8_e4m3,
    decode_f8_e8m0,
    decode_f16_le,
    decode_f32_le,
    decode_tensor_ndarray,
    dequantize_fp4_e2m1_packed,
    dequantize_fp8_blockwise,
    encode_bf16_le,
    encode_f8_e4m3,
    encode_f8_e8m0,
    encode_f32_le,
    encode_fp4_e2m1_packed,
    has_numpy,
    hc_split_sinkhorn,
    quantize_dequantize_fp4_rows,
)
from vld.accel.native import NativeRustAccelerator, get_native_accelerator
from vld.accel.structural_ops import (
    LinearRecurrenceGround,
    StructuralLinearOperator,
    benchmark_virtual_vs_traditional,
)

__all__ = [
    "FP4_E2M1_TABLE",
    "LinearRecurrenceGround",
    "NativeRustAccelerator",
    "NumericEngine",
    "StructuralLinearOperator",
    "apply_interleaved_rope_tail",
    "benchmark_virtual_vs_traditional",
    "compute_yarn_inv_freq",
    "decode_bf16_le",
    "decode_f8_e4m3",
    "decode_f8_e8m0",
    "decode_f16_le",
    "decode_f32_le",
    "decode_tensor_ndarray",
    "dequantize_fp4_e2m1_packed",
    "dequantize_fp8_blockwise",
    "encode_bf16_le",
    "encode_f8_e4m3",
    "encode_f8_e8m0",
    "encode_f32_le",
    "encode_fp4_e2m1_packed",
    "get_native_accelerator",
    "has_numpy",
    "hc_split_sinkhorn",
    "quantize_dequantize_fp4_rows",
]
