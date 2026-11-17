r"""Multi-Tier Numeric, Native Rust CDLL, and Structural Acceleration (`vld.accel`)."""

from __future__ import annotations

from vld.accel.kernels import (
    NumericEngine,
    decode_bf16_le,
    decode_f16_le,
    decode_f32_le,
    encode_bf16_le,
    encode_f32_le,
    has_numpy,
)
from vld.accel.native import NativeRustAccelerator, get_native_accelerator
from vld.accel.structural_ops import (
    LinearRecurrenceGround,
    StructuralLinearOperator,
    benchmark_virtual_vs_traditional,
)

__all__ = [
    "LinearRecurrenceGround",
    "NativeRustAccelerator",
    "NumericEngine",
    "StructuralLinearOperator",
    "benchmark_virtual_vs_traditional",
    "decode_bf16_le",
    "decode_f16_le",
    "decode_f32_le",
    "encode_bf16_le",
    "encode_f32_le",
    "get_native_accelerator",
    "has_numpy",
]
