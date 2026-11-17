r"""Optional LLVM-compiled Rust CDLL kernel loader (`vld.accel.native`).

Compiles `crates/vld_native/src/lib.rs` via `rustc -O --crate-type cdylib` into a
SHA-256-pinned cache artifact when `rustc` is available on the host, falling back
cleanly (`available = False`) when no Rust toolchain is installed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Sequence

_COMPILE_LOCK = threading.Lock()


def _shared_lib_suffix() -> str:
    if os.name == "nt":
        return ".dll"
    if sys.platform == "darwin":
        return ".dylib"
    return ".so"


def _default_rust_source() -> Path:
    # Locate repo-relative crates/vld_native/src/lib.rs when running from source checkout
    here = Path(__file__).resolve()
    candidate = here.parents[3] / "crates" / "vld_native" / "src" / "lib.rs"
    return candidate


class NativeRustAccelerator:
    """Thread-safe wrapper around optional compiled Rust C-ABI kernels."""

    def __init__(self, lib_path: Path | None = None, source_sha256: str = "", lib_sha256: str = "") -> None:
        self._lib_path = lib_path
        self.source_sha256 = source_sha256
        self.lib_sha256 = lib_sha256
        self._dll: ctypes.CDLL | None = None
        if lib_path is not None and lib_path.is_file():
            dll = ctypes.CDLL(str(lib_path))
            dll.vld_lcg_jump_u64.argtypes = [
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
            ]
            dll.vld_lcg_jump_u64.restype = ctypes.c_uint64

            dll.vld_bilinear_periodic_matvec_f64.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            dll.vld_bilinear_periodic_matvec_f64.restype = ctypes.c_int32

            dll.vld_rms_norm_f32.argtypes = [
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
            ]
            dll.vld_rms_norm_f32.restype = ctypes.c_int32
            self._dll = dll

    @property
    def available(self) -> bool:
        return self._dll is not None

    @classmethod
    def load_or_compile(cls, *, cache_dir: Path | str | None = None) -> NativeRustAccelerator:
        src_path = _default_rust_source()
        rustc_bin = shutil.which("rustc")
        if not src_path.is_file() or rustc_bin is None:
            return cls(None)

        src_bytes = src_path.read_bytes()
        src_sha = hashlib.sha256(src_bytes).hexdigest()
        root = Path(cache_dir) if cache_dir is not None else (Path(tempfile.gettempdir()) / "vld_native_cache")
        suffix = _shared_lib_suffix()
        target_lib = root / f"vld_native_{src_sha[:24]}_{sys.platform}{suffix}"

        with _COMPILE_LOCK:
            if not target_lib.is_file():
                root.mkdir(parents=True, exist_ok=True)
                tmp_lib = root / f"tmp_{os.getpid()}_{src_sha[:16]}{suffix}"
                proc = subprocess.run(
                    [
                        rustc_bin,
                        "--edition",
                        "2021",
                        "--crate-type",
                        "cdylib",
                        "-C",
                        "opt-level=3",
                        str(src_path),
                        "-o",
                        str(tmp_lib),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                if proc.returncode != 0 or not tmp_lib.is_file():
                    return cls(None)
                try:
                    os.replace(tmp_lib, target_lib)
                except OSError:
                    if not target_lib.is_file():
                        return cls(None)

        lib_sha = hashlib.sha256(target_lib.read_bytes()).hexdigest()
        try:
            return cls(target_lib, source_sha256=src_sha, lib_sha256=lib_sha)
        except OSError:
            return cls(None)

    def lcg_jump_u64(self, a: int, c: int, modulus: int, x0: int, steps: int) -> int:
        if self._dll is None:
            raise RuntimeError("NativeRustAccelerator is not available on this host.")
        if not (2 <= modulus < (1 << 64)) or not (0 <= steps < (1 << 64)):
            raise ValueError("Parameters exceed u64 native domain.")
        return int(self._dll.vld_lcg_jump_u64(a % modulus, c % modulus, modulus, x0 % modulus, steps))

    def bilinear_periodic_matvec_f64(
        self,
        *,
        rows: int,
        cols: int,
        alpha: float,
        beta: float,
        gamma: float,
        delta: float,
        orbit: Sequence[float],
        stride: int,
        x: Sequence[float],
    ) -> list[float]:
        if self._dll is None:
            raise RuntimeError("NativeRustAccelerator is not available on this host.")
        period = len(orbit)
        orbit_arr = (ctypes.c_double * max(1, period))(*orbit) if period > 0 else (ctypes.c_double * 1)(0.0)
        x_arr = (ctypes.c_double * max(1, cols))(*x) if cols > 0 else (ctypes.c_double * 1)(0.0)
        out_arr = (ctypes.c_double * max(1, rows))()
        rc = int(
            self._dll.vld_bilinear_periodic_matvec_f64(
                rows,
                cols,
                float(alpha),
                float(beta),
                float(gamma),
                float(delta),
                orbit_arr if period > 0 else None,
                period,
                stride,
                x_arr if cols > 0 else None,
                out_arr if rows > 0 else None,
            )
        )
        if rc != 0:
            raise RuntimeError(f"vld_bilinear_periodic_matvec_f64 failed with code {rc}.")
        return [float(out_arr[i]) for i in range(rows)]

    def rms_norm_f32(self, x_2d: Sequence[Sequence[float]], weight: Sequence[float], eps: float) -> list[list[float]]:
        if self._dll is None:
            raise RuntimeError("NativeRustAccelerator is not available on this host.")
        rows = len(x_2d)
        cols = len(weight)
        flat = [float(val) for row in x_2d for val in row]
        x_arr = (ctypes.c_float * max(1, rows * cols))(*flat)
        w_arr = (ctypes.c_float * cols)(*(float(w) for w in weight))
        out_arr = (ctypes.c_float * max(1, rows * cols))()
        rc = int(self._dll.vld_rms_norm_f32(rows, cols, x_arr, w_arr, float(eps), out_arr))
        if rc != 0:
            raise RuntimeError(f"vld_rms_norm_f32 failed with code {rc}.")
        return [[float(out_arr[r * cols + c]) for c in range(cols)] for r in range(rows)]


_DEFAULT_NATIVE: NativeRustAccelerator | None = None


def get_native_accelerator() -> NativeRustAccelerator:
    global _DEFAULT_NATIVE
    if _DEFAULT_NATIVE is None:
        _DEFAULT_NATIVE = NativeRustAccelerator.load_or_compile()
    return _DEFAULT_NATIVE
