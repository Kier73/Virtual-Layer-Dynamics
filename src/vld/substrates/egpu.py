r"""Phase 5 — SIMT / E-GPU Workgroup Barriers & $O(1)$-Incremental Delta Tensor Checkpoints (`vld.substrates.egpu`, `G23`, `G24`).

Mathematical contract:
1. **SIMT Workgroup Barrier Invariant (`SIMTWorkgroupExecutor`):**
   Across workgroup lanes $\ell \in \{0, \dots, W-1\}$ sharing scratchpad $S$, synchronization instruction
   `BARRIER(id)` requires every active lane to arrive at the **identical** `(pc, barrier_id)` coordinate
   before any lane advances; lane divergence across barriers raises `BarrierDivergenceError`.
2. **$O(1)$-Incremental Append-Only Delta Tensor Checkpoints (`DeltaTensorCheckpointLog`):**
   Eliminates the legacy $O(K^2)$ cumulative prefix rewrite overhead (`PR183`): each completed tile
   $(t_r, t_c)$ of $C = A B$ ($M \times K \times N$) appends a single `fsync`ed hash-chained delta record
   costing $O(B_M B_N)$ bytes independent of total tile count $K_{\text{tiles}}$. Upon `os._exit` recovery,
   the executor verifies the hash chain and per-tile SHA-256 digests (`ForgedTileDigestError`), skips
   verified tiles, and computes exclusively the remaining tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from vld.accel import decode_f32_le, encode_f32_le
from vld.canonical import canonical_bytes, domain_digest, parse_strict_json
from vld.journal import GENESIS_PREV_DIGEST


class EGPUError(ValueError):
    """Base error for SIMT workgroup execution, tiled operators, or delta tensor checkpoints."""


class BarrierDivergenceError(EGPUError):
    """Raised when lanes in a SIMT workgroup diverge across a workgroup `BARRIER`."""


class ForgedTileDigestError(EGPUError):
    """Raised when a persisted tensor tile delta fails SHA-256 or hash-chain verification."""


@dataclass(frozen=True, slots=True)
class SIMTInstruction:
    """Instruction for `SIMTWorkgroupExecutor`."""

    op: str
    args: tuple[Any, ...] = ()


class SIMTWorkgroupExecutor:
    """Deterministic lockstep SIMT workgroup evaluator with shared memory and barrier divergence detection."""

    def __init__(self, workgroup_size: int = 8, shared_elements: int = 64) -> None:
        if workgroup_size <= 0 or shared_elements <= 0:
            raise EGPUError("workgroup_size and shared_elements must be positive.")
        self.workgroup_size = workgroup_size
        self.shared_elements = shared_elements

    def execute_kernel(
        self,
        program: Sequence[SIMTInstruction],
        global_mem: list[float],
        *,
        max_steps: int = 10_000,
    ) -> tuple[list[float], list[float]]:
        """Execute `program` across `workgroup_size` lanes sharing `shared_mem`."""
        w = self.workgroup_size
        shared = [0.0] * self.shared_elements
        out_global = [float(x) for x in global_mem]
        pcs = [0] * w
        halted = [False] * w
        regs: list[dict[str, float]] = [{"lane_id": float(lid), "r0": 0.0, "r1": 0.0, "r2": 0.0, "r3": 0.0} for lid in range(w)]

        steps = 0
        while not all(halted):
            if steps >= max_steps:
                raise EGPUError("SIMT workgroup exceeded max_steps.")
            steps += 1

            # Advance each non-halted lane until it either halts or hits a BARRIER!
            barrier_arrivals: dict[int, tuple[int, str]] = {}
            for lid in range(w):
                while not halted[lid]:
                    pc = pcs[lid]
                    if not (0 <= pc < len(program)):
                        raise EGPUError(f"Lane {lid} PC {pc} out of bounds.")
                    ins = program[pc]
                    op, a = ins.op, ins.args
                    R = regs[lid]

                    if op == "BARRIER":
                        bid = str(a[0]) if a else "default"
                        barrier_arrivals[lid] = (pc, bid)
                        break
                    if op == "HALT":
                        halted[lid] = True
                        break
                    if op == "LOAD_GLOBAL":
                        idx = int(R[str(a[1])]) + (int(a[2]) if len(a) > 2 else 0)
                        R[str(a[0])] = out_global[idx]
                    elif op == "STORE_GLOBAL":
                        idx = int(R[str(a[0])]) + (int(a[2]) if len(a) > 2 else 0)
                        out_global[idx] = R[str(a[1])]
                    elif op == "LOAD_SHARED":
                        idx = int(R[str(a[1])]) + (int(a[2]) if len(a) > 2 else 0)
                        R[str(a[0])] = shared[idx]
                    elif op == "STORE_SHARED":
                        idx = int(R[str(a[0])]) + (int(a[2]) if len(a) > 2 else 0)
                        shared[idx] = R[str(a[1])]
                    elif op == "MOV_IMM":
                        R[str(a[0])] = float(a[1])
                    elif op == "ADD":
                        R[str(a[0])] = R[str(a[1])] + R[str(a[2])]
                    elif op == "MUL":
                        R[str(a[0])] = R[str(a[1])] * R[str(a[2])]
                    elif op == "JZ_EVEN_LANE":
                        if (lid % 2) == 0 and R[str(a[0])] == 0.0:
                            pcs[lid] = int(a[1])
                            continue
                    else:
                        raise EGPUError(f"Unsupported SIMT op {op!r}.")
                    pcs[lid] += 1

            if barrier_arrivals:
                # ALL lanes in the workgroup must arrive at the exact same (pc, barrier_id)!
                if len(barrier_arrivals) != w or len(set(barrier_arrivals.values())) != 1:
                    raise BarrierDivergenceError(
                        f"Workgroup BARRIER divergence detected: arrivals={barrier_arrivals}, halted={halted}."
                    )
                for lid in range(w):
                    pcs[lid] += 1

        return out_global, shared


@dataclass(frozen=True, slots=True)
class TileDeltaRecord:
    """Single $O(1)$-size append-only delta checkpoint entry for one verified output tile."""

    seq: int
    plan_digest: str
    operation_id: str
    tile_row: int
    tile_col: int
    rows: int
    cols: int
    tile_f32_hex: str
    tile_sha256: str
    prev_digest: str
    record_digest: str = ""

    def __post_init__(self) -> None:
        raw = bytes.fromhex(self.tile_f32_hex)
        if hashlib.sha256(raw).hexdigest() != self.tile_sha256:
            raise ForgedTileDigestError(f"Tile ({self.tile_row}, {self.tile_col}) SHA-256 mismatch.")
        if len(raw) != self.rows * self.cols * 4:
            raise ForgedTileDigestError("Tile byte length disagrees with (rows * cols * 4).")
        computed = domain_digest("vld.tile-delta.v1", self._body())
        if self.record_digest and self.record_digest != computed:
            raise ForgedTileDigestError("TileDeltaRecord hash-chain record_digest mismatch.")
        object.__setattr__(self, "record_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "cols": self.cols,
            "operation_id": self.operation_id,
            "plan_digest": self.plan_digest,
            "prev_digest": self.prev_digest,
            "rows": self.rows,
            "schema": "vld.tile-delta.v1",
            "seq": self.seq,
            "tile_col": self.tile_col,
            "tile_f32_hex": self.tile_f32_hex,
            "tile_row": self.tile_row,
            "tile_sha256": self.tile_sha256,
        }

    def decode_tile_2d(self) -> list[list[float]]:
        flat = decode_f32_le(bytes.fromhex(self.tile_f32_hex), use_numpy=False)
        return [flat[r * self.cols : (r + 1) * self.cols] for r in range(self.rows)]

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TileDeltaRecord:
        req = {
            "cols",
            "operation_id",
            "plan_digest",
            "prev_digest",
            "record_digest",
            "rows",
            "schema",
            "seq",
            "tile_col",
            "tile_f32_hex",
            "tile_row",
            "tile_sha256",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.tile-delta.v1":
            raise ForgedTileDigestError("Invalid TileDeltaRecord schema.")
        return cls(
            seq=int(data["seq"]),
            plan_digest=str(data["plan_digest"]),
            operation_id=str(data["operation_id"]),
            tile_row=int(data["tile_row"]),
            tile_col=int(data["tile_col"]),
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            tile_f32_hex=str(data["tile_f32_hex"]),
            tile_sha256=str(data["tile_sha256"]),
            prev_digest=str(data["prev_digest"]),
            record_digest=str(data["record_digest"]),
        )


class DeltaTensorCheckpointLog:
    r"""Append-only $O(1)$-per-tile JSONL delta log eliminating $O(K^2)$ prefix rewrites."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.bytes_appended_history: list[int] = []

    def load_verified_tiles(self, expected_plan_digest: str) -> tuple[dict[tuple[int, int], TileDeltaRecord], str, int]:
        """Verify entire delta chain and return `(tiles_by_coord, head_digest, next_seq)`."""
        if not self.path.exists():
            return {}, GENESIS_PREV_DIGEST, 0

        raw_text = self.path.read_bytes().decode("utf-8")
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        tiles: dict[tuple[int, int], TileDeltaRecord] = {}
        prev = GENESIS_PREV_DIGEST
        for idx, ln in enumerate(lines):
            try:
                parsed = parse_strict_json(ln)
            except Exception as exc:
                raise ForgedTileDigestError(f"Corrupted JSONL line {idx} in delta checkpoint: {exc}") from exc
            rec = TileDeltaRecord.from_dict(parsed)
            if rec.seq != idx or rec.prev_digest != prev:
                raise ForgedTileDigestError(f"Broken hash chain at tile seq {idx}.")
            if rec.plan_digest != expected_plan_digest:
                raise ForgedTileDigestError("Delta checkpoint plan_digest disagrees with active tensor plan.")
            tiles[(rec.tile_row, rec.tile_col)] = rec
            prev = rec.record_digest
        return tiles, prev, len(lines)

    def append_tile(
        self,
        *,
        seq: int,
        prev_digest: str,
        plan_digest: str,
        operation_id: str,
        tile_row: int,
        tile_col: int,
        tile_values: Sequence[Sequence[float]],
    ) -> TileDeltaRecord:
        rows = len(tile_values)
        cols = len(tile_values[0]) if rows > 0 else 0
        flat = [float(v) for row in tile_values for v in row]
        raw_f32 = encode_f32_le(flat)
        rec = TileDeltaRecord(
            seq=seq,
            plan_digest=plan_digest,
            operation_id=operation_id,
            tile_row=tile_row,
            tile_col=tile_col,
            rows=rows,
            cols=cols,
            tile_f32_hex=raw_f32.hex(),
            tile_sha256=hashlib.sha256(raw_f32).hexdigest(),
            prev_digest=prev_digest,
        )
        line_bytes = canonical_bytes(rec.to_dict()) + b"\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as fh:
            fh.write(line_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        self.bytes_appended_history.append(len(line_bytes))
        return rec


class TiledTensorExecutor:
    """Tiled matrix multiplication and 2D convolution operator with $O(1)$-incremental crash recovery."""

    def __init__(self, tile_m: int = 4, tile_n: int = 4) -> None:
        if tile_m <= 0 or tile_n <= 0:
            raise EGPUError("tile_m and tile_n must be positive.")
        self.tile_m = tile_m
        self.tile_n = tile_n

    def matmul_tiled_recoverable(
        self,
        mat_a: Sequence[Sequence[float]],
        mat_b: Sequence[Sequence[float]],
        *,
        checkpoint_log: DeltaTensorCheckpointLog | None = None,
        receiver_op_prefix: str = "egpu.matmul",
        _fault_after_tiles: int | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Compute $C = A B$ in tiles, checkpointing $O(1)$ deltas per tile and skipping verified tiles on resume."""
        m = len(mat_a)
        k = len(mat_a[0]) if m > 0 else 0
        if m == 0 or k == 0 or len(mat_b) != k or len(mat_b[0]) == 0:
            raise EGPUError("Incompatible matrix dimensions for matmul.")
        n = len(mat_b[0])

        a_sha = hashlib.sha256(encode_f32_le([v for r in mat_a for v in r])).hexdigest()
        b_sha = hashlib.sha256(encode_f32_le([v for r in mat_b for v in r])).hexdigest()
        plan_digest = domain_digest(
            "vld.matmul-plan.v1",
            {"a_sha": a_sha, "b_sha": b_sha, "dims": [m, k, n], "tiles": [self.tile_m, self.tile_n]},
        )

        verified_tiles: dict[tuple[int, int], TileDeltaRecord] = {}
        head = GENESIS_PREV_DIGEST
        seq = 0
        if checkpoint_log is not None:
            verified_tiles, head, seq = checkpoint_log.load_verified_tiles(plan_digest)

        out = [[0.0] * n for _ in range(m)]
        newly_computed = 0
        reused_tiles = 0

        num_tr = (m + self.tile_m - 1) // self.tile_m
        num_tc = (n + self.tile_n - 1) // self.tile_n

        for tr in range(num_tr):
            r0 = tr * self.tile_m
            r1 = min(m, r0 + self.tile_m)
            for tc in range(num_tc):
                c0 = tc * self.tile_n
                c1 = min(n, c0 + self.tile_n)

                if (tr, tc) in verified_tiles:
                    saved_2d = verified_tiles[(tr, tc)].decode_tile_2d()
                    for dr, row_vals in enumerate(saved_2d):
                        out[r0 + dr][c0:c1] = row_vals
                    reused_tiles += 1
                    continue

                tile_2d: list[list[float]] = []
                for r in range(r0, r1):
                    row_out: list[float] = []
                    for c in range(c0, c1):
                        val = math.fsum(float(mat_a[r][p]) * float(mat_b[p][c]) for p in range(k))
                        # Round to binary32 so saved F32 tile & live output agree identically!
                        val_f32 = decode_f32_le(encode_f32_le([val]), use_numpy=False)[0]
                        row_out.append(val_f32)
                    tile_2d.append(row_out)
                    out[r][c0:c1] = row_out

                if checkpoint_log is not None:
                    rec = checkpoint_log.append_tile(
                        seq=seq,
                        prev_digest=head,
                        plan_digest=plan_digest,
                        operation_id=f"{receiver_op_prefix}:{tr}:{tc}",
                        tile_row=tr,
                        tile_col=tc,
                        tile_values=tile_2d,
                    )
                    head = rec.record_digest
                    seq += 1

                newly_computed += 1
                if _fault_after_tiles is not None and newly_computed >= _fault_after_tiles:
                    os._exit(97)

        telemetry = {
            "newly_computed_tiles": newly_computed,
            "plan_digest": plan_digest,
            "reused_verified_tiles": reused_tiles,
            "total_tiles": num_tr * num_tc,
        }
        return out, telemetry

    @staticmethod
    def conv2d_valid(
        image_2d: Sequence[Sequence[float]],
        kernel_2d: Sequence[Sequence[float]],
    ) -> list[list[float]]:
        """Deterministic 2D valid spatial convolution operator."""
        h = len(image_2d)
        w = len(image_2d[0]) if h > 0 else 0
        kh = len(kernel_2d)
        kw = len(kernel_2d[0]) if kh > 0 else 0
        if kh == 0 or kw == 0 or kh > h or kw > w:
            raise EGPUError("Invalid 2D convolution dimensions.")
        out_h = h - kh + 1
        out_w = w - kw + 1
        return [
            [
                math.fsum(
                    float(image_2d[r + kr][c + kc]) * float(kernel_2d[kr][kc])
                    for kr in range(kh)
                    for kc in range(kw)
                )
                for c in range(out_w)
            ]
            for r in range(out_h)
        ]


__all__ = [
    "BarrierDivergenceError",
    "DeltaTensorCheckpointLog",
    "EGPUError",
    "ForgedTileDigestError",
    "SIMTInstruction",
    "SIMTWorkgroupExecutor",
    "TileDeltaRecord",
    "TiledTensorExecutor",
]
