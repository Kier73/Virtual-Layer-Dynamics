r"""Phase 5 — SIMT / E-GPU Workgroup Barriers & $O(1)$-Incremental Delta Tensor Checkpoints (`vld.substrates.egpu`, `G09`, `G10`).

Mathematical contract:
1. **SIMT Workgroup Barrier & Shared-Memory Publication Invariant (`SIMTWorkgroupExecutor`):**
   Across workgroup lanes $\ell \in \{0, \dots, W-1\}$ sharing scratchpad $S$:
   - Lanes step in bounded lockstep cycles; writes to shared memory $S$ within an epoch are staged per lane
     and published across the workgroup **only** at `BARRIER(id)` (eliminating sequential lane-order races
     while trapping unordered write-write races).
   - Synchronization instruction `BARRIER(id)` requires every active lane to arrive at the **identical**
     `(pc, barrier_id)` coordinate before any lane advances; lane divergence across barriers raises
     `BarrierDivergenceError`.
2. **$O(1)$-Incremental Append-Only Delta Tensor Checkpoints (`DeltaTensorCheckpointLog`):**
   Eliminates the legacy $O(K^2)$ cumulative prefix rewrite overhead (`PR183`): each completed tile
   $(t_r, t_c)$ of $C = A B$ ($M \times K \times N$) appends a single `fsync`ed hash-chained delta record
   costing $O(B_M B_N)$ bytes independent of total tile count $K_{\text{tiles}}$. Upon `os._exit` recovery,
   the executor verifies the hash chain, per-tile SHA-256 digests, and independent arithmetic projections
   (`ForgedTileDigestError`), skips verified tiles with zero worker re-dispatch, and computes exclusively
   the remaining suffix tiles.
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
    """Raised when a persisted or returned tensor tile fails SHA-256, hash-chain, or arithmetic verification."""


@dataclass(frozen=True, slots=True)
class SIMTInstruction:
    """Instruction for `SIMTWorkgroupExecutor`."""

    op: str
    args: tuple[Any, ...] = ()


class SIMTWorkgroupExecutor:
    """Deterministic lockstep SIMT workgroup evaluator with barrier-published shared memory and divergence detection."""

    def __init__(self, workgroup_size: int = 8, shared_elements: int = 64) -> None:
        if workgroup_size <= 0 or shared_elements <= 0:
            raise EGPUError("workgroup_size and shared_elements must be positive.")
        self.workgroup_size = workgroup_size
        self.shared_elements = shared_elements

    def execute_kernel(
        self,
        program: Sequence[SIMTInstruction],
        global_mem: Sequence[float],
        *,
        max_steps: int = 10_000,
    ) -> tuple[list[float], list[float]]:
        """Execute `program` across `workgroup_size` lanes under barrier-published shared-memory semantics."""
        if max_steps <= 0:
            raise EGPUError("max_steps must be positive.")
        w = self.workgroup_size
        shared = [0.0] * self.shared_elements
        out_global = [float(x) for x in global_mem]
        pcs = [0] * w
        halted = [False] * w
        barrier_arrivals: dict[int, tuple[int, str]] = {}
        pending_shared: list[dict[int, float]] = [{} for _ in range(w)]
        regs: list[dict[str, float]] = [
            {"lane_id": float(lid), "r0": 0.0, "r1": 0.0, "r2": 0.0, "r3": 0.0}
            for lid in range(w)
        ]

        def _publish_pending_shared() -> None:
            writer_by_idx: dict[int, int] = {}
            for lid in range(w):
                for idx, val in pending_shared[lid].items():
                    if idx in writer_by_idx:
                        raise EGPUError(
                            f"Unordered cross-lane shared memory write race on shared[{idx}] "
                            f"between lanes {writer_by_idx[idx]} and {lid}."
                        )
                    writer_by_idx[idx] = lid
                    shared[idx] = val
                pending_shared[lid].clear()

        steps = 0
        while not all(halted):
            if steps >= max_steps:
                raise EGPUError("SIMT workgroup exceeded max_steps.")
            steps += 1

            progressed = False
            for lid in range(w):
                if halted[lid] or lid in barrier_arrivals:
                    continue
                progressed = True
                pc = pcs[lid]
                if not (0 <= pc < len(program)):
                    raise EGPUError(f"Lane {lid} PC {pc} out of bounds.")
                ins = program[pc]
                op, a = ins.op, ins.args
                R = regs[lid]

                if op == "BARRIER":
                    bid = str(a[0]) if a else "default"
                    barrier_arrivals[lid] = (pc, bid)
                    continue
                if op == "HALT":
                    halted[lid] = True
                    continue
                if op == "LOAD_GLOBAL":
                    idx = int(R[str(a[1])]) + (int(a[2]) if len(a) > 2 else 0)
                    if not (0 <= idx < len(out_global)):
                        raise EGPUError(f"Lane {lid} out-of-bounds LOAD_GLOBAL index {idx}.")
                    R[str(a[0])] = out_global[idx]
                elif op == "STORE_GLOBAL":
                    idx = int(R[str(a[0])]) + (int(a[2]) if len(a) > 2 else 0)
                    if not (0 <= idx < len(out_global)):
                        raise EGPUError(f"Lane {lid} out-of-bounds STORE_GLOBAL index {idx}.")
                    out_global[idx] = R[str(a[1])]
                elif op == "LOAD_SHARED":
                    idx = int(R[str(a[1])]) + (int(a[2]) if len(a) > 2 else 0)
                    if not (0 <= idx < self.shared_elements):
                        raise EGPUError(f"Lane {lid} out-of-bounds LOAD_SHARED index {idx}.")
                    # Lane sees its own staged write or the last barrier-published value!
                    R[str(a[0])] = pending_shared[lid].get(idx, shared[idx])
                elif op == "STORE_SHARED":
                    idx = int(R[str(a[0])]) + (int(a[2]) if len(a) > 2 else 0)
                    if not (0 <= idx < self.shared_elements):
                        raise EGPUError(f"Lane {lid} out-of-bounds STORE_SHARED index {idx}.")
                    pending_shared[lid][idx] = R[str(a[1])]
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

            # Check if all non-halted lanes have parked at a barrier (or halted)!
            if barrier_arrivals and all(halted[lid] or (lid in barrier_arrivals) for lid in range(w)):
                if len(barrier_arrivals) != w or len(set(barrier_arrivals.values())) != 1:
                    raise BarrierDivergenceError(
                        f"Workgroup BARRIER divergence detected: arrivals={barrier_arrivals}, halted={halted}."
                    )
                _publish_pending_shared()
                barrier_arrivals.clear()
                for lid in range(w):
                    pcs[lid] += 1
            elif not progressed and barrier_arrivals:
                raise BarrierDivergenceError(
                    f"Workgroup BARRIER divergence detected: arrivals={barrier_arrivals}, halted={halted}."
                )

        _publish_pending_shared()
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
        if self.rows <= 0 or self.cols <= 0 or self.tile_row < 0 or self.tile_col < 0:
            raise ForgedTileDigestError("Invalid TileDeltaRecord dimensions or coordinates.")
        try:
            raw = bytes.fromhex(self.tile_f32_hex)
        except ValueError as exc:
            raise ForgedTileDigestError(f"Malformed tile hex: {exc}") from exc
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
        if any(not math.isfinite(v) for v in flat):
            raise ForgedTileDigestError("Decoded tile contains non-finite binary32 values.")
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
            coord = (rec.tile_row, rec.tile_col)
            if coord in tiles:
                raise ForgedTileDigestError(f"Duplicate tile delta entry for coordinate {coord}.")
            tiles[coord] = rec
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


def _verify_tile_arithmetic_freivalds(
    a_rows: tuple[tuple[float, ...], ...],
    b_cols_k_by_n: tuple[tuple[float, ...], ...],
    c_tile: Sequence[Sequence[float]],
    *,
    plan_digest: str,
    tr: int,
    tc: int,
) -> None:
    r"""Verify $A_I (B_J v) \approx C_{I,J} v$ in $O((b_m + b_n) K)$ to reject rehashed false arithmetic."""
    bm = len(a_rows)
    k_dim = len(a_rows[0])
    bn = len(b_cols_k_by_n[0])
    if len(c_tile) != bm or any(len(row) != bn for row in c_tile):
        raise ForgedTileDigestError(f"Tile ({tr}, {tc}) shape mismatch against plan.")

    # Build probe vectors: exact standard basis when bn <= 8, else 6 deterministic Rademacher probes + all-ones!
    probes: list[list[float]] = []
    if bn <= 8:
        for j in range(bn):
            vec = [0.0] * bn
            vec[j] = 1.0
            probes.append(vec)
    else:
        probes.append([1.0] * bn)
        for s_idx in range(6):
            seed = hashlib.sha256(f"{plan_digest}:{tr}:{tc}:{s_idx}".encode("utf-8")).digest()
            probes.append([1.0 if (seed[j % 32] & (1 << (j % 8))) else -1.0 for j in range(bn)])

    abs_a_row = [math.fsum(abs(x) for x in r) for r in a_rows]
    max_b = max((abs(v) for row in b_cols_k_by_n for v in row), default=0.0)

    for v in probes:
        # w = B_J @ v  (length K)
        w = [math.fsum(b_cols_k_by_n[p][j] * v[j] for j in range(bn)) for p in range(k_dim)]
        for i in range(bm):
            lhs = math.fsum(a_rows[i][p] * w[p] for p in range(k_dim))
            rhs = math.fsum(float(c_tile[i][j]) * v[j] for j in range(bn))
            if not math.isfinite(rhs):
                raise ForgedTileDigestError(f"Non-finite entry in tile ({tr}, {tc}).")
            scale = abs_a_row[i] * max_b * max(1, bn)
            tol = max(1e-4, 16.0 * (2.0**-23) * scale)
            if abs(lhs - rhs) > tol:
                raise ForgedTileDigestError(
                    f"Tile ({tr}, {tc}) failed arithmetic oracle check: |{lhs} - {rhs}| > {tol}."
                )


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
        tile_compute_fn: Callable[[int, int, list[list[float]], list[list[float]]], Sequence[Sequence[float]]] | None = None,
        _fault_after_tiles: int | None = None,
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Compute $C = A B$ in tiles, checkpointing $O(1)$ deltas per tile and skipping verified tiles on resume."""
        m = len(mat_a)
        k = len(mat_a[0]) if m > 0 else 0
        if m == 0 or k == 0 or any(len(r) != k for r in mat_a) or len(mat_b) != k or len(mat_b[0]) == 0:
            raise EGPUError("Incompatible or jagged matrix dimensions for matmul.")
        n = len(mat_b[0])
        if any(len(r) != n for r in mat_b):
            raise EGPUError("Jagged matrix B dimensions for matmul.")

        # Immutable defensive snapshot bound to the original intention (defeating callback operand mutation)!
        a_snap = tuple(tuple(float(v) for v in r) for r in mat_a)
        b_snap = tuple(tuple(float(v) for v in r) for r in mat_b)
        if any(not math.isfinite(v) for r in a_snap for v in r) or any(not math.isfinite(v) for r in b_snap for v in r):
            raise EGPUError("Non-finite float in matmul operands.")

        a_sha = hashlib.sha256(encode_f32_le([v for r in a_snap for v in r])).hexdigest()
        b_sha = hashlib.sha256(encode_f32_le([v for r in b_snap for v in r])).hexdigest()
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
        for (vr, vc) in verified_tiles:
            if not (0 <= vr < num_tr and 0 <= vc < num_tc):
                raise ForgedTileDigestError(f"Out-of-bounds tile coordinate {(vr, vc)} in delta log.")

        for tr in range(num_tr):
            r0 = tr * self.tile_m
            r1 = min(m, r0 + self.tile_m)
            a_sub = a_snap[r0:r1]
            for tc in range(num_tc):
                c0 = tc * self.tile_n
                c1 = min(n, c0 + self.tile_n)
                b_sub = tuple(row[c0:c1] for row in b_snap)

                if (tr, tc) in verified_tiles:
                    saved_2d = verified_tiles[(tr, tc)].decode_tile_2d()
                    _verify_tile_arithmetic_freivalds(a_sub, b_sub, saved_2d, plan_digest=plan_digest, tr=tr, tc=tc)
                    for dr, row_vals in enumerate(saved_2d):
                        out[r0 + dr][c0:c1] = row_vals
                    reused_tiles += 1
                    continue

                if tile_compute_fn is not None:
                    # Pass mutable copies so malicious callback mutations cannot taint `a_snap` or `b_snap`!
                    cand_2d = tile_compute_fn(
                        tr,
                        tc,
                        [list(r) for r in a_sub],
                        [list(r) for r in b_sub],
                    )
                    tile_2d = [
                        [decode_f32_le(encode_f32_le([float(v)]), use_numpy=False)[0] for v in row]
                        for row in cand_2d
                    ]
                    _verify_tile_arithmetic_freivalds(a_sub, b_sub, tile_2d, plan_digest=plan_digest, tr=tr, tc=tc)
                    for dr, row_vals in enumerate(tile_2d):
                        out[r0 + dr][c0:c1] = list(row_vals)
                else:
                    tile_2d = []
                    for r in range(r0, r1):
                        row_out: list[float] = []
                        for c in range(c0, c1):
                            val = math.fsum(a_snap[r][p] * b_snap[p][c] for p in range(k))
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
        if kh == 0 or kw == 0 or kh > h or kw > w or any(len(r) != w for r in image_2d) or any(len(r) != kw for r in kernel_2d):
            raise EGPUError("Invalid or jagged 2D convolution dimensions.")
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
