r"""Algebraic Operator Contraction & Trajectory Semigroup Leaping (`vld.accel.structural_ops`).

Changing the shape of computation:
1. **Sufficient-Statistic Operator Contraction (`StructuralLinearOperator`):**
   Traditional execution of $y = W x$ ($W \in \mathbb{R}^{M \times N}$) materializes $M N$ floats ($\Theta(MN)$ bytes)
   and performs $2 M N$ FLOPs. When $W$ is lifted into its Virtual Layer decomposition:
     $$W_{i,j} = (\alpha + \beta i + \gamma j + \delta i j) + c_{(j - s \cdot i) \bmod P} + \sum_{r=0}^{R-1} U_{i,r} V_{j,r} + \Delta_{i,j}$$
   contraction proceeds via input sufficient statistics $(S_0, S_1, \tilde{x}_0 \dots \tilde{x}_{P-1}, z_0 \dots z_{R-1})$ in
     $$O\!\left(M + N + M P + R(M + N) + |\operatorname{supp}(\Delta)|\right) \ll \Theta(MN)$$
   FLOPs and $O(P + R(M + N) + |\operatorname{supp}(\Delta)|)$ bytes—with zero dense matrix allocation!

2. **Block-Monoid Trajectory Leaping (`LinearRecurrenceGround`):**
   Advances contractive/orthogonal affine state-space dynamics $h_{t+1} = A h_t + b \in \mathbb{R}^d$ across $T$ steps via:
     $$\begin{pmatrix} h_T \\ 1 \end{pmatrix} = \begin{pmatrix} A & b \\ 0^\top & 1 \end{pmatrix}^T \begin{pmatrix} h_0 \\ 1 \end{pmatrix}$$
   reducing $\Theta(T d^2)$ sequential iterations to $O(d^3 \log_2 T)$ operations.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import statistics
import struct
import time
from typing import Any, Sequence

from vld.accel.kernels import has_numpy
from vld.accel.native import get_native_accelerator
from vld.canonical import domain_digest
from vld.ground import NavigationCostVector


def _spectral_norm_f64(mat: tuple[tuple[float, ...], ...]) -> float:
    r"""Compute exact $\ell_2$ operator norm $\|A\|_2 = \sqrt{\lambda_{\max}(A^\top A)}$ via symmetric Jacobi rotations."""
    d = len(mat)
    g = [
        [math.fsum(float(mat[k][i]) * float(mat[k][j]) for k in range(d)) for j in range(d)]
        for i in range(d)
    ]
    max_sweeps = max(16, 6 * d * d)
    for _ in range(max_sweeps):
        p, q = 0, 1
        best = 0.0
        for i in range(d):
            for j in range(i + 1, d):
                val = abs(g[i][j])
                if val > best:
                    best = val
                    p, q = i, j
        if best <= 1e-14:
            break
        diff = g[q][q] - g[p][p]
        if abs(diff) <= 1e-30:
            t = 1.0
        else:
            tau = diff / (2.0 * g[p][q])
            t = math.copysign(1.0 / (abs(tau) + math.hypot(1.0, tau)), tau)
        c = 1.0 / math.hypot(1.0, t)
        s = t * c
        g_pp, g_qq, g_pq = g[p][p], g[q][q], g[p][q]
        g[p][p] = c * c * g_pp - 2.0 * s * c * g_pq + s * s * g_qq
        g[q][q] = s * s * g_pp + 2.0 * s * c * g_pq + c * c * g_qq
        g[p][q] = 0.0
        g[q][p] = 0.0
        for r in range(d):
            if r != p and r != q:
                grp, grq = g[r][p], g[r][q]
                g[r][p] = g[p][r] = c * grp - s * grq
                g[r][q] = g[q][r] = s * grp + c * grq
    max_eig = max((g[i][i] for i in range(d)), default=0.0)
    return math.sqrt(max(0.0, max_eig))


@dataclass(frozen=True, slots=True)
class StructuralLinearOperator:
    r"""Generative linear operator $W \in \mathbb{R}^{M \times N}$ contracted via Virtual Layer sufficient statistics."""

    rows: int
    cols: int
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    periodic_orbit: tuple[float, ...] = ()
    orbit_stride: int = 1
    low_rank_u: tuple[tuple[float, ...], ...] = ()  # shape (rows, R)
    low_rank_v: tuple[tuple[float, ...], ...] = ()  # shape (cols, R)
    sparse_entries: tuple[tuple[int, int, float], ...] = ()

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows and cols must be positive integers.")
        if self.orbit_stride < 0:
            raise ValueError("orbit_stride must be non-negative.")
        for scalar in (*((self.alpha, self.beta, self.gamma, self.delta)), *self.periodic_orbit):
            if not math.isfinite(float(scalar)):
                raise ValueError("Bilinear and periodic law coefficients must be finite.")
        if self.low_rank_u or self.low_rank_v:
            if len(self.low_rank_u) != self.rows or len(self.low_rank_v) != self.cols:
                raise ValueError("low_rank_u/v outer dimensions must match (rows, cols).")
            rank = len(self.low_rank_u[0])
            if any(len(r) != rank for r in self.low_rank_u) or any(len(c) != rank for c in self.low_rank_v):
                raise ValueError("Inconsistent low-rank inner dimension R.")
            if any(not math.isfinite(float(v)) for row in self.low_rank_u for v in row) or any(
                not math.isfinite(float(v)) for row in self.low_rank_v for v in row
            ):
                raise ValueError("Low-rank factor entries must be finite.")
        seen: set[tuple[int, int]] = set()
        for r_idx, c_idx, val in self.sparse_entries:
            if not (0 <= r_idx < self.rows and 0 <= c_idx < self.cols):
                raise ValueError(f"Sparse coordinate {(r_idx, c_idx)} outside ({self.rows}, {self.cols}).")
            if (r_idx, c_idx) in seen:
                raise ValueError(f"Duplicate sparse coordinate {(r_idx, c_idx)}; consolidate deltas first.")
            if not math.isfinite(float(val)):
                raise ValueError("Sparse delta must be finite.")
            seen.add((r_idx, c_idx))

    @property
    def rank(self) -> int:
        return len(self.low_rank_u[0]) if self.low_rank_u else 0

    @property
    def dense_bytes_f64(self) -> int:
        return self.rows * self.cols * 8

    @property
    def structural_bytes_f64(self) -> int:
        base_scalars = 4 * 8
        orbit_bytes = len(self.periodic_orbit) * 8
        lr_bytes = (self.rows + self.cols) * self.rank * 8
        sparse_bytes = len(self.sparse_entries) * 24
        return base_scalars + orbit_bytes + lr_bytes + sparse_bytes

    @property
    def structural_flops(self) -> int:
        period = len(self.periodic_orbit)
        r = self.rank
        # S0/S1/fold pass + row synthesis + rank-R projections + sparse nnz
        return (
            3 * self.cols
            + self.rows * (6 + 2 * period)
            + 2 * r * (self.cols + self.rows)
            + 2 * len(self.sparse_entries)
        )

    @property
    def dense_flops(self) -> int:
        return 2 * self.rows * self.cols

    @property
    def fingerprint(self) -> str:
        u_flat = [float(v) for row in self.low_rank_u for v in row]
        v_flat = [float(v) for row in self.low_rank_v for v in row]
        u_sha = hashlib.sha256(struct.pack(f"<{len(u_flat)}d", *u_flat)).hexdigest()
        v_sha = hashlib.sha256(struct.pack(f"<{len(v_flat)}d", *v_flat)).hexdigest()
        sorted_sparse = [[r_i, c_i, float(dv)] for r_i, c_i, dv in sorted(self.sparse_entries)]
        return domain_digest(
            "vld.structural-linear-op.v1",
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "cols": self.cols,
                "delta": self.delta,
                "gamma": self.gamma,
                "low_rank_u_sha256": u_sha,
                "low_rank_v_sha256": v_sha,
                "orbit_stride": self.orbit_stride,
                "periodic_orbit": list(self.periodic_orbit),
                "rank": self.rank,
                "rows": self.rows,
                "sparse_entries": sorted_sparse,
            },
        )

    def entry(self, i: int, j: int) -> float:
        """Evaluate single matrix entry $W_{i,j}$ on demand in $O(R)$ without dense storage."""
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise ValueError(f"Coordinate {(i, j)} outside ({self.rows}, {self.cols}).")
        val = self.alpha + self.beta * i + self.gamma * j + self.delta * (i * j)
        if self.periodic_orbit:
            p = len(self.periodic_orbit)
            val += self.periodic_orbit[(j - self.orbit_stride * i) % p]
        if self.rank > 0:
            val += math.fsum(u * v for u, v in zip(self.low_rank_u[i], self.low_rank_v[j]))
        for r_idx, c_idx, dv in self.sparse_entries:
            if r_idx == i and c_idx == j:
                val += dv
                break
        return val

    def materialize_dense(self) -> list[list[float]]:
        r"""Materialize explicit $\Theta(MN)$ dense matrix (for traditional comparison / small verification)."""
        p = len(self.periodic_orbit)
        dense = [[0.0] * self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            row = dense[i]
            a_i = self.alpha + self.beta * i
            b_i = self.gamma + self.delta * i
            u_i = self.low_rank_u[i] if self.rank > 0 else ()
            for j in range(self.cols):
                v = a_i + b_i * j
                if p > 0:
                    v += self.periodic_orbit[(j - self.orbit_stride * i) % p]
                if u_i:
                    v += math.fsum(uk * vk for uk, vk in zip(u_i, self.low_rank_v[j]))
                row[j] = v
        for r_idx, c_idx, dv in self.sparse_entries:
            dense[r_idx][c_idx] += dv
        return dense

    def contract(self, x: Sequence[float], *, allow_native: bool = True) -> list[float]:
        """Evaluate $y = W x$ via Virtual Layer sufficient statistics without materializing $W$."""
        if len(x) != self.cols:
            raise ValueError(f"Input vector length {len(x)} != operator cols {self.cols}.")
        if has_numpy():
            import numpy as np

            xa = np.asarray(x, dtype=np.float64)
            if not bool(np.isfinite(xa).all()):
                raise ValueError("Input vector contains non-finite values.")
            native = get_native_accelerator() if allow_native else None
            if native is not None and native.available:
                out_arr = np.asarray(
                    native.bilinear_periodic_matvec_f64(
                        rows=self.rows,
                        cols=self.cols,
                        alpha=self.alpha,
                        beta=self.beta,
                        gamma=self.gamma,
                        delta=self.delta,
                        orbit=self.periodic_orbit,
                        stride=self.orbit_stride,
                        x=xa.tolist(),
                    ),
                    dtype=np.float64,
                )
            else:
                p = len(self.periodic_orbit)
                s0 = float(np.sum(xa))
                jj = np.arange(self.cols, dtype=np.float64)
                s1 = float(np.dot(jj, xa))
                ii = np.arange(self.rows, dtype=np.float64)
                out_arr = (self.alpha + self.beta * ii) * s0 + (self.gamma + self.delta * ii) * s1
                if p > 0:
                    folded = np.bincount(np.arange(self.cols) % p, weights=xa, minlength=p)
                    shifts = (self.orbit_stride * np.arange(self.rows, dtype=np.int64)) % p
                    orb_arr = np.asarray(self.periodic_orbit, dtype=np.float64)
                    idx = (np.arange(p, dtype=np.int64)[None, :] - shifts[:, None]) % p
                    out_arr = out_arr + (orb_arr[idx] @ folded)
            if self.rank > 0:
                u_arr = np.asarray(self.low_rank_u, dtype=np.float64)
                v_arr = np.asarray(self.low_rank_v, dtype=np.float64)
                out_arr = out_arr + (u_arr @ (v_arr.T @ xa))
            for r_idx, c_idx, dv in self.sparse_entries:
                out_arr[r_idx] += dv * xa[c_idx]
            return [float(v) for v in out_arr.tolist()]

        xf = [float(v) for v in x]
        if any(not math.isfinite(v) for v in xf):
            raise ValueError("Input vector contains non-finite values.")

        native = get_native_accelerator() if allow_native else None
        if native is not None and native.available:
            out = native.bilinear_periodic_matvec_f64(
                rows=self.rows,
                cols=self.cols,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                orbit=self.periodic_orbit,
                stride=self.orbit_stride,
                x=xf,
            )
        else:
            p = len(self.periodic_orbit)
            s0 = math.fsum(xf)
            s1 = math.fsum(j * xj for j, xj in enumerate(xf))
            folded = [0.0] * p
            if p > 0:
                for j, xj in enumerate(xf):
                    folded[j % p] += xj

            out = [0.0] * self.rows
            for i in range(self.rows):
                acc = (self.alpha + self.beta * i) * s0 + (self.gamma + self.delta * i) * s1
                if p > 0:
                    shift = (self.orbit_stride * i) % p
                    for q in range(p):
                        acc += self.periodic_orbit[(q - shift) % p] * folded[q]
                out[i] = acc

        # Add low-rank manifold U (V^T x) in O(R * (N + M))
        if self.rank > 0:
            r = self.rank
            z = [math.fsum(self.low_rank_v[j][k] * xf[j] for j in range(self.cols)) for k in range(r)]
            for i in range(self.rows):
                u_i = self.low_rank_u[i]
                out[i] += math.fsum(u_i[k] * z[k] for k in range(r))

        # Add sparse exception deltas in O(|supp(Delta)|)
        for r_idx, c_idx, dv in self.sparse_entries:
            out[r_idx] += dv * xf[c_idx]

        return out


@dataclass(frozen=True, slots=True)
class LinearRecurrenceGround:
    r"""Contractive/isometric affine state-space recurrence $h_{t+1} = A h_t + b \in \mathbb{R}^d$."""

    transition_matrix: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]

    def __post_init__(self) -> None:
        d = len(self.transition_matrix)
        if d == 0 or len(self.bias) != d or any(len(row) != d for row in self.transition_matrix):
            raise ValueError("transition_matrix must be non-empty square (d x d) matching bias (d,).")
        if any(not math.isfinite(float(v)) for row in self.transition_matrix for v in row) or any(
            not math.isfinite(float(b)) for b in self.bias
        ):
            raise ValueError("Recurrence parameters must be finite.")
        # Sound induced operator-norm non-expansiveness gate: min(||A||_inf, ||A||_1, ||A||_2) <= 1 + 1e-9
        norm_inf = max(math.fsum(abs(float(v)) for v in row) for row in self.transition_matrix)
        norm_1 = max(math.fsum(abs(float(self.transition_matrix[i][j])) for i in range(d)) for j in range(d))
        norm_2 = _spectral_norm_f64(self.transition_matrix)
        if min(norm_inf, norm_1, norm_2) > 1.0 + 1e-9:
            raise ValueError(
                "LinearRecurrenceGround refuses unstable floating-point matrix exponentiation "
                f"(||A||_inf={norm_inf:.6g}, ||A||_1={norm_1:.6g}, ||A||_2={norm_2:.6g} all exceed 1)."
            )

    @property
    def dim(self) -> int:
        return len(self.bias)

    def step_sequential(self, h0: Sequence[float], steps: int) -> list[float]:
        r"""Traditional $\Theta(T d^2)$ step-by-step evaluation."""
        if len(h0) != self.dim or steps < 0 or any(not math.isfinite(float(v)) for v in h0):
            raise ValueError("Invalid/non-finite h0 or negative steps.")
        h = [float(v) for v in h0]
        a = self.transition_matrix
        b = self.bias
        for _ in range(steps):
            h = [math.fsum(a[i][j] * h[j] for j in range(self.dim)) + b[i] for i in range(self.dim)]
        return h

    def navigate_horizon(self, h0: Sequence[float], steps: int) -> tuple[list[float], int]:
        r"""Evaluate $h_T$ in $O(d^3 \log_2 T)$ homogeneous block-matrix multiplications."""
        if len(h0) != self.dim or steps < 0 or any(not math.isfinite(float(v)) for v in h0):
            raise ValueError("Invalid/non-finite h0 or negative steps.")
        m = self.dim + 1

        def mat_mul(p: list[list[float]], q: list[list[float]]) -> list[list[float]]:
            qt = [[q[r][c] for r in range(m)] for c in range(m)]
            return [[math.fsum(p[r][k] * qt[c][k] for k in range(m)) for c in range(m)] for r in range(m)]

        base = [[0.0] * m for _ in range(m)]
        for i in range(self.dim):
            for j in range(self.dim):
                base[i][j] = float(self.transition_matrix[i][j])
            base[i][self.dim] = float(self.bias[i])
        base[self.dim][self.dim] = 1.0

        acc = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
        rem = steps
        binary_steps = 0
        while rem > 0:
            binary_steps += 1
            if rem & 1:
                acc = mat_mul(acc, base)
            rem >>= 1
            if rem > 0:
                base = mat_mul(base, base)

        vec = [*(float(v) for v in h0), 1.0]
        out = [math.fsum(acc[i][j] * vec[j] for j in range(m)) for i in range(self.dim)]
        return out, binary_steps


def benchmark_virtual_vs_traditional(
    *,
    matrix_dim: int = 1024,
    recurrence_steps: int = 25_000,
) -> dict[str, Any]:
    """Empirically benchmark Virtual Layer algebraic contraction & horizon leaping vs traditional compute."""
    rows = cols = matrix_dim
    orbit = tuple(math.sin(0.25 * k) for k in range(8))
    u = tuple((math.cos(0.01 * i), math.sin(0.02 * i)) for i in range(rows))
    v = tuple((math.sin(0.01 * j), math.cos(0.02 * j)) for j in range(cols))
    op = StructuralLinearOperator(
        rows=rows,
        cols=cols,
        alpha=0.125,
        beta=1e-4,
        gamma=-2e-4,
        delta=1e-7,
        periodic_orbit=orbit,
        orbit_stride=3,
        low_rank_u=u,
        low_rank_v=v,
        sparse_entries=((0, 0, 1.5), (rows // 2, cols // 2, -0.75), (rows - 1, cols - 1, 2.25)),
    )
    x = [math.cos(0.05 * j) for j in range(cols)]

    def _eval_trad() -> list[float]:
        if has_numpy():
            import numpy as np

            ii = np.arange(rows, dtype=np.float64)[:, None]
            jj = np.arange(cols, dtype=np.float64)[None, :]
            orb_arr = np.asarray(orbit, dtype=np.float64)
            w_dense = (
                op.alpha
                + op.beta * ii
                + op.gamma * jj
                + op.delta * (ii * jj)
                + orb_arr[(np.arange(cols)[None, :] - op.orbit_stride * np.arange(rows)[:, None]) % len(orbit)]
                + (np.asarray(u, dtype=np.float64) @ np.asarray(v, dtype=np.float64).T)
            )
            for r_idx, c_idx, dv in op.sparse_entries:
                w_dense[r_idx, c_idx] += dv
            return (w_dense @ np.asarray(x, dtype=np.float64)).tolist()
        dense = op.materialize_dense()
        return [math.fsum(r_val * x_val for r_val, x_val in zip(row, x)) for row in dense]

    # Warm up compiled DLL / allocator once before honest median-of-3 wall-clock measurement
    _ = op.contract(x)
    vld_samples_ms: list[float] = []
    y_vld: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        y_vld = op.contract(x)
        vld_samples_ms.append((time.perf_counter() - t0) * 1000.0)
    vld_matvec_ms = statistics.median(vld_samples_ms)

    trad_samples_ms: list[float] = []
    y_trad: list[float] = []
    for _ in range(3):
        t1 = time.perf_counter()
        y_trad = _eval_trad()
        trad_samples_ms.append((time.perf_counter() - t1) * 1000.0)
    trad_matvec_ms = statistics.median(trad_samples_ms)

    matvec_max_err = max(abs(a - b) for a, b in zip(y_vld, y_trad))
    assert matvec_max_err < 1e-8, f"Contraction error too large: {matvec_max_err}"

    cost_vld_matvec = NavigationCostVector(
        latency_ms=vld_matvec_ms,
        compute_ops=float(op.structural_flops),
        bytes_moved=float(op.structural_bytes_f64 + cols * 8 + rows * 8),
        verification_ops=float(rows),
    )
    cost_trad_matvec = NavigationCostVector(
        latency_ms=trad_matvec_ms,
        compute_ops=float(op.dense_flops),
        bytes_moved=float(op.dense_bytes_f64 + cols * 8 + rows * 8),
        verification_ops=float(rows),
    )

    # 3. Linear recurrence trajectory leap vs sequential stepping
    angle = 0.03
    c_ang, s_ang = math.cos(angle), math.sin(angle)
    # damped 4D block rotation (contractive: row L2 <= 0.999)
    damp = 0.9995
    rec = LinearRecurrenceGround(
        transition_matrix=(
            (damp * c_ang, -damp * s_ang, 0.0, 0.0),
            (damp * s_ang, damp * c_ang, 0.0, 0.0),
            (0.0, 0.0, damp * c_ang, damp * s_ang),
            (0.0, 0.0, -damp * s_ang, damp * c_ang),
        ),
        bias=(0.01, -0.02, 0.005, 0.015),
    )
    h0 = (1.0, 0.0, -0.5, 0.25)

    t2 = time.perf_counter()
    h_vld, bin_steps = rec.navigate_horizon(h0, recurrence_steps)
    vld_rec_ms = (time.perf_counter() - t2) * 1000.0

    t3 = time.perf_counter()
    h_trad = rec.step_sequential(h0, recurrence_steps)
    trad_rec_ms = (time.perf_counter() - t3) * 1000.0

    rec_max_err = max(abs(a - b) for a, b in zip(h_vld, h_trad))
    assert rec_max_err < 1e-9, f"Recurrence leap error too large: {rec_max_err}"

    d = rec.dim
    cost_vld_rec = NavigationCostVector(
        latency_ms=vld_rec_ms,
        compute_ops=float(bin_steps * 2 * ((d + 1) ** 3)),
        bytes_moved=float(((d + 1) ** 2) * 8),
        verification_ops=float(d),
    )
    cost_trad_rec = NavigationCostVector(
        latency_ms=trad_rec_ms,
        compute_ops=float(recurrence_steps * 2 * d * d),
        bytes_moved=float(recurrence_steps * d * 8),
        verification_ops=float(d),
    )

    return {
        "linear_recurrence_leap": {
            "binary_steps": bin_steps,
            "max_abs_error": rec_max_err,
            "pareto_dominates_traditional": cost_vld_rec.pareto_dominates(cost_trad_rec),
            "recurrence_steps": recurrence_steps,
            "speedup_factor": round(trad_rec_ms / max(vld_rec_ms, 1e-6), 2),
            "trad_latency_ms": round(trad_rec_ms, 4),
            "vld_latency_ms": round(vld_rec_ms, 4),
        },
        "native_rust_cdll_active": get_native_accelerator().available,
        "numpy_vectorization_active": has_numpy(),
        "structural_operator_contraction": {
            "byte_reduction_factor": round(op.dense_bytes_f64 / max(1, op.structural_bytes_f64), 2),
            "dense_bytes": op.dense_bytes_f64,
            "dense_flops": op.dense_flops,
            "flop_reduction_factor": round(op.dense_flops / max(1, op.structural_flops), 2),
            "matrix_shape": [rows, cols],
            "max_abs_error": matvec_max_err,
            "pareto_dominates_dense_flops_and_bytes": (
                op.structural_flops < op.dense_flops and op.structural_bytes_f64 < op.dense_bytes_f64
            ),
            "speedup_factor": round(trad_matvec_ms / max(vld_matvec_ms, 1e-6), 2),
            "structural_bytes": op.structural_bytes_f64,
            "structural_flops": op.structural_flops,
            "trad_latency_ms": round(trad_matvec_ms, 4),
            "vld_latency_ms": round(vld_matvec_ms, 4),
            "vld_pareto_dominates_cost_vector": cost_vld_matvec.pareto_dominates(cost_trad_matvec),
        },
    }
