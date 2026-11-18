r"""Phase 10 (`IDEA-005`) — Cross-Model Structural Consensus & Centered Relational Sketches (`vld.models.consensus`).

Mathematical foundation:
Heterogeneous frontier & compact models (`DeepSeek-V4.1` $d=4096$, `GLM-5.3` $d=5120$, `Qwen3-0.6B` $d=1024$,
`SmolLM2-135M` $d=576$, `DeepSeek-R1-Distill-1.5B` $d=1536$) inhabit incompatible hidden dimensions $d_i \neq d_j$
and distinct tokenizers/vocabularies.

1. **Basis-Invariant Token-Gram Geometry:**
   Given hidden states $H \in \mathbb{R}^{T \times d}$ ($T \ge 2$), normalize token vectors
   $\hat{h}_t = h_t / \max(\|h_t\|_2, 10^{-12})$ and form either:
   - Exact Token-Gram $G_{t, t'} = \langle \hat{h}_t, \hat{h}_{t'} \rangle \in \mathbb{R}^{T \times T}$, OR
   - Seed-pinned Achlioptas sparse JL sketch $Z = \hat{H} \Pi^\top \in \mathbb{R}^{T \times r}$
     ($\Pi_{k, j} \in \{-\sqrt{3/r}, 0, +\sqrt{3/r}\}$ with probabilities $(\tfrac{1}{6}, \tfrac{2}{3}, \tfrac{1}{6})$),
     approximating $\tilde{G} = Z Z^\top$ in $O(T r)$ wire floats without transmitting full hidden activations!

2. **Anisotropy Removal via Double-Centering ($H_T = I_T - \frac{1}{T}\mathbf{1}\mathbf{1}^\top$):**
   Transformer hidden states concentrate in a narrow positive cone ($\langle \hat{h}_t, \hat{h}_{t'} \rangle > 0$),
   which inflates naive uncentered inner products even on unrelated inputs. Double-centering
   $$\bar{G} = H_T G H_T, \qquad \bar{G}_{t, t'} = G_{t, t'} - \mu_{t,\cdot} - \mu_{\cdot, t'} + \mu_{\cdot, \cdot}$$
   yields **Linear Centered Kernel Alignment (`CKA`)** and **Normalized Frobenius Relational Divergence**:
   $$\operatorname{CKA}(M_i, M_j) = \frac{\langle \bar{G}^{(M_i)}, \bar{G}^{(M_j)} \rangle_F}{\|\bar{G}^{(M_i)}\|_F \, \|\bar{G}^{(M_j)}\|_F} \in [-1, 1], \qquad \Delta_{\text{rel}}(M_i, M_j) = \left\| \frac{\bar{G}^{(M_i)}}{\|\bar{G}^{(M_i)}\|_F} - \frac{\bar{G}^{(M_j)}}{\|\bar{G}^{(M_j)}\|_F} \right\|_F,$$
   satisfying exact orthogonal invariance $\operatorname{CKA}(H Q_1, H Q_2) = 1$ for all $Q_1 \in O(d_1), Q_2 \in O(d_2)$.

3. **Adversarial Permutation Null Control:**
   Computes empirical null distribution $\{\operatorname{CKA}(\bar{G}^{(M_i)}, P_\pi \bar{G}^{(M_j)} P_\pi^\top)\}_{\pi}$
   over deterministic Fisher-Yates token permutations (`T >= 4`) to separate genuine token-relational
   alignment from chance finite-sample correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest


class RelationalConsensusError(ValueError):
    """Raised on invalid hidden-state trajectory shapes, non-finite activations, or mismatched sequence lengths."""


def _normalize_rows(hidden_2d: Sequence[Sequence[float]]) -> tuple[int, int, list[list[float]]]:
    t_len = len(hidden_2d)
    if t_len < 2:
        raise RelationalConsensusError("Relational Gram comparison requires sequence length T >= 2.")
    d_dim = len(hidden_2d[0])
    if d_dim <= 0 or any(len(row) != d_dim for row in hidden_2d):
        raise RelationalConsensusError("Malformed hidden_2d dimensions.")
    normed: list[list[float]] = []
    for row in hidden_2d:
        vals = [float(x) for x in row]
        if any(not math.isfinite(v) for v in vals):
            raise RelationalConsensusError("Non-finite activation in hidden_2d.")
        nrm = math.sqrt(math.fsum(v * v for v in vals))
        denom = nrm if nrm > 1e-12 else 1.0
        normed.append([v / denom for v in vals])
    return t_len, d_dim, normed


def _double_center_gram(gram: Sequence[Sequence[float]]) -> list[list[float]]:
    t_len = len(gram)
    row_means = [math.fsum(row) / t_len for row in gram]
    col_means = [math.fsum(gram[r][c] for r in range(t_len)) / t_len for c in range(t_len)]
    grand_mean = math.fsum(row_means) / t_len
    return [
        [gram[r][c] - row_means[r] - col_means[c] + grand_mean for c in range(t_len)]
        for r in range(t_len)
    ]


def _frob_inner(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    return math.fsum(a[r][c] * b[r][c] for r in range(len(a)) for c in range(len(a)))


def _achlioptas_project(
    normed_rows: Sequence[Sequence[float]],
    *,
    sketch_dim: int,
    seed_tag: str,
) -> list[list[float]]:
    """Deterministic domain-separated Achlioptas sparse JL projection onto $\\mathbb{R}^{T \\times r}$."""
    if sketch_dim <= 0:
        raise RelationalConsensusError("sketch_dim must be positive.")
    d_dim = len(normed_rows[0])
    scale = math.sqrt(3.0 / float(sketch_dim))
    # Precompute sparse non-zero indices & signs per sketch coordinate k in [0, r)
    sparse_rows: list[list[tuple[int, float]]] = []
    for k in range(sketch_dim):
        nz: list[tuple[int, float]] = []
        # Stream SHA-256 blocks deterministically (1 byte per coordinate j)
        blk_idx = 0
        buf = b""
        pos = 0
        for j in range(d_dim):
            if pos >= len(buf):
                buf = hashlib.sha256(f"vld.jl.v1:{seed_tag}:{k}:{blk_idx}".encode("utf-8")).digest()
                blk_idx += 1
                pos = 0
            b_val = buf[pos] % 6
            pos += 1
            if b_val == 0:
                nz.append((j, +scale))
            elif b_val == 1:
                nz.append((j, -scale))
        sparse_rows.append(nz)

    out: list[list[float]] = []
    for row in normed_rows:
        proj = [math.fsum(sign_s * row[j] for j, sign_s in nz) for nz in sparse_rows]
        out.append(proj)
    return out


@dataclass(frozen=True, slots=True)
class CenteredGramRelationalSketch:
    """Compact $T \\times T$ double-centered relational summary independent of model hidden dimension $d$."""

    model_id: str
    seq_len: int
    source_hidden_dim: int
    sketch_mode: str
    sketch_dim: int
    centered_gram: tuple[tuple[float, ...], ...]
    frobenius_norm: float
    wire_bytes_estimate: int
    sketch_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.relational-gram-sketch.v1",
            {
                "centered_gram": [[round(float(x), 8) for x in r] for r in self.centered_gram],
                "model_id": self.model_id,
                "seq_len": self.seq_len,
                "sketch_dim": self.sketch_dim,
                "sketch_mode": self.sketch_mode,
                "source_hidden_dim": self.source_hidden_dim,
            },
        )
        if self.sketch_digest and self.sketch_digest != comp:
            raise RelationalConsensusError("CenteredGramRelationalSketch digest mismatch.")
        object.__setattr__(self, "sketch_digest", comp)

    @classmethod
    def from_hidden_states(
        cls,
        model_id: str,
        hidden_2d: Sequence[Sequence[float]],
        *,
        sketch_dim: int | None = None,
        seed_tag: str = "shared-consensus",
    ) -> CenteredGramRelationalSketch:
        t_len, d_dim, normed = _normalize_rows(hidden_2d)
        if sketch_dim is None:
            gram = [
                [math.fsum(normed[r][k] * normed[c][k] for k in range(d_dim)) for c in range(t_len)]
                for r in range(t_len)
            ]
            mode = "exact_gram"
            eff_dim = d_dim
            wire_est = 4 * t_len * t_len
        else:
            proj = _achlioptas_project(normed, sketch_dim=sketch_dim, seed_tag=seed_tag)
            gram = [
                [math.fsum(proj[r][k] * proj[c][k] for k in range(sketch_dim)) for c in range(t_len)]
                for r in range(t_len)
            ]
            mode = "sparse_jl_gram"
            eff_dim = int(sketch_dim)
            wire_est = 4 * min(t_len * t_len, t_len * eff_dim)

        centered = _double_center_gram(gram)
        f_norm = math.sqrt(max(0.0, _frob_inner(centered, centered)))
        return cls(
            model_id=model_id,
            seq_len=t_len,
            source_hidden_dim=d_dim,
            sketch_mode=mode,
            sketch_dim=eff_dim,
            centered_gram=tuple(tuple(row) for row in centered),
            frobenius_norm=f_norm,
            wire_bytes_estimate=wire_est,
        )

    @classmethod
    def from_pooled_trajectories(
        cls,
        model_id: str,
        statement_trajectories: Sequence[Sequence[Sequence[float]]],
        *,
        sketch_dim: int | None = None,
        seed_tag: str = "shared-consensus",
    ) -> CenteredGramRelationalSketch:
        r"""Build a position-debiased $M \times M$ cross-statement Gram sketch across variable-length BPE sequences.

        Eliminates both subword tokenization length mismatch ($T_m^{(A)} \neq T_m^{(B)}$) and universal
        intra-sequence RoPE causal position drift $\mu_{\text{pos}}(t)$ discovered in `CMC-001`.
        """
        if len(statement_trajectories) < 2:
            raise RelationalConsensusError("from_pooled_trajectories requires M >= 2 distinct statement trajectories.")
        pooled: list[list[float]] = []
        d_expected: int | None = None
        for traj in statement_trajectories:
            if not traj:
                raise RelationalConsensusError("Empty token trajectory in statement_trajectories.")
            d_cur = len(traj[0])
            if d_cur <= 0 or any(len(tok_v) != d_cur for tok_v in traj):
                raise RelationalConsensusError("Inconsistent hidden dimension inside statement trajectory.")
            if d_expected is None:
                d_expected = d_cur
            elif d_cur != d_expected:
                raise RelationalConsensusError("Inconsistent hidden dimension across statement_trajectories.")
            inv_t = 1.0 / float(len(traj))
            pooled.append([math.fsum(float(tok_v[j]) for tok_v in traj) * inv_t for j in range(d_cur)])
        return cls.from_hidden_states(model_id, pooled, sketch_dim=sketch_dim, seed_tag=seed_tag)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frobenius_norm": round(self.frobenius_norm, 8),
            "model_id": self.model_id,
            "seq_len": self.seq_len,
            "sketch_digest": self.sketch_digest,
            "sketch_dim": self.sketch_dim,
            "sketch_mode": self.sketch_mode,
            "source_hidden_dim": self.source_hidden_dim,
            "wire_bytes_estimate": self.wire_bytes_estimate,
        }


@dataclass(frozen=True, slots=True)
class CrossModelConsensusReport:
    """Pairwise structural alignment & adversarial permutation calibration between two models."""

    model_a: str
    model_b: str
    seq_len: int
    linear_cka: float
    relational_divergence: float
    permutation_null_mean: float
    permutation_null_max: float
    exceeds_permutation_null: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "exceeds_permutation_null": self.exceeds_permutation_null,
            "linear_cka": round(self.linear_cka, 6),
            "model_a": self.model_a,
            "model_b": self.model_b,
            "permutation_null_max": round(self.permutation_null_max, 6),
            "permutation_null_mean": round(self.permutation_null_mean, 6),
            "relational_divergence": round(self.relational_divergence, 6),
            "seq_len": self.seq_len,
        }


def _deterministic_perm(n: int, trial: int) -> list[int]:
    arr = list(range(n))
    digest = hashlib.sha256(f"vld.perm.null.v1:{n}:{trial}".encode("utf-8")).digest()
    for i in range(n - 1, 0, -1):
        j = digest[i % len(digest)] % (i + 1)
        arr[i], arr[j] = arr[j], arr[i]
    # Ensure non-identity permutation when n >= 2
    if arr == list(range(n)) and n >= 2:
        arr[0], arr[1] = arr[1], arr[0]
    return arr


def compare_relational_sketches(
    sketch_a: CenteredGramRelationalSketch,
    sketch_b: CenteredGramRelationalSketch,
    *,
    num_permutations: int = 16,
) -> CrossModelConsensusReport:
    """Compare two models' centered Token-Gram geometries against token-shuffled negative controls."""
    if sketch_a.seq_len != sketch_b.seq_len:
        raise RelationalConsensusError(
            f"Aligned token count mismatch: {sketch_a.seq_len} vs {sketch_b.seq_len}."
        )
    t_len = sketch_a.seq_len
    na = sketch_a.frobenius_norm
    nb = sketch_b.frobenius_norm
    if na <= 1e-12 or nb <= 1e-12:
        cka = 0.0
        div = math.sqrt(2.0)
    else:
        dot = _frob_inner(sketch_a.centered_gram, sketch_b.centered_gram)
        cka = max(-1.0, min(1.0, dot / (na * nb)))
        # ||A/||A|| - B/||B||||_F^2 = 2 - 2 * CKA
        div = math.sqrt(max(0.0, 2.0 - 2.0 * cka))

    null_scores: list[float] = []
    if na > 1e-12 and nb > 1e-12 and num_permutations > 0:
        gb = sketch_b.centered_gram
        for tr in range(num_permutations):
            perm = _deterministic_perm(t_len, tr)
            p_dot = math.fsum(
                sketch_a.centered_gram[r][c] * gb[perm[r]][perm[c]]
                for r in range(t_len)
                for c in range(t_len)
            )
            null_scores.append(max(-1.0, min(1.0, p_dot / (na * nb))))

    null_mean = (math.fsum(null_scores) / len(null_scores)) if null_scores else 0.0
    null_max = max(null_scores) if null_scores else 0.0
    exceeds = (cka > null_max + 0.05) if t_len >= 5 else (cka > null_mean + 0.1)
    return CrossModelConsensusReport(
        model_a=sketch_a.model_id,
        model_b=sketch_b.model_id,
        seq_len=t_len,
        linear_cka=cka,
        relational_divergence=div,
        permutation_null_mean=null_mean,
        permutation_null_max=null_max,
        exceeds_permutation_null=exceeds,
    )


__all__ = [
    "CenteredGramRelationalSketch",
    "CrossModelConsensusReport",
    "RelationalConsensusError",
    "compare_relational_sketches",
]
