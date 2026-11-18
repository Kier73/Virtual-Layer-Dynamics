r"""Phase 10 (`IDEA-007`) — Proof-Carrying Numerical Accuracy & Certified Logit-Margin Escalation (`vld.models.precision_cert`).

Governing equations:
Let a multi-layer or cross-host approximate trajectory produce $\tilde{h}^{(L)} \in \mathbb{R}^d$ under
quantized (`FP4`/`FP8`), Sinkhorn `mHC`, or distributed streaming execution.
1. **Exact Rational Error Propagation (`ErrorEnclosureCertificate`):**
   Each stage $\ell \in \{0, \dots, L-1\}$ supplies a rigorous upper-bounded operator contraction/Lipschitz
   factor $L_\ell \in \mathbb{Q}_{\ge 0}$ and local perturbation radius $\eta_\ell \in \mathbb{Q}_{\ge 0}$:
   $$\varepsilon^{(0)} = \eta_{\text{init}}, \qquad \varepsilon^{(\ell+1)} = L_\ell \cdot \varepsilon^{(\ell)} + \eta_\ell \in \mathbb{Q}_{\ge 0}.$$
   All recurrences are evaluated in exact `Fraction` arithmetic (converting floating-point norms strictly
   upward via `_upper_fraction`), eliminating floating-point under-approximation in the certificate itself!
2. **Certified Argmax Token Invariance vs. Selective Escalation:**
   Let $\tilde{z} = W_{\text{head}} \tilde{h}^{(L)} + b_{\text{head}}$ with top-1 index $v_1 = \arg\max_v \tilde{z}_v$
   and runner-up $v_2 = \arg\max_{v \neq v_1} \tilde{z}_v$. By Cauchy-Schwarz, for every competitor $v \neq v_1$:
   $$z_{v_1}^{\text{exact}} - z_v^{\text{exact}} \ge (\tilde{z}_{v_1} - \tilde{z}_v) - \|w_{v_1} - w_v\|_2 \cdot \varepsilon^{(L)} - 2 \eta_{\text{head}}.$$
   - If the worst-case lower bound is strictly positive across all $v \neq v_1$, the certificate emits
     `CERTIFIED_TOKEN_INVARIANT`: **provably** $\arg\max_v z_v^{\text{exact}} = v_1$ without fetching higher-precision weights!
   - Otherwise, `CertifiedPrecisionEscalator` flags `AMBIGUOUS_MARGIN_ESCALATE` and selectively invokes the
     high-precision refinement realizer *only* on that ambiguous decision step.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import domain_digest


class PrecisionCertificateError(ValueError):
    """Raised when a Lipschitz constant, error radius, or precision certificate fails validation."""


# 1 ULP safety cushion above IEEE-754 binary64 float-to-rational conversion for rigorous upper bounds
_UPPER_ULP_SCALE = Fraction(1) + Fraction(1, 1 << 50)


def upper_rational_bound(val: int | float | Fraction | str) -> Fraction:
    """Convert a non-negative real/rational bound conservatively upward into $\\mathbb{Q}_{\\ge 0}$."""
    if isinstance(val, bool):
        raise PrecisionCertificateError("Boolean not valid as numerical bound.")
    if isinstance(val, (int, Fraction, str)):
        f = Fraction(val)
        if f < 0:
            raise PrecisionCertificateError("Error/Lipschitz bound must be non-negative.")
        return f
    fv = float(val)
    if not math.isfinite(fv) or fv < 0.0:
        raise PrecisionCertificateError("Float bound must be finite and non-negative.")
    if fv == 0.0:
        return Fraction(0)
    return Fraction.from_float(fv) * _UPPER_ULP_SCALE


def _fmt_frac(f: Fraction) -> str:
    return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)


@dataclass(frozen=True, slots=True)
class LayerErrorStep:
    """Per-layer Lipschitz amplification $L_\\ell$ and additive quantization/rounding error $\\eta_\\ell$."""

    layer_id: str
    lipschitz_upper: Fraction
    local_error_radius: Fraction
    sinkhorn_col_l1_norm: Fraction | None = None

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise PrecisionCertificateError("layer_id must be non-empty.")
        lip = upper_rational_bound(self.lipschitz_upper)
        loc = upper_rational_bound(self.local_error_radius)
        object.__setattr__(self, "lipschitz_upper", lip)
        object.__setattr__(self, "local_error_radius", loc)
        if self.sinkhorn_col_l1_norm is not None:
            sc = upper_rational_bound(self.sinkhorn_col_l1_norm)
            object.__setattr__(self, "sinkhorn_col_l1_norm", sc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "lipschitz_upper": _fmt_frac(self.lipschitz_upper),
            "local_error_radius": _fmt_frac(self.local_error_radius),
            "sinkhorn_col_l1_norm": (
                None if self.sinkhorn_col_l1_norm is None else _fmt_frac(self.sinkhorn_col_l1_norm)
            ),
        }


@dataclass(frozen=True, slots=True)
class ErrorEnclosureCertificate:
    """Portable proof-carrying error enclosure accompanying a cross-host neural hidden state."""

    trajectory_id: str
    initial_error_radius: Fraction
    steps: tuple[LayerErrorStep, ...]
    terminal_error_radius: Fraction
    certificate_digest: str = ""

    def __post_init__(self) -> None:
        init_r = upper_rational_bound(self.initial_error_radius)
        cur = init_r
        for st in self.steps:
            cur = st.lipschitz_upper * cur + st.local_error_radius
        term_r = upper_rational_bound(self.terminal_error_radius)
        if term_r < cur:
            raise PrecisionCertificateError(
                f"Declared terminal_error_radius {term_r} under-approximates propagated bound {cur}."
            )
        object.__setattr__(self, "initial_error_radius", init_r)
        object.__setattr__(self, "terminal_error_radius", term_r)
        comp = domain_digest(
            "vld.error-enclosure-certificate.v1",
            {
                "initial_error_radius": _fmt_frac(init_r),
                "steps": [s.to_dict() for s in self.steps],
                "terminal_error_radius": _fmt_frac(term_r),
                "trajectory_id": self.trajectory_id,
            },
        )
        if self.certificate_digest and self.certificate_digest != comp:
            raise PrecisionCertificateError("ErrorEnclosureCertificate digest mismatch.")
        object.__setattr__(self, "certificate_digest", comp)

    @classmethod
    def propagate(
        cls,
        trajectory_id: str,
        steps: Sequence[LayerErrorStep],
        *,
        initial_error_radius: int | float | Fraction | str = 0,
    ) -> ErrorEnclosureCertificate:
        cur = upper_rational_bound(initial_error_radius)
        init_r = cur
        step_tup = tuple(steps)
        for st in step_tup:
            cur = st.lipschitz_upper * cur + st.local_error_radius
        return cls(
            trajectory_id=trajectory_id,
            initial_error_radius=init_r,
            steps=step_tup,
            terminal_error_radius=cur,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "certificate_digest": self.certificate_digest,
            "initial_error_radius": _fmt_frac(self.initial_error_radius),
            "schema": "vld.error-enclosure-certificate.v1",
            "steps": [s.to_dict() for s in self.steps],
            "terminal_error_radius": _fmt_frac(self.terminal_error_radius),
            "trajectory_id": self.trajectory_id,
        }


@dataclass(frozen=True, slots=True)
class CertifiedDecisionVerdict:
    """Standalone-verifiable proof that top-1 token `selected_token` is invariant (or requires escalation)."""

    trajectory_id: str
    selected_token: int
    runner_up_token: int
    observed_logit_margin: Fraction
    worst_case_logit_perturbation: Fraction
    certified_lower_margin: Fraction
    certified_invariant: bool
    enclosure: ErrorEnclosureCertificate
    head_max_row_l2_upper: Fraction
    head_rounding_eps: Fraction
    verdict_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.certified-decision-verdict.v1",
            {
                "certified_invariant": self.certified_invariant,
                "certified_lower_margin": _fmt_frac(self.certified_lower_margin),
                "enclosure_digest": self.enclosure.certificate_digest,
                "head_max_row_l2_upper": _fmt_frac(self.head_max_row_l2_upper),
                "head_rounding_eps": _fmt_frac(self.head_rounding_eps),
                "observed_logit_margin": _fmt_frac(self.observed_logit_margin),
                "runner_up_token": self.runner_up_token,
                "selected_token": self.selected_token,
                "trajectory_id": self.trajectory_id,
                "worst_case_logit_perturbation": _fmt_frac(self.worst_case_logit_perturbation),
            },
        )
        if self.verdict_digest and self.verdict_digest != comp:
            raise PrecisionCertificateError("CertifiedDecisionVerdict digest mismatch.")
        object.__setattr__(self, "verdict_digest", comp)

    def to_bundle_dict(self) -> dict[str, Any]:
        return {
            "certified_invariant": self.certified_invariant,
            "certified_lower_margin": _fmt_frac(self.certified_lower_margin),
            "enclosure": self.enclosure.to_dict(),
            "head_max_row_l2_upper": _fmt_frac(self.head_max_row_l2_upper),
            "head_rounding_eps": _fmt_frac(self.head_rounding_eps),
            "observed_logit_margin": _fmt_frac(self.observed_logit_margin),
            "runner_up_token": self.runner_up_token,
            "schema": "vld.certified-decision-bundle.v1",
            "selected_token": self.selected_token,
            "trajectory_id": self.trajectory_id,
            "verdict_digest": self.verdict_digest,
            "worst_case_logit_perturbation": _fmt_frac(self.worst_case_logit_perturbation),
        }


def certify_logit_margin_invariance(
    approx_logits: Sequence[float],
    enclosure: ErrorEnclosureCertificate,
    *,
    lm_head_weight: Sequence[Sequence[float]] | None = None,
    head_max_row_l2: float | Fraction | None = None,
    head_rounding_eps: float | Fraction = Fraction(1, 10**6),
) -> CertifiedDecisionVerdict:
    r"""Certify whether $\arg\max_v z_v$ is provably invariant inside the error ball $\varepsilon^{(L)}$."""
    if len(approx_logits) < 2:
        raise PrecisionCertificateError("approx_logits must contain at least 2 vocabulary entries.")
    if any(not math.isfinite(float(z)) for z in approx_logits):
        raise PrecisionCertificateError("Non-finite entry in approx_logits.")

    # Identify top-1 and runner-up tokens
    best_idx = 0
    second_idx = 1
    if float(approx_logits[1]) > float(approx_logits[0]):
        best_idx, second_idx = 1, 0
    for idx in range(2, len(approx_logits)):
        val = float(approx_logits[idx])
        if val > float(approx_logits[best_idx]):
            second_idx = best_idx
            best_idx = idx
        elif val > float(approx_logits[second_idx]):
            second_idx = idx

    # Convert observed margin conservatively downward
    raw_gap = float(approx_logits[best_idx]) - float(approx_logits[second_idx])
    obs_margin = max(Fraction(0), Fraction.from_float(raw_gap) - Fraction(1, 1 << 48))
    head_eps = upper_rational_bound(head_rounding_eps)

    if head_max_row_l2 is not None:
        max_row_norm = upper_rational_bound(head_max_row_l2)
    elif lm_head_weight is not None:
        if len(lm_head_weight) != len(approx_logits):
            raise PrecisionCertificateError("lm_head_weight row count disagrees with approx_logits length.")
        max_sq = max(math.fsum(float(x) * float(x) for x in row) for row in lm_head_weight)
        max_row_norm = upper_rational_bound(math.sqrt(max_sq))
    else:
        raise PrecisionCertificateError("Supply either lm_head_weight or head_max_row_l2.")

    # Global uniform perturbation bound: 2 * ||W_head||_{2,inf} * eps_L + 2 * head_eps
    global_pert = 2 * max_row_norm * enclosure.terminal_error_radius + 2 * head_eps
    worst_pert = global_pert

    # Tighter pairwise row-difference Cauchy-Schwarz refinement when lm_head_weight is available!
    if lm_head_weight is not None and obs_margin <= global_pert:
        w_best = lm_head_weight[best_idx]
        min_lower: Fraction | None = None
        max_eff_pert = Fraction(0)
        for v_idx, z_v in enumerate(approx_logits):
            if v_idx == best_idx:
                continue
            gap_v = max(Fraction(0), Fraction.from_float(float(approx_logits[best_idx]) - float(z_v)) - Fraction(1, 1 << 48))
            if gap_v > global_pert:
                cand_lower = gap_v - global_pert
            else:
                w_v = lm_head_weight[v_idx]
                diff_norm = upper_rational_bound(
                    math.sqrt(math.fsum((float(a) - float(b)) ** 2 for a, b in zip(w_best, w_v)))
                )
                pert_v = diff_norm * enclosure.terminal_error_radius + 2 * head_eps
                max_eff_pert = max(max_eff_pert, pert_v)
                cand_lower = gap_v - pert_v
            if min_lower is None or cand_lower < min_lower:
                min_lower = cand_lower
        assert min_lower is not None
        cert_lower = min_lower
        worst_pert = obs_margin - cert_lower
    else:
        cert_lower = obs_margin - worst_pert

    is_inv = cert_lower > 0
    return CertifiedDecisionVerdict(
        trajectory_id=enclosure.trajectory_id,
        selected_token=best_idx,
        runner_up_token=second_idx,
        observed_logit_margin=obs_margin,
        worst_case_logit_perturbation=worst_pert,
        certified_lower_margin=cert_lower,
        certified_invariant=is_inv,
        enclosure=enclosure,
        head_max_row_l2_upper=max_row_norm,
        head_rounding_eps=head_eps,
    )


class CertifiedPrecisionEscalator:
    """Executes low-precision neural inference first and escalates ONLY when the error ball straddles the logit margin."""

    def __init__(
        self,
        *,
        head_max_row_l2: float | Fraction | None = None,
        lm_head_weight: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.head_max_row_l2 = head_max_row_l2
        self.lm_head_weight = lm_head_weight
        self.total_steps = 0
        self.certified_fast_steps = 0
        self.escalated_steps = 0

    def step_with_certificate(
        self,
        *,
        low_precision_fn: Callable[[], tuple[Sequence[float], ErrorEnclosureCertificate]],
        high_precision_fallback_fn: Callable[[], Sequence[float]],
    ) -> tuple[int, str, CertifiedDecisionVerdict]:
        self.total_steps += 1
        approx_logits, enclosure = low_precision_fn()
        verdict = certify_logit_margin_invariance(
            approx_logits,
            enclosure,
            lm_head_weight=self.lm_head_weight,
            head_max_row_l2=self.head_max_row_l2,
        )
        if verdict.certified_invariant:
            self.certified_fast_steps += 1
            return verdict.selected_token, "CERTIFIED_LOW_PRECISION", verdict
        self.escalated_steps += 1
        exact_logits = high_precision_fallback_fn()
        refined_tok = max(range(len(exact_logits)), key=lambda i: float(exact_logits[i]))
        return refined_tok, "ESCALATED_HIGH_PRECISION", verdict


__all__ = [
    "CertifiedDecisionVerdict",
    "CertifiedPrecisionEscalator",
    "ErrorEnclosureCertificate",
    "LayerErrorStep",
    "PrecisionCertificateError",
    "certify_logit_margin_invariance",
    "upper_rational_bound",
]
