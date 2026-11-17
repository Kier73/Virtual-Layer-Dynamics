r"""Phase 7 — Constructive Semantic-to-Neural Compilation, Weight Geometry Morphisms & Generative Weight Grounds.

Governing equations:
1. **Exact 1D Piecewise-Affine Hinge Compilation (`compile_piecewise_affine_relu`):**
   Given breakpoints $t_1 < \dots < t_K$, leftmost affine branch $m_0 x + b_0$, and consecutive slopes $m_1, \dots, m_K$
   satisfying $C^0$ continuity at every $t_k$:
     $$f(x) = b_0 + m_0\big(\operatorname{ReLU}(x) - \operatorname{ReLU}(-x)\big) + \sum_{k=1}^K (m_k - m_{k-1})\operatorname{ReLU}(x - t_k)$$
   Evaluates with **zero approximation error** over $\mathbb{Q}$ and carries an independently verifiable
   activation-cell certificate without importing producer compiler logic.

2. **Finite-State Machine Neural Synthesis (`compile_fsm_to_neural`):**
   Embeds $(s, a) \in \Sigma \times A$ as concatenated one-hot coordinates $(e_s, e_a) \in \{0,1\}^{|\Sigma| + |A|}$.
   Conjunction neuron $h_{s,a}(e_{s'}, e_{a'}) = \operatorname{ReLU}(e_{s',s} + e_{a',a} - 1) = \mathbb{I}[s'=s \land a'=a]$
   realizes arbitrary deterministic Mealy transitions with zero training data.

3. **Function-Preserving Morphisms (`widen_block_dyadic`, `insert_identity_depth`, `permute_hidden_units`):**
   - *Dyadic Net2Wider:* Duplicating hidden neuron $j$ and halving outgoing row $V_{j,:} \mapsto \frac{1}{2} V_{j,:}$
     conserves exact binary32 coefficients (refusing subnormal underflow) and block mapping $\sigma(x W + b)V + c$.
   - *Exact ReLU Depth Identity:* $x = \operatorname{ReLU}(x) - \operatorname{ReLU}(-x)$ inserts hidden layers without changing outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import re
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.accel import decode_f32_le, encode_f32_le
from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.memory import PagedObjectStore, StructuralMemoryError, StructuralMemoryRegistry
from vld.models.artifacts import ModelParameterSpace, ModelWeightSnapshot
from vld.signal.transitions import GroundTransitionSystem

_RAT_TOKEN_RE = re.compile(r"(?:0|-?[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z")
_MIN_NORMAL_F32 = float.fromhex("0x1.0p-126")


class ProceduralModelError(ValueError):
    """Raised when a semantic atlas, compiled neural capsule, morphism, or weight recipe fails verification."""


def _parse_canonical_rational(token: Any) -> Fraction:
    if not isinstance(token, str) or len(token) > 128 or _RAT_TOKEN_RE.fullmatch(token) is None:
        raise ProceduralModelError(f"Non-canonical rational token: {token!r}")
    val = Fraction(token)
    if str(val) != token:
        raise ProceduralModelError(f"Unreduced rational token rejected: {token!r}")
    return val


def _format_rational(val: Fraction | None, *, inf_label: str = "") -> str:
    if val is None:
        return inf_label
    return str(Fraction(val))


# ============================================================================
# 1. Exact Piecewise-Affine Semantic-to-Neural Compiler & Independent Verifier
# ============================================================================


@dataclass(frozen=True, slots=True)
class PiecewiseAffineRegion:
    left: Fraction | None
    right: Fraction | None
    affines: tuple[tuple[Fraction, Fraction], ...]  # (slope, intercept) per output channel


def compile_piecewise_affine_relu(
    atlas_spec: Sequence[Mapping[str, Any]],
    *,
    max_regions: int = 64,
) -> dict[str, Any]:
    """Compile a continuous full-line 1D piecewise-affine atlas into an exact 2-layer rational ReLU capsule."""
    if not isinstance(atlas_spec, (list, tuple)) or not (1 <= len(atlas_spec) <= max_regions):
        raise ProceduralModelError("atlas_spec must contain 1..max_regions interval cells.")

    regions: list[PiecewiseAffineRegion] = []
    out_dim: int | None = None
    for idx, cell in enumerate(atlas_spec):
        if not isinstance(cell, Mapping) or set(cell.keys()) != {"left", "right", "affines"}:
            raise ProceduralModelError("Each atlas cell requires exact keys {'left', 'right', 'affines'}.")
        l_raw, r_raw = cell["left"], cell["right"]
        left = None if l_raw == "-inf" else _parse_canonical_rational(l_raw)
        right = None if r_raw == "+inf" else _parse_canonical_rational(r_raw)
        if (left is None) != (idx == 0) or (right is None) != (idx == len(atlas_spec) - 1):
            raise ProceduralModelError("Atlas must span (-inf, +inf) with infinite endpoints only at boundaries.")
        if left is not None and right is not None and left >= right:
            raise ProceduralModelError("Empty or inverted interval in atlas.")

        raw_affs = cell["affines"]
        if not isinstance(raw_affs, (list, tuple)) or not (1 <= len(raw_affs) <= 16):
            raise ProceduralModelError("Invalid output channel count in atlas.")
        if out_dim is None:
            out_dim = len(raw_affs)
        elif len(raw_affs) != out_dim:
            raise ProceduralModelError("Inconsistent output arity across atlas cells.")

        affs = tuple((_parse_canonical_rational(pair[0]), _parse_canonical_rational(pair[1])) for pair in raw_affs)
        reg = PiecewiseAffineRegion(left, right, affs)
        if regions:
            prev = regions[-1]
            if prev.right != left:
                raise ProceduralModelError("Gap or overlap between adjacent atlas intervals.")
            if prev.affines == reg.affines:
                raise ProceduralModelError("Non-maximal atlas: adjacent intervals have identical affine laws.")
            assert left is not None
            for (m1, b1), (m2, b2) in zip(prev.affines, reg.affines):
                if m1 * left + b1 != m2 * left + b2:
                    raise ProceduralModelError(f"Discontinuous piecewise-affine specification at breakpoint {left}.")
        regions.append(reg)

    breakpoints = [r.left for r in regions[1:] if r.left is not None]
    # Hidden layer: [ReLU(x), ReLU(-x), ReLU(x - t_1), ..., ReLU(x - t_K)]
    w1 = [["1"], ["-1"], *(["1"] for _ in breakpoints)]
    b1 = ["0", "0", *(_format_rational(-bp) for bp in breakpoints)]

    assert out_dim is not None
    w2: list[list[str]] = []
    b2: list[str] = []
    # Intercept at x = 0: find which region contains 0 to anchor exact constant!
    # Wait: let's check f(x) = c_channel + m_0*(ReLU(x) - ReLU(-x)) + sum_{k=1}^K (m_k - m_{k-1}) ReLU(x - t_k)!
    # Notice: for x < t_1 (so x - t_k < 0 for ALL k >= 1, regardless of sign of t_k!),
    # ALL ReLU(x - t_k) == 0, while ReLU(x) - ReLU(-x) == x identically for ALL x in R!
    # Therefore for x < t_1, the formula evaluates to c_channel + m_0 * x, so c_channel == b_0 ALWAYS!
    for ch in range(out_dim):
        m0, b0 = regions[0].affines[ch]
        slopes = [r.affines[ch][0] for r in regions]
        row = [_format_rational(m0), _format_rational(-m0)]
        for prev_m, cur_m in zip(slopes, slopes[1:]):
            row.append(_format_rational(cur_m - prev_m))
        w2.append(row)
        b2.append(_format_rational(b0))

    canonical_atlas = [
        {
            "affines": [[_format_rational(m), _format_rational(b)] for m, b in r.affines],
            "left": _format_rational(r.left, inf_label="-inf"),
            "right": _format_rational(r.right, inf_label="+inf"),
        }
        for r in regions
    ]
    body = {
        "network": {
            "hidden_activation": "relu",
            "hidden_bias": b1,
            "hidden_weight": w1,
            "output_bias": b2,
            "output_weight": w2,
        },
        "schema": "vld.snc-relu-capsule.v1",
        "source_atlas": canonical_atlas,
        "source_atlas_digest": domain_digest("vld.semantic-atlas.v1", canonical_atlas),
    }
    capsule = {**body, "capsule_digest": domain_digest("vld.snc-relu-capsule.v1", body)}
    # Self-check via independent verifier before returning
    verify_compiled_capsule(capsule)
    return capsule


def evaluate_compiled_capsule(capsule: Mapping[str, Any], x: int | float | Fraction | str) -> list[Fraction]:
    """Evaluate a compiled rational ReLU capsule at exact rational/numeric input `x`."""
    if isinstance(x, bool):
        raise ProceduralModelError("Boolean input rejected in evaluate_compiled_capsule.")
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ProceduralModelError("Non-finite float input rejected in evaluate_compiled_capsule.")
        qx = Fraction(str(x))
    else:
        try:
            qx = Fraction(x)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise ProceduralModelError(f"Invalid rational input {x!r}: {exc}") from exc
    net = capsule["network"]
    w1 = [[_parse_canonical_rational(v) for v in row] for row in net["hidden_weight"]]
    b1 = [_parse_canonical_rational(v) for v in net["hidden_bias"]]
    w2 = [[_parse_canonical_rational(v) for v in row] for row in net["output_weight"]]
    b2 = [_parse_canonical_rational(v) for v in net["output_bias"]]

    hidden = [max(Fraction(0), row[0] * qx + bias) for row, bias in zip(w1, b1)]
    return [
        sum((w_ij * h_j for w_ij, h_j in zip(out_row, hidden)), start=out_bias)
        for out_row, out_bias in zip(w2, b2)
    ]


def verify_compiled_capsule(capsule: Mapping[str, Any] | str | bytes) -> dict[str, Any]:
    r"""Independently verify whole-domain symbolic equivalence of a serialized ReLU capsule without compiler internals."""
    try:
        data = parse_strict_json(capsule) if isinstance(capsule, (str, bytes)) else dict(capsule)
    except Exception as exc:
        raise ProceduralModelError(f"Malformed capsule JSON: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("schema") != "vld.snc-relu-capsule.v1":
        raise ProceduralModelError("Unsupported capsule schema.")
    supplied_digest = data.get("capsule_digest")
    body = {k: v for k, v in data.items() if k != "capsule_digest"}
    if supplied_digest != domain_digest("vld.snc-relu-capsule.v1", body):
        raise ProceduralModelError("Capsule digest mismatch.")

    if "source_atlas" not in data or "network" not in data or not isinstance(data["network"], Mapping):
        raise ProceduralModelError("Capsule missing required 'source_atlas' or 'network' mapping.")
    atlas = data["source_atlas"]
    if data.get("source_atlas_digest") != domain_digest("vld.semantic-atlas.v1", atlas):
        raise ProceduralModelError("Source atlas digest mismatch.")

    net = data["network"]
    if net.get("hidden_activation") != "relu":
        raise ProceduralModelError("Only exact ReLU hidden activation is verifiable over Q.")
    for req_key in ("hidden_weight", "hidden_bias", "output_weight", "output_bias"):
        if req_key not in net or not isinstance(net[req_key], (list, tuple)):
            raise ProceduralModelError(f"Capsule network missing list field {req_key!r}.")
    w1 = [[_parse_canonical_rational(v) for v in row] for row in net["hidden_weight"]]
    b1 = [_parse_canonical_rational(v) for v in net["hidden_bias"]]
    w2 = [[_parse_canonical_rational(v) for v in row] for row in net["output_weight"]]
    b2 = [_parse_canonical_rational(v) for v in net["output_bias"]]

    if any(len(row) != 1 for row in w1) or len(w1) != len(b1):
        raise ProceduralModelError("Invalid 1D hidden layer geometry.")
    if len(w2) != len(b2) or any(len(row) != len(w1) for row in w2):
        raise ProceduralModelError("Invalid output layer geometry.")

    # Extract all candidate kink roots {-b_i / w_i} from hidden units with w_i != 0
    roots = sorted({-b_i / row[0] for row, b_i in zip(w1, b1) if row[0] != 0})
    # Partition (-inf, +inf) into elementary intervals and merge adjacent intervals with identical affine law
    elementary_bounds: list[tuple[Fraction | None, Fraction | None]] = []
    if not roots:
        elementary_bounds.append((None, None))
    else:
        elementary_bounds.append((None, roots[0]))
        for r_a, r_b in zip(roots, roots[1:]):
            elementary_bounds.append((r_a, r_b))
        elementary_bounds.append((roots[-1], None))

    reconstructed: list[tuple[Fraction | None, Fraction | None, tuple[tuple[Fraction, Fraction], ...]]] = []
    for left, right in elementary_bounds:
        # Interior test point strictly inside (left, right) determines exact ReLU active set!
        if left is None and right is None:
            probe = Fraction(0)
        elif left is None:
            assert right is not None
            probe = right - 1
        elif right is None:
            probe = left + 1
        else:
            probe = (left + right) / 2

        active_slopes: list[Fraction] = []
        active_biases: list[Fraction] = []
        for row, bi in zip(w1, b1):
            pre = row[0] * probe + bi
            if pre > 0:
                active_slopes.append(row[0])
                active_biases.append(bi)
            else:
                active_slopes.append(Fraction(0))
                active_biases.append(Fraction(0))

        cell_affines: list[tuple[Fraction, Fraction]] = []
        for out_row, out_b in zip(w2, b2):
            m_ch = sum((w_ij * s_j for w_ij, s_j in zip(out_row, active_slopes)), start=Fraction(0))
            b_ch = sum((w_ij * c_j for w_ij, c_j in zip(out_row, active_biases)), start=out_b)
            cell_affines.append((m_ch, b_ch))
        aff_tup = tuple(cell_affines)
        if reconstructed and reconstructed[-1][2] == aff_tup:
            prev_l, _, prev_aff = reconstructed[-1]
            reconstructed[-1] = (prev_l, right, prev_aff)
        else:
            reconstructed.append((left, right, aff_tup))

    reconstructed_json = [
        {
            "affines": [[_format_rational(m), _format_rational(b)] for m, b in affs],
            "left": _format_rational(l_pt, inf_label="-inf"),
            "right": _format_rational(r_pt, inf_label="+inf"),
        }
        for l_pt, r_pt, affs in reconstructed
    ]
    if reconstructed_json != atlas:
        raise ProceduralModelError("Reconstructed neural piecewise-affine law disagrees with declared source_atlas.")
    return {
        "capsule_digest": supplied_digest,
        "elementary_cells_checked": len(elementary_bounds),
        "reconstructed_regions": len(reconstructed_json),
        "status": "VERIFIED_WHOLE_DOMAIN",
    }


def _find_strict_polyhedron_interior(
    constraints: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    dim: int,
) -> tuple[Fraction, ...] | None:
    r"""Find an exact interior witness $x \in \mathbb{Q}^d$ satisfying $\langle a_q, x \rangle + \delta_q > 0$ via Fourier-Motzkin."""
    if dim == 0:
        return () if all(delta > 0 for _, delta in constraints) else None

    zero_c: list[tuple[tuple[Fraction, ...], Fraction]] = []
    lowers: list[tuple[tuple[Fraction, ...], Fraction]] = []
    uppers: list[tuple[tuple[Fraction, ...], Fraction]] = []
    seen_proj: set[tuple[tuple[Fraction, ...], Fraction]] = set()

    def _add_proj(vec: tuple[Fraction, ...], d_val: Fraction) -> None:
        # Scale by positive L_inf norm when non-zero to deduplicate positive scalar multiples!
        scale = max((abs(c) for c in vec), default=Fraction(0))
        key = (tuple(c / scale for c in vec), d_val / scale) if scale > 0 else (vec, Fraction(1 if d_val > 0 else (-1 if d_val < 0 else 0)))
        if key not in seen_proj:
            seen_proj.add(key)
            zero_c.append((vec, d_val))

    for a_vec, delta in constraints:
        last = a_vec[dim - 1]
        prefix = a_vec[: dim - 1]
        if last == 0:
            if all(c == 0 for c in prefix) and delta <= 0:
                return None
            _add_proj(prefix, delta)
        elif last > 0:
            lowers.append((tuple(-c / last for c in prefix), -delta / last))
        else:
            uppers.append((tuple(-c / last for c in prefix), -delta / last))

    for l_vec, l_c in lowers:
        for u_vec, u_c in uppers:
            diff_vec = tuple(u_i - l_i for u_i, l_i in zip(u_vec, l_vec))
            diff_c = u_c - l_c
            if all(c == 0 for c in diff_vec) and diff_c <= 0:
                return None
            _add_proj(diff_vec, diff_c)

    sub_x = _find_strict_polyhedron_interior(zero_c, dim - 1)
    if sub_x is None:
        return None

    lb = max((sum((c * x_i for c, x_i in zip(l_vec, sub_x)), start=l_c) for l_vec, l_c in lowers), default=None)
    ub = min((sum((c * x_i for c, x_i in zip(u_vec, sub_x)), start=u_c) for u_vec, u_c in uppers), default=None)
    if lb is not None and ub is not None:
        if not (lb < ub):
            return None
        coord = (lb + ub) / 2
    elif lb is not None:
        coord = lb + 1
    elif ub is not None:
        coord = ub - 1
    else:
        coord = Fraction(0)
    return (*sub_x, coord)


def _split_open_chamber(
    ineqs: list[tuple[tuple[Fraction, ...], Fraction]],
    witness: tuple[Fraction, ...],
    n_vec: tuple[Fraction, ...],
    beta: Fraction,
    dim: int,
) -> list[tuple[list[tuple[tuple[Fraction, ...], Fraction]], tuple[Fraction, ...]]]:
    r"""Incrementally split open polyhedron $C = \{x : A x + \delta > 0\}$ by hyperplane $\langle n, x \rangle + \beta = 0$.

    Uses two exact convex-geometric accelerations before invoking Fourier-Motzkin:
      1. **Normal-Ray Witness Crossing:** Walks from $w \in C$ along $\pm n$ to cross $h(x)=0$ in $O(|C|\cdot d)$
         operations whenever $h$ is visible along the normal ray before hitting $\partial C$.
      2. **Open-Polyhedron Redundancy Elimination:** Because $C$ is open and $n \neq 0$,
         $C \cap \{h < 0\} = \emptyset \implies C \subseteq \{h > 0\}$, so redundant non-slicing hyperplanes
         are never appended to $C$'s constraint list.
    """
    neg_n = tuple(-v for v in n_vec)
    v0 = sum((n_j * w_j for n_j, w_j in zip(n_vec, witness)), start=beta)
    slacks = [sum((a_j * w_j for a_j, w_j in zip(a_q, witness)), start=d_q) for a_q, d_q in ineqs]

    if v0 == 0:
        # Witness lies directly on hyperplane strictly inside open chamber C: step analytically to both sides!
        pos_caps = [r_q / (-s_q) for (a_q, _), r_q in zip(ineqs, slacks) if (s_q := sum(a * b for a, b in zip(a_q, n_vec))) < 0]
        neg_caps = [r_q / s_q for (a_q, _), r_q in zip(ineqs, slacks) if (s_q := sum(a * b for a, b in zip(a_q, n_vec))) > 0]
        t_pos = (min(pos_caps) / 2) if pos_caps else Fraction(1)
        t_neg = (min(neg_caps) / 2) if neg_caps else Fraction(1)
        w_pos = tuple(w_j + t_pos * n_j for w_j, n_j in zip(witness, n_vec))
        w_neg = tuple(w_j - t_neg * n_j for w_j, n_j in zip(witness, n_vec))
        return [
            ([*ineqs, (n_vec, beta)], w_pos),
            ([*ineqs, (neg_n, -beta)], w_neg),
        ]

    if v0 > 0:
        same_h = (n_vec, beta)
        opp_h = (neg_n, -beta)
    else:
        same_h = (neg_n, -beta)
        opp_h = (n_vec, beta)

    opp_dir = opp_h[0]
    norm_sq = sum(v * v for v in opp_dir)
    t_cross = abs(v0) / norm_sq
    wall_caps = [
        r_q / (-s_q)
        for (a_q, _), r_q in zip(ineqs, slacks)
        if (s_q := sum(a * b for a, b in zip(a_q, opp_dir))) < 0
    ]
    t_max = min(wall_caps) if wall_caps else None

    if t_max is None or t_cross < t_max:
        t_star = (t_cross + Fraction(1)) if t_max is None else ((t_cross + t_max) / 2)
        w_opp: tuple[Fraction, ...] | None = tuple(w_j + t_star * d_j for w_j, d_j in zip(witness, opp_dir))
    else:
        w_opp = _find_strict_polyhedron_interior([*ineqs, opp_h], dim)

    if w_opp is None:
        # Hyperplane does not intersect open polyhedron C; same_h > 0 is strictly redundant on C!
        return [(ineqs, witness)]
    return [
        ([*ineqs, same_h], witness),
        ([*ineqs, opp_h], w_opp),
    ]


def compile_multivariate_hinge_relu(
    *,
    input_dim: int,
    base_linear: Sequence[Sequence[str]],
    base_bias: Sequence[str],
    hinges: Sequence[Mapping[str, Any]] = (),
    max_hinges: int = 256,
) -> dict[str, Any]:
    r"""Compile a multivariate polyhedral hinge specification $f: \mathbb{Q}^d \to \mathbb{Q}^m$ into an exact ReLU capsule."""
    if isinstance(input_dim, bool) or not isinstance(input_dim, int) or not (1 <= input_dim <= 256):
        raise ProceduralModelError("input_dim must be an integer in [1, 256].")
    out_dim = len(base_bias)
    if not (1 <= out_dim <= 64) or len(base_linear) != out_dim:
        raise ProceduralModelError("base_linear / base_bias output dimensions must match and lie in [1, 64].")
    if len(hinges) > max_hinges:
        raise ProceduralModelError(f"hinges count exceeds max_hinges={max_hinges}.")

    a_mat = [[_parse_canonical_rational(v) for v in row] for row in base_linear]
    if any(len(row) != input_dim for row in a_mat):
        raise ProceduralModelError("Each row of base_linear must have length input_dim.")
    c_vec = [_parse_canonical_rational(v) for v in base_bias]

    parsed_hinges: list[tuple[list[Fraction], Fraction, list[Fraction]]] = []
    for h in hinges:
        if not isinstance(h, Mapping) or set(h.keys()) != {"normal", "offset", "weights"}:
            raise ProceduralModelError("Each hinge requires exact keys {'normal', 'offset', 'weights'}.")
        n_vec = [_parse_canonical_rational(v) for v in h["normal"]]
        if len(n_vec) != input_dim or all(v == 0 for v in n_vec):
            raise ProceduralModelError("Hinge normal must be a non-zero vector of length input_dim.")
        beta = _parse_canonical_rational(h["offset"])
        gamma = [_parse_canonical_rational(v) for v in h["weights"]]
        if len(gamma) != out_dim:
            raise ProceduralModelError("Hinge weights length must match output dimension.")
        parsed_hinges.append((n_vec, beta, gamma))

    # Hidden units: [ReLU(x_0), ReLU(-x_0), ..., ReLU(x_{d-1}), ReLU(-x_{d-1}), ReLU(<n_1, x> + beta_1), ...]
    w1: list[list[str]] = []
    b1: list[str] = []
    for i in range(input_dim):
        pos_row = ["1" if j == i else "0" for j in range(input_dim)]
        neg_row = ["-1" if j == i else "0" for j in range(input_dim)]
        w1.extend([pos_row, neg_row])
        b1.extend(["0", "0"])
    for n_vec, beta, _ in parsed_hinges:
        w1.append([_format_rational(v) for v in n_vec])
        b1.append(_format_rational(beta))

    w2: list[list[str]] = []
    for out_i in range(out_dim):
        row: list[str] = []
        for in_j in range(input_dim):
            a_ij = a_mat[out_i][in_j]
            row.extend([_format_rational(a_ij), _format_rational(-a_ij)])
        for _, _, gamma in parsed_hinges:
            row.append(_format_rational(gamma[out_i]))
        w2.append(row)
    b2 = [_format_rational(v) for v in c_vec]

    canonical_spec = {
        "base_bias": [_format_rational(v) for v in c_vec],
        "base_linear": [[_format_rational(v) for v in row] for row in a_mat],
        "hinges": [
            {
                "normal": [_format_rational(v) for v in n_vec],
                "offset": _format_rational(beta),
                "weights": [_format_rational(v) for v in gamma],
            }
            for n_vec, beta, gamma in parsed_hinges
        ],
        "input_dim": input_dim,
    }
    body = {
        "network": {
            "hidden_activation": "relu",
            "hidden_bias": b1,
            "hidden_weight": w1,
            "output_bias": b2,
            "output_weight": w2,
        },
        "schema": "vld.snc-multivariate-relu-capsule.v1",
        "source_spec": canonical_spec,
        "source_spec_digest": domain_digest("vld.multivariate-hinge-spec.v1", canonical_spec),
    }
    capsule = {**body, "capsule_digest": domain_digest("vld.snc-multivariate-relu-capsule.v1", body)}
    verify_multivariate_capsule(capsule)
    return capsule


def evaluate_multivariate_capsule(
    capsule: Mapping[str, Any],
    x: Sequence[int | float | Fraction | str],
) -> list[Fraction]:
    r"""Evaluate a compiled multivariate rational ReLU capsule at $x \in \mathbb{Q}^d$."""
    qx: list[Fraction] = []
    for val in x:
        if isinstance(val, bool):
            raise ProceduralModelError("Boolean coordinate rejected.")
        if isinstance(val, float):
            if not math.isfinite(val):
                raise ProceduralModelError("Non-finite float coordinate rejected.")
            qx.append(Fraction(str(val)))
        else:
            qx.append(Fraction(val))
    net = capsule["network"]
    w1 = [[_parse_canonical_rational(v) for v in row] for row in net["hidden_weight"]]
    b1 = [_parse_canonical_rational(v) for v in net["hidden_bias"]]
    w2 = [[_parse_canonical_rational(v) for v in row] for row in net["output_weight"]]
    b2 = [_parse_canonical_rational(v) for v in net["output_bias"]]
    if not w1 or len(qx) != len(w1[0]):
        raise ProceduralModelError("Input vector length disagrees with multivariate capsule input_dim.")

    hidden = [
        max(Fraction(0), sum((w_ij * x_j for w_ij, x_j in zip(row, qx)), start=bias))
        for row, bias in zip(w1, b1)
    ]
    return [
        sum((w_ij * h_j for w_ij, h_j in zip(out_row, hidden)), start=out_bias)
        for out_row, out_bias in zip(w2, b2)
    ]


def _construct_ridge_counterexample(
    *,
    d: int,
    out_i: int,
    const_diff: Fraction,
    lin_diff: Sequence[Fraction],
    ridge_diff: Mapping[tuple[tuple[Fraction, ...], Fraction], Fraction],
    all_hyperplanes: Sequence[tuple[tuple[Fraction, ...], Fraction]],
    eval_diff_i: Callable[[tuple[Fraction, ...]], Fraction],
) -> tuple[Fraction, ...]:
    r"""Construct an exact rational witness $x^* \in \mathbb{Q}^d$ satisfying $f_{\text{net},i}(x^*) \neq f_{\text{spec},i}(x^*)$."""
    zero_pt = tuple(Fraction(0) for _ in range(d))
    if eval_diff_i(zero_pt) != 0:
        return zero_pt

    if not ridge_diff:
        # Pure affine discrepancy: since eval_diff_i(0) == 0, some lin_diff[j] != 0
        j_nz = next((j for j, g_j in enumerate(lin_diff) if g_j != 0), 0)
        unit_pt = tuple(Fraction(1 if j == j_nz else 0) for j in range(d))
        if eval_diff_i(unit_pt) != 0:
            return unit_pt
        raise ProceduralModelError(f"Channel {out_i} failed affine ridge consistency check.")

    # Select canonical hyperplane H* = (a*, delta*) with non-zero absolute-value ridge residual c_{i, r*} != 0
    (a_star, delta_star), _c_star = next(iter(ridge_diff.items()))
    j_star = next(j for j, v in enumerate(a_star) if v != 0)  # a_star[j_star] == 1 by canonicalization
    p0 = [Fraction(0)] * d
    p0[j_star] = -delta_star

    other_hps = [hp for hp in all_hyperplanes if hp != (a_star, delta_star)]
    if d == 1 or not other_hps:
        p = tuple(p0)
    else:
        free_coords = [k for k in range(d) if k != j_star]
        # Non-parallel hyperplanes induce non-zero linear forms q_s over free_coords
        forms: list[list[Fraction]] = []
        for a_s, _ in other_hps:
            if a_s != a_star:
                forms.append([a_s[k] - a_s[j_star] * a_star[k] for k in free_coords])

        # Deterministically pick tiny integer u in Z^{d-1} avoiding every hyperplane kernel <q_s, u> == 0
        u_vals = [Fraction(0)] * len(free_coords)
        last_nz = [max(m for m, val in enumerate(q) if val != 0) for q in forms]
        accs = [Fraction(0)] * len(forms)
        for m in range(len(free_coords)):
            forbidden_u: set[Fraction] = set()
            for idx_f, q in enumerate(forms):
                if last_nz[idx_f] == m:
                    forbidden_u.add(-accs[idx_f] / q[m])
            cand_u = 0
            while Fraction(cand_u) in forbidden_u:
                cand_u += 1
            u_m = Fraction(cand_u)
            u_vals[m] = u_m
            for idx_f, q in enumerate(forms):
                accs[idx_f] += q[m] * u_m

        v_vec = [Fraction(0)] * d
        for m, k in enumerate(free_coords):
            v_vec[k] = u_vals[m]
        v_vec[j_star] = -sum((a_star[k] * v_vec[k] for k in free_coords), start=Fraction(0))

        forbidden_t: set[Fraction] = set()
        for a_s, d_s in other_hps:
            denom = sum((a_s[k] * v_vec[k] for k in range(d)), start=Fraction(0))
            if denom != 0:
                val0 = sum((a_s[k] * p0[k] for k in range(d)), start=d_s)
                forbidden_t.add(-val0 / denom)
        t_int = 0
        while Fraction(t_int) in forbidden_t:
            t_int += 1
        t_q = Fraction(t_int)
        p = tuple(p0[k] + t_q * v_vec[k] for k in range(d))

    # Step +/- eps along normal a_star without crossing any other arrangement hyperplane
    margins: list[Fraction] = [Fraction(1)]
    for a_s, d_s in other_hps:
        dir_dot = sum((a_s[k] * a_star[k] for k in range(d)), start=Fraction(0))
        if dir_dot != 0:
            dist = abs(sum((a_s[k] * p[k] for k in range(d)), start=d_s))
            margins.append(dist / abs(dir_dot))
    eps = min(margins) / 2
    p_plus = tuple(p[k] + eps * a_star[k] for k in range(d))
    p_minus = tuple(p[k] - eps * a_star[k] for k in range(d))
    for cand in (p, p_plus, p_minus):
        if eval_diff_i(cand) != 0:
            return cand
    raise ProceduralModelError(f"Channel {out_i} ridge residual non-zero at {a_star}.")


def _verify_multivariate_ridge_exact(
    *,
    d: int,
    m_out: int,
    w1: Sequence[tuple[Fraction, ...]],
    b1: Sequence[Fraction],
    w2: Sequence[Sequence[Fraction]],
    b2: Sequence[Fraction],
    a_spec: Sequence[Sequence[Fraction]],
    c_spec: Sequence[Fraction],
    hinges_spec: Sequence[tuple[tuple[Fraction, ...], Fraction, tuple[Fraction, ...]]],
    all_hyperplanes: Sequence[tuple[tuple[Fraction, ...], Fraction]],
) -> None:
    r"""Verify whole-domain $\mathbb{R}^d$ identity $f_{\text{net}} \equiv f_{\text{spec}}$ via canonical ridge basis.

    Uses $\operatorname{ReLU}(\langle w, x \rangle + b) = \frac{1}{2}(\langle w, x \rangle + b) + \frac{1}{2}|\lambda|\cdot|\langle \tilde{a}, x \rangle + \tilde{\delta}|$
    and linear independence of distinct affine hyperplane ridges $\{|\langle \tilde{a}_r, x \rangle + \tilde{\delta}_r|\}$ on $\mathbb{R}^d$.
    """
    for out_i in range(m_out):
        const_diff = b2[out_i] - c_spec[out_i]
        lin_diff = [-a_ij for a_ij in a_spec[out_i]]
        ridge_diff: dict[tuple[tuple[Fraction, ...], Fraction], Fraction] = {}

        def _accum_relu(w_vec: tuple[Fraction, ...], bias_val: Fraction, coeff: Fraction) -> None:
            nonlocal const_diff
            if coeff == 0:
                return
            first_nz = next((v for v in w_vec if v != 0), Fraction(0))
            if first_nz == 0:
                if bias_val > 0:
                    const_diff += coeff * bias_val
                return
            const_diff += (coeff * bias_val) / 2
            for col_j in range(d):
                lin_diff[col_j] += (coeff * w_vec[col_j]) / 2
            canon_hp = (tuple(v / first_nz for v in w_vec), bias_val / first_nz)
            ridge_coeff = (coeff * abs(first_nz)) / 2
            prev = ridge_diff.get(canon_hp, Fraction(0)) + ridge_coeff
            if prev == 0:
                ridge_diff.pop(canon_hp, None)
            else:
                ridge_diff[canon_hp] = prev

        for u_idx, (row, bi) in enumerate(zip(w1, b1)):
            _accum_relu(row, bi, w2[out_i][u_idx])
        for n_vec, beta, gamma in hinges_spec:
            _accum_relu(n_vec, beta, -gamma[out_i])

        if const_diff != 0 or any(v != 0 for v in lin_diff) or ridge_diff:
            def _eval_diff_channel(pt: tuple[Fraction, ...], ch: int = out_i) -> Fraction:
                net_v = b2[ch] + sum(
                    (
                        w2[ch][u_k] * max(Fraction(0), sum((w_j * x_j for w_j, x_j in zip(r_k, pt)), start=b_k))
                        for u_k, (r_k, b_k) in enumerate(zip(w1, b1))
                    ),
                    start=Fraction(0),
                )
                spec_v = c_spec[ch] + sum((a_j * x_j for a_j, x_j in zip(a_spec[ch], pt)), start=Fraction(0))
                for n_h, b_h, g_h in hinges_spec:
                    spec_v += g_h[ch] * max(Fraction(0), sum((n_j * x_j for n_j, x_j in zip(n_h, pt)), start=b_h))
                return net_v - spec_v

            witness = _construct_ridge_counterexample(
                d=d,
                out_i=out_i,
                const_diff=const_diff,
                lin_diff=lin_diff,
                ridge_diff=ridge_diff,
                all_hyperplanes=all_hyperplanes,
                eval_diff_i=_eval_diff_channel,
            )
            raise ProceduralModelError(
                f"Multivariate ReLU network disagrees with source_spec on output {out_i} at rational witness {witness}."
            )


def verify_multivariate_capsule(
    capsule: Mapping[str, Any] | str | bytes,
    *,
    max_chambers: int | None = None,
    default_chamber_limit: int = 512,
) -> dict[str, Any]:
    r"""Independently verify whole-domain $\mathbb{R}^d$ equivalence via chamber arrangement + exact ridge algebra.

    When `max_chambers` is `None` (default), arrangements up to `default_chamber_limit` are dual-verified by both
    open-polyhedron chamber enumeration and canonical hyperplane ridge algebra, while higher-dimensional
    arrangements exceeding `default_chamber_limit` complete exact whole-domain $\mathbb{R}^d$ verification via
    the algebraic ridge-independence theorem without combinatorial chamber explosion. When an explicit integer
    `max_chambers` is supplied, exceeding that chamber count raises `ProceduralModelError`.
    """
    if max_chambers is not None and (isinstance(max_chambers, bool) or not isinstance(max_chambers, int) or max_chambers < 1):
        raise ProceduralModelError("max_chambers must be a positive integer.")
    if isinstance(default_chamber_limit, bool) or not isinstance(default_chamber_limit, int) or default_chamber_limit < 1:
        raise ProceduralModelError("default_chamber_limit must be a positive integer.")
    try:
        data = parse_strict_json(capsule) if isinstance(capsule, (str, bytes)) else dict(capsule)
    except Exception as exc:
        raise ProceduralModelError(f"Malformed multivariate capsule JSON: {exc}") from exc
    if not isinstance(data, Mapping) or data.get("schema") != "vld.snc-multivariate-relu-capsule.v1":
        raise ProceduralModelError("Unsupported multivariate capsule schema.")
    supplied_digest = data.get("capsule_digest")
    body = {k: v for k, v in data.items() if k != "capsule_digest"}
    if supplied_digest != domain_digest("vld.snc-multivariate-relu-capsule.v1", body):
        raise ProceduralModelError("Multivariate capsule digest mismatch.")
    if "source_spec" not in data or "network" not in data or not isinstance(data["network"], Mapping):
        raise ProceduralModelError("Capsule missing 'source_spec' or 'network'.")

    spec = data["source_spec"]
    if data.get("source_spec_digest") != domain_digest("vld.multivariate-hinge-spec.v1", spec):
        raise ProceduralModelError("Source multivariate spec digest mismatch.")

    raw_d = spec.get("input_dim") if isinstance(spec, Mapping) else None
    if isinstance(raw_d, bool) or not isinstance(raw_d, int) or not (1 <= raw_d <= 256):
        raise ProceduralModelError("Untrusted multivariate capsule input_dim must be an integer in [1, 256].")
    d = raw_d
    a_spec = [[_parse_canonical_rational(v) for v in row] for row in spec["base_linear"]]
    c_spec = [_parse_canonical_rational(v) for v in spec["base_bias"]]
    m_out = len(c_spec)
    if m_out < 1 or len(a_spec) != m_out or any(len(row) != d for row in a_spec):
        raise ProceduralModelError("Invalid base_linear / base_bias dimensions in multivariate capsule spec.")
    hinges_spec = [
        (
            tuple(_parse_canonical_rational(v) for v in h["normal"]),
            _parse_canonical_rational(v) if isinstance((v := h["offset"]), str) else _parse_canonical_rational(v),
            tuple(_parse_canonical_rational(v) for v in h["weights"]),
        )
        for h in spec["hinges"]
    ]
    for n_vec_h, _beta_h, gamma_h in hinges_spec:
        if len(n_vec_h) != d or len(gamma_h) != m_out or all(v == 0 for v in n_vec_h):
            raise ProceduralModelError("Invalid hinge normal or output weight dimensions in multivariate capsule spec.")

    net = data["network"]
    if net.get("hidden_activation") != "relu":
        raise ProceduralModelError("Only exact ReLU hidden activation is verifiable over Q^d.")
    w1 = [tuple(_parse_canonical_rational(v) for v in row) for row in net["hidden_weight"]]
    b1 = [_parse_canonical_rational(v) for v in net["hidden_bias"]]
    w2 = [[_parse_canonical_rational(v) for v in row] for row in net["output_weight"]]
    b2 = [_parse_canonical_rational(v) for v in net["output_bias"]]
    if len(w1) != len(b1) or any(len(row) != d for row in w1) or len(w2) != m_out or any(len(r) != len(w1) for r in w2):
        raise ProceduralModelError("Invalid multivariate network dimensions.")

    # Collect canonical unique arrangement hyperplanes (first non-zero normal coordinate normalized to +1)
    hyperplanes: list[tuple[tuple[Fraction, ...], Fraction]] = []
    seen_hp: set[tuple[tuple[Fraction, ...], Fraction]] = set()
    for n_vec, beta in [*(zip(w1, b1)), *((h[0], h[1]) for h in hinges_spec)]:
        first_nz = next((v for v in n_vec if v != 0), Fraction(0))
        if first_nz == 0:
            continue
        canon = (tuple(v / first_nz for v in n_vec), beta / first_nz)
        if canon not in seen_hp:
            seen_hp.add(canon)
            hyperplanes.append(canon)

    # 1. Always run O(U * d + U * m) exact whole-domain algebraic ridge decomposition over Q^d
    _verify_multivariate_ridge_exact(
        d=d,
        m_out=m_out,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        a_spec=a_spec,
        c_spec=c_spec,
        hinges_spec=hinges_spec,
        all_hyperplanes=hyperplanes,
    )

    # 2. Dual-derivation open-polyhedron chamber enumeration when arrangement stays within chamber budget
    chamber_cap = max_chambers if max_chambers is not None else default_chamber_limit
    active_chambers: list[tuple[list[tuple[tuple[Fraction, ...], Fraction]], tuple[Fraction, ...]]] = [
        ([], tuple(Fraction(0) for _ in range(d)))
    ]
    chamber_overflow = False
    for n_vec, beta in hyperplanes:
        next_chambers: list[tuple[list[tuple[tuple[Fraction, ...], Fraction]], tuple[Fraction, ...]]] = []
        for ineqs, w_pt in active_chambers:
            next_chambers.extend(_split_open_chamber(ineqs, w_pt, n_vec, beta, d))
            if len(next_chambers) > chamber_cap:
                if max_chambers is not None:
                    raise ProceduralModelError(
                        f"Polyhedral chamber count exceeded max_chambers={max_chambers} budget."
                    )
                chamber_overflow = True
                break
        if chamber_overflow:
            break
        active_chambers = next_chambers

    chambers = [w_pt for _, w_pt in active_chambers]
    if not chambers:
        raise ProceduralModelError("Empty polyhedral domain arrangement.")

    if not chamber_overflow:
        for witness in chambers:
            # Witness lies strictly off ALL arrangement hyperplanes: compare exact local affine Jacobian & bias
            net_jac = [[Fraction(0)] * d for _ in range(m_out)]
            net_bias = list(b2)
            for u_idx, (row, bi) in enumerate(zip(w1, b1)):
                pre = sum((w_j * x_j for w_j, x_j in zip(row, witness)), start=bi)
                if pre > 0:
                    for out_i in range(m_out):
                        coeff = w2[out_i][u_idx]
                        net_bias[out_i] += coeff * bi
                        for col_j in range(d):
                            net_jac[out_i][col_j] += coeff * row[col_j]

            exp_jac = [list(r) for r in a_spec]
            exp_bias = list(c_spec)
            for n_vec, beta, gamma in hinges_spec:
                if sum((n_j * x_j for n_j, x_j in zip(n_vec, witness)), start=beta) > 0:
                    for out_i in range(m_out):
                        exp_bias[out_i] += gamma[out_i] * beta
                        for col_j in range(d):
                            exp_jac[out_i][col_j] += gamma[out_i] * n_vec[col_j]

            if net_jac != exp_jac or net_bias != exp_bias:
                raise ProceduralModelError(
                    f"Multivariate ReLU network affine law disagrees with source_spec on chamber containing {witness}."
                )
    else:
        # Partial-prefix witnesses may lie directly on un-split hyperplanes; check exact pointwise values in addition to ridge proof
        for witness in chambers[:64]:
            net_vals = [
                b2[out_i]
                + sum(
                    (
                        w2[out_i][u_idx] * max(Fraction(0), sum((w_j * x_j for w_j, x_j in zip(row, witness)), start=bi))
                        for u_idx, (row, bi) in enumerate(zip(w1, b1))
                    ),
                    start=Fraction(0),
                )
                for out_i in range(m_out)
            ]
            exp_vals = [
                c_spec[out_i] + sum((a_j * x_j for a_j, x_j in zip(a_spec[out_i], witness)), start=Fraction(0))
                for out_i in range(m_out)
            ]
            for n_vec, beta, gamma in hinges_spec:
                act = max(Fraction(0), sum((n_j * x_j for n_j, x_j in zip(n_vec, witness)), start=beta))
                if act > 0:
                    for out_i in range(m_out):
                        exp_vals[out_i] += gamma[out_i] * act
            if net_vals != exp_vals:
                raise ProceduralModelError(
                    f"Multivariate ReLU network disagrees with source_spec at witness {witness}."
                )

    return {
        "capsule_digest": supplied_digest,
        "chambers_verified": 0 if chamber_overflow else len(chambers),
        "hyperplanes": len(hyperplanes),
        "input_dim": d,
        "status": "VERIFIED_WHOLE_DOMAIN",
        "verification_method": "algebraic_ridge_exact" if chamber_overflow else "dual_chamber_and_ridge",
    }


# ============================================================================
# 2. Constructive Finite-State Machine to Neural Synthesizer
# ============================================================================


@dataclass(frozen=True, slots=True)
class CompiledNeuralFSM:
    """Zero-training-data ReLU network realizing a complete deterministic `GroundTransitionSystem`."""

    states: tuple[str, ...]
    actions: tuple[str, ...]
    channels: tuple[str, ...]
    hidden_weight: tuple[tuple[int, ...], ...]
    hidden_bias: tuple[int, ...]
    next_state_weight: tuple[tuple[int, ...], ...]
    channel_weight: tuple[tuple[int, ...], ...]
    system_fingerprint: str

    def step_neural(self, state: str, action: str) -> tuple[str, tuple[int, ...]]:
        s_idx = self.states.index(state)
        a_idx = self.actions.index(action)
        n_s, n_a = len(self.states), len(self.actions)
        x = [1 if i == s_idx else 0 for i in range(n_s)] + [1 if j == a_idx else 0 for j in range(n_a)]
        h = [max(0, sum(w * xv for w, xv in zip(row, x)) + b) for row, b in zip(self.hidden_weight, self.hidden_bias)]
        next_s_vec = [sum(w * hv for w, hv in zip(row, h)) for row in self.next_state_weight]
        if sum(next_s_vec) != 1 or any(v not in (0, 1) for v in next_s_vec):
            raise ProceduralModelError("CompiledNeuralFSM violated one-hot successor invariant.")
        target_state = self.states[next_s_vec.index(1)]
        ch_out = tuple(sum(w * hv for w, hv in zip(row, h)) for row in self.channel_weight)
        return target_state, ch_out


def compile_fsm_to_neural(system: GroundTransitionSystem) -> CompiledNeuralFSM:
    """Compile a complete deterministic `GroundTransitionSystem` into an exact indicator-ReLU network."""
    if not system.complete:
        raise ProceduralModelError("compile_fsm_to_neural requires a complete deterministic GroundTransitionSystem.")
    n_s = len(system.states)
    n_a = len(system.actions)
    n_c = len(system.channels)
    s_map = {s: i for i, s in enumerate(system.states)}

    w_hidden: list[tuple[int, ...]] = []
    b_hidden: list[int] = []
    w_next = [[0] * (n_s * n_a) for _ in range(n_s)]
    w_chan = [[0] * (n_s * n_a) for _ in range(n_c)]

    edges = {(obs.source, obs.action): obs.outcomes[0] for obs in system.observations}
    unit_idx = 0
    for i, s_name in enumerate(system.states):
        for j, a_name in enumerate(system.actions):
            row = [0] * (n_s + n_a)
            row[i] = 1
            row[n_s + j] = 1
            w_hidden.append(tuple(row))
            b_hidden.append(-1)
            out = edges[(s_name, a_name)]
            w_next[s_map[out.target]][unit_idx] = 1
            for c_idx, c_val in enumerate(out.output):
                w_chan[c_idx][unit_idx] = c_val
            unit_idx += 1

    compiled = CompiledNeuralFSM(
        states=system.states,
        actions=system.actions,
        channels=system.channels,
        hidden_weight=tuple(w_hidden),
        hidden_bias=tuple(b_hidden),
        next_state_weight=tuple(tuple(r) for r in w_next),
        channel_weight=tuple(tuple(r) for r in w_chan),
        system_fingerprint=system.fingerprint,
    )
    # Whole-domain verification over every (state, action) pair!
    for s_name in system.states:
        for a_name in system.actions:
            direct = system.step(s_name, a_name)
            ns, nout = compiled.step_neural(s_name, a_name)
            if ns != direct.target or nout != direct.output:
                raise ProceduralModelError("CompiledNeuralFSM failed whole-domain equivalence check.")
    return compiled


# ============================================================================
# 3. Function-Preserving Neural Block Morphisms (PNF)
# ============================================================================


def _f32_round(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(val)))[0]


@dataclass(frozen=True, slots=True)
class FeedForwardBlock:
    r"""Explicit binary32 two-layer neural block $\sigma(x W_{\text{up}} + b_{\text{up}}) W_{\text{down}} + b_{\text{down}}$."""

    up_weight: tuple[tuple[float, ...], ...]  # (in_dim, hidden_dim)
    up_bias: tuple[float, ...]  # (hidden_dim,)
    down_weight: tuple[tuple[float, ...], ...]  # (hidden_dim, out_dim)
    down_bias: tuple[float, ...]  # (out_dim,)
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.activation not in ("relu", "gelu_tanh"):
            raise ProceduralModelError(f"Unsupported block activation {self.activation!r}.")
        in_d = len(self.up_weight)
        if in_d == 0:
            raise ProceduralModelError("up_weight cannot be empty.")
        hid_d = len(self.up_weight[0])
        out_d = len(self.down_bias)
        if hid_d == 0 or out_d == 0:
            raise ProceduralModelError("hidden_dim and out_dim must be positive.")
        if (
            any(len(r) != hid_d for r in self.up_weight)
            or len(self.up_bias) != hid_d
            or len(self.down_weight) != hid_d
            or any(len(r) != out_d for r in self.down_weight)
        ):
            raise ProceduralModelError("FeedForwardBlock dimensions do not compose.")
        # Normalize every scalar to exact IEEE 754 binary32
        uw = tuple(tuple(_f32_round(v) for v in row) for row in self.up_weight)
        ub = tuple(_f32_round(v) for v in self.up_bias)
        dw = tuple(tuple(_f32_round(v) for v in row) for row in self.down_weight)
        db = tuple(_f32_round(v) for v in self.down_bias)
        for seq in (*uw, ub, *dw, db):
            if any(not math.isfinite(v) for v in seq):
                raise ProceduralModelError("Non-finite binary32 weight in FeedForwardBlock.")
        object.__setattr__(self, "up_weight", uw)
        object.__setattr__(self, "up_bias", ub)
        object.__setattr__(self, "down_weight", dw)
        object.__setattr__(self, "down_bias", db)

    @property
    def dims(self) -> tuple[int, int, int]:
        return len(self.up_weight), len(self.up_bias), len(self.down_bias)

    @property
    def digest(self) -> str:
        return domain_digest(
            "vld.ffn-block.v1",
            {
                "activation": self.activation,
                "dims": list(self.dims),
                "down_bias_sha": hashlib.sha256(encode_f32_le(self.down_bias)).hexdigest(),
                "down_weight_sha": hashlib.sha256(
                    encode_f32_le([v for r in self.down_weight for v in r])
                ).hexdigest(),
                "up_bias_sha": hashlib.sha256(encode_f32_le(self.up_bias)).hexdigest(),
                "up_weight_sha": hashlib.sha256(encode_f32_le([v for r in self.up_weight for v in r])).hexdigest(),
            },
        )

    def evaluate(self, x_batch: Sequence[Sequence[float]]) -> list[list[float]]:
        in_d, hid_d, out_d = self.dims
        out: list[list[float]] = []
        for x in x_batch:
            if len(x) != in_d:
                raise ProceduralModelError("Input dimension mismatch.")
            h_vec: list[float] = []
            for j in range(hid_d):
                z = math.fsum(float(x[i]) * self.up_weight[i][j] for i in range(in_d)) + self.up_bias[j]
                if self.activation == "relu":
                    h_vec.append(max(0.0, z))
                else:
                    h_vec.append(0.5 * z * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * (z**3)))))
            y = [
                math.fsum(h_vec[j] * self.down_weight[j][k] for j in range(hid_d)) + self.down_bias[k]
                for k in range(out_d)
            ]
            out.append(y)
        return out


def widen_block_dyadic(source: FeedForwardBlock, target_width: int) -> tuple[FeedForwardBlock, dict[str, Any]]:
    r"""Widen hidden dimension ($H < H' \le 2H$) via exact dyadic coefficient halving with underflow refusal."""
    in_d, hid_d, out_d = source.dims
    if isinstance(target_width, bool) or not isinstance(target_width, int) or not (hid_d < target_width <= 2 * hid_d):
        raise ProceduralModelError(f"target_width must satisfy {hid_d} < target_width <= {2 * hid_d}.")

    dup_count = target_width - hid_d
    parents = list(range(hid_d)) + list(range(dup_count))
    multiplicity = [2 if j < dup_count else 1 for j in range(hid_d)]

    new_up_w = tuple(tuple(source.up_weight[i][p] for p in parents) for i in range(in_d))
    new_up_b = tuple(source.up_bias[p] for p in parents)
    new_down_w_rows: list[tuple[float, ...]] = []
    for p in parents:
        mult = multiplicity[p]
        row: list[float] = []
        for k in range(out_d):
            orig = source.down_weight[p][k]
            scaled = _f32_round(orig / mult)
            if orig != 0.0 and abs(scaled) < _MIN_NORMAL_F32:
                raise ProceduralModelError("Dyadic halving underflowed into binary32 subnormal/zero.")
            if scaled * mult != orig:
                raise ProceduralModelError("Dyadic halving failed exact binary32 conservation.")
            row.append(scaled)
        new_down_w_rows.append(tuple(row))

    child = FeedForwardBlock(
        up_weight=new_up_w,
        up_bias=new_up_b,
        down_weight=tuple(new_down_w_rows),
        down_bias=source.down_bias,
        activation=source.activation,
    )
    cert_body = {
        "child_digest": child.digest,
        "morphism": "dyadic_net2wider",
        "parent_digest": source.digest,
        "parent_map": parents,
        "schema": "vld.morphism-certificate.v1",
    }
    cert = {**cert_body, "certificate_digest": domain_digest("vld.morphism-certificate.v1", cert_body)}
    verify_widening_certificate(source, child, cert)
    return child, cert


def verify_widening_certificate(
    parent: FeedForwardBlock,
    child: FeedForwardBlock,
    certificate: Mapping[str, Any],
) -> None:
    """Independently verify exact binary32 incoming duplication and outgoing sum conservation."""
    body = {k: v for k, v in certificate.items() if k != "certificate_digest"}
    if certificate.get("certificate_digest") != domain_digest("vld.morphism-certificate.v1", body):
        raise ProceduralModelError("Widening certificate digest mismatch.")
    if certificate.get("parent_digest") != parent.digest or certificate.get("child_digest") != child.digest:
        raise ProceduralModelError("Widening certificate parent/child digest mismatch.")
    if parent.activation != child.activation or parent.down_bias != child.down_bias:
        raise ProceduralModelError("Activation or output bias mutated during widening.")

    in_d, hid_d, out_d = parent.dims
    c_in, c_hid, c_out = child.dims
    parents = certificate["parent_map"]
    if c_in != in_d or c_out != out_d or len(parents) != c_hid or set(parents) != set(range(hid_d)):
        raise ProceduralModelError("Invalid parent_map coverage.")

    summed_down = [[0.0] * out_d for _ in range(hid_d)]
    for child_j, p_j in enumerate(parents):
        for i in range(in_d):
            if child.up_weight[i][child_j] != parent.up_weight[i][p_j]:
                raise ProceduralModelError("Incoming column mismatch in widened block.")
        if child.up_bias[child_j] != parent.up_bias[p_j]:
            raise ProceduralModelError("Hidden bias mismatch in widened block.")
        for k in range(out_d):
            v = child.down_weight[child_j][k]
            if v != 0.0 and abs(v) < _MIN_NORMAL_F32:
                raise ProceduralModelError("Subnormal outgoing weight detected in widened block.")
            summed_down[p_j][k] += v

    for p_j in range(hid_d):
        for k in range(out_d):
            if summed_down[p_j][k] != parent.down_weight[p_j][k]:
                raise ProceduralModelError("Outgoing weight sum failed exact conservation.")


def insert_identity_depth(dim: int) -> FeedForwardBlock:
    r"""Construct an exact $2d$-width ReLU identity block satisfying $f(x) = \operatorname{ReLU}(x) - \operatorname{ReLU}(-x) = x$."""
    if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
        raise ProceduralModelError("dim must be a positive integer.")
    up_w = tuple(
        tuple((1.0 if j == i else (-1.0 if j == dim + i else 0.0)) for j in range(2 * dim))
        for i in range(dim)
    )
    up_b = tuple(0.0 for _ in range(2 * dim))
    down_w = tuple(
        tuple((1.0 if j == k else (-1.0 if j == dim + k else 0.0)) for k in range(dim))
        for j in range(2 * dim)
    )
    down_b = tuple(0.0 for _ in range(dim))
    return FeedForwardBlock(up_w, up_b, down_w, down_b, activation="relu")


def permute_hidden_units(source: FeedForwardBlock, permutation: Sequence[int]) -> FeedForwardBlock:
    """Apply compensated hidden-unit permutation symmetry preserving exact real outputs."""
    in_d, hid_d, _ = source.dims
    perm = [int(p) for p in permutation]
    if len(perm) != hid_d or sorted(perm) != list(range(hid_d)):
        raise ProceduralModelError("permutation must be a bijection of range(hidden_dim).")
    new_up_w = tuple(tuple(source.up_weight[i][p] for p in perm) for i in range(in_d))
    new_up_b = tuple(source.up_bias[p] for p in perm)
    new_down_w = tuple(source.down_weight[p] for p in perm)
    return FeedForwardBlock(new_up_w, new_up_b, new_down_w, source.down_bias, activation=source.activation)


def rescale_block_dyadic(
    source: FeedForwardBlock,
    exponents: Sequence[int],
) -> tuple[FeedForwardBlock, dict[str, Any]]:
    r"""Apply positive-homogeneous dyadic rescaling $W_{:,j} \mapsto 2^{e_j} W_{:,j},\; V_{j,:} \mapsto 2^{-e_j} V_{j,:}$ on ReLU blocks."""
    if source.activation != "relu":
        raise ProceduralModelError("Compensated positive rescaling symmetry requires positive-homogeneous 'relu' activation.")
    in_d, hid_d, out_d = source.dims
    exps = [int(e) for e in exponents]
    if any(isinstance(e, bool) for e in exponents) or len(exps) != hid_d or any(abs(e) > 60 for e in exps):
        raise ProceduralModelError("exponents must contain exact integers in [-60, 60] of length hidden_dim.")

    scales_up = [math.ldexp(1.0, e) for e in exps]
    scales_down = [math.ldexp(1.0, -e) for e in exps]

    def _checked_scale(val: float, factor: float, inv_factor: float) -> float:
        if val == 0.0:
            return 0.0
        cand = val * factor
        if not math.isfinite(cand):
            raise ProceduralModelError("Dyadic rescaling overflowed IEEE 754 binary32.")
        rounded = _f32_round(cand)
        if not math.isfinite(rounded) or abs(rounded) < _MIN_NORMAL_F32 or _f32_round(rounded * inv_factor) != val:
            raise ProceduralModelError("Dyadic rescaling overflowed or underflowed into binary32 subnormal.")
        return rounded

    new_up_w = tuple(
        tuple(_checked_scale(source.up_weight[i][j], scales_up[j], scales_down[j]) for j in range(hid_d))
        for i in range(in_d)
    )
    new_up_b = tuple(_checked_scale(source.up_bias[j], scales_up[j], scales_down[j]) for j in range(hid_d))
    new_down_w = tuple(
        tuple(_checked_scale(source.down_weight[j][k], scales_down[j], scales_up[j]) for k in range(out_d))
        for j in range(hid_d)
    )
    child = FeedForwardBlock(new_up_w, new_up_b, new_down_w, source.down_bias, activation="relu")
    cert_body = {
        "child_digest": child.digest,
        "exponents": exps,
        "morphism": "dyadic_positive_rescale",
        "parent_digest": source.digest,
        "schema": "vld.morphism-certificate.v1",
    }
    cert = {**cert_body, "certificate_digest": domain_digest("vld.morphism-certificate.v1", cert_body)}
    return child, cert


# ============================================================================
# 4. Generative Weight Grounds (MWS-002: Structural-Memory & Delta Recipes)
# ============================================================================


@dataclass(frozen=True, slots=True)
class TensorRecipe:
    """Byte-exact chunk-committed tensor synthesis recipe (`stored`, `replacement`, `structural-law`)."""

    space_digest: str
    tensor_name: str
    kind: str
    target_digest: str
    byte_size: int
    chunk_size: int
    chunk_digests: tuple[str, ...]
    body: Mapping[str, Any]

    @property
    def identity(self) -> str:
        return domain_digest(
            "vld.tensor-recipe.v1",
            {
                "body": dict(self.body),
                "byte_size": self.byte_size,
                "chunk_digests": list(self.chunk_digests),
                "chunk_size": self.chunk_size,
                "kind": self.kind,
                "space_digest": self.space_digest,
                "target_digest": self.target_digest,
                "tensor_name": self.tensor_name,
            },
        )


class GenerativeWeightGround:
    """Synthesize and chunk-verify model tensors from Structural Memory laws or sparse snapshot deltas."""

    @staticmethod
    def build_structural_law_recipe(
        space: ModelParameterSpace,
        tensor_name: str,
        registry: StructuralMemoryRegistry,
        *,
        memory_id: str,
        revision: int,
        coordinate_start: int = 0,
        chunk_size: int = 256,
    ) -> TensorRecipe:
        t = space.tensor(tensor_name)
        if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
            raise ProceduralModelError("chunk_size must be a power of two.")
        spec, cert, _ = registry.active_record(memory_id)
        if spec.revision != revision:
            raise ProceduralModelError("Requested revision is not the active Structural Memory revision.")

        whole = hashlib.sha256()
        c_digests: list[str] = []
        offset = 0
        while offset < t.byte_size:
            length = min(chunk_size, t.byte_size - offset)
            vals, _ = registry.read_range(memory_id, coordinate_start + offset, length, revision=revision)
            if any(not (0 <= v <= 255) for v in vals):
                raise ProceduralModelError("Structural Memory law values must lie in Z/256Z [0..255] for byte synthesis.")
            raw = bytes(vals)
            whole.update(raw)
            c_digests.append(hashlib.sha256(raw).hexdigest())
            offset += length

        return TensorRecipe(
            space_digest=space.identity,
            tensor_name=tensor_name,
            kind="structural-law",
            target_digest=whole.hexdigest(),
            byte_size=t.byte_size,
            chunk_size=chunk_size,
            chunk_digests=tuple(c_digests),
            body={
                "certificate_digest": cert.digest,
                "context_digest": spec.context_digest,
                "coordinate_start": coordinate_start,
                "memory_id": memory_id,
                "revision": revision,
                "spec_digest": spec.digest,
            },
        )

    @staticmethod
    def build_stored_recipe(
        space: ModelParameterSpace,
        tensor_name: str,
        store: PagedObjectStore,
        *,
        source_snapshot_id: str,
        snapshot: ModelWeightSnapshot,
        chunk_size: int = 256,
    ) -> TensorRecipe:
        t = space.tensor(tensor_name)
        if snapshot.space.identity != space.identity:
            raise ProceduralModelError("Snapshot parameter space disagrees with space.")
        if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
            raise ProceduralModelError("chunk_size must be a power of two.")
        m = snapshot.manifests[tensor_name]
        whole = hashlib.sha256()
        c_digests: list[str] = []
        offset = 0
        while offset < t.byte_size:
            length = min(chunk_size, t.byte_size - offset)
            raw = store.read_range(m, offset, length)
            whole.update(raw)
            c_digests.append(hashlib.sha256(raw).hexdigest())
            offset += length
        return TensorRecipe(
            space_digest=space.identity,
            tensor_name=tensor_name,
            kind="stored",
            target_digest=whole.hexdigest(),
            byte_size=t.byte_size,
            chunk_size=chunk_size,
            chunk_digests=tuple(c_digests),
            body={"source_snapshot": source_snapshot_id},
        )

    @staticmethod
    def build_replacement_recipe(
        space: ModelParameterSpace,
        tensor_name: str,
        store: PagedObjectStore,
        *,
        base_snapshot_id: str,
        base_snapshot: ModelWeightSnapshot,
        replacement_chunks: Mapping[int, bytes],
        chunk_size: int = 256,
    ) -> TensorRecipe:
        """Synthesize a chunk-committed sparse delta recipe replacing selected chunks over `base_snapshot`."""
        t = space.tensor(tensor_name)
        if base_snapshot.space.identity != space.identity:
            raise ProceduralModelError("Base snapshot parameter space disagrees with space.")
        if chunk_size <= 0 or (chunk_size & (chunk_size - 1)) != 0:
            raise ProceduralModelError("chunk_size must be a power of two.")
        total_chunks = (t.byte_size + chunk_size - 1) // chunk_size
        m = base_snapshot.manifests[tensor_name]
        repl_hex: dict[str, str] = {}
        for c_idx, raw_patch in sorted(replacement_chunks.items()):
            if isinstance(c_idx, bool) or not isinstance(c_idx, int) or not (0 <= c_idx < total_chunks):
                raise ProceduralModelError(f"Replacement chunk index {c_idx!r} out of bounds.")
            exp_len = min(chunk_size, t.byte_size - c_idx * chunk_size)
            if not isinstance(raw_patch, bytes) or len(raw_patch) != exp_len:
                raise ProceduralModelError(f"Replacement chunk {c_idx} byte length mismatch.")
            repl_hex[str(c_idx)] = raw_patch.hex()

        whole = hashlib.sha256()
        c_digests: list[str] = []
        for c_idx in range(total_chunks):
            offset = c_idx * chunk_size
            length = min(chunk_size, t.byte_size - offset)
            raw = replacement_chunks[c_idx] if c_idx in replacement_chunks else store.read_range(m, offset, length)
            whole.update(raw)
            c_digests.append(hashlib.sha256(raw).hexdigest())

        return TensorRecipe(
            space_digest=space.identity,
            tensor_name=tensor_name,
            kind="replacement",
            target_digest=whole.hexdigest(),
            byte_size=t.byte_size,
            chunk_size=chunk_size,
            chunk_digests=tuple(c_digests),
            body={"base_snapshot": base_snapshot_id, "replacements_hex": repl_hex},
        )

    @staticmethod
    def read_verified_chunk(
        space: ModelParameterSpace,
        recipe: TensorRecipe,
        chunk_index: int,
        *,
        registry: StructuralMemoryRegistry | None = None,
        store: PagedObjectStore | None = None,
        source_snapshots: Mapping[str, ModelWeightSnapshot] | None = None,
    ) -> bytes:
        t = space.tensor(recipe.tensor_name)
        if recipe.space_digest != space.identity or recipe.byte_size != t.byte_size:
            raise ProceduralModelError("TensorRecipe does not match ModelParameterSpace.")
        if not (0 <= chunk_index < len(recipe.chunk_digests)):
            raise ProceduralModelError("chunk_index out of bounds.")

        offset = chunk_index * recipe.chunk_size
        length = min(recipe.chunk_size, recipe.byte_size - offset)
        b = recipe.body

        if recipe.kind == "structural-law":
            if registry is None:
                raise ProceduralModelError("StructuralMemoryRegistry required for structural-law recipe.")
            try:
                vals, rec = registry.read_range(
                    str(b["memory_id"]),
                    int(b["coordinate_start"]) + offset,
                    length,
                    revision=int(b["revision"]),
                )
            except StructuralMemoryError as exc:
                raise ProceduralModelError(f"Structural Memory read refused: {exc}") from exc
            ok, reason = registry.verify_read_receipt(rec)
            if not ok:
                raise ProceduralModelError(f"Structural Memory receipt invalidated ({reason}).")
            if rec.spec_digest != b["spec_digest"] or rec.certificate_digest != b["certificate_digest"]:
                raise ProceduralModelError("Structural Memory spec/certificate pin mismatch.")
            raw = bytes(vals)
        elif recipe.kind == "stored":
            if store is None or source_snapshots is None:
                raise ProceduralModelError("store and source_snapshots required for stored recipe.")
            snap = source_snapshots[str(b["source_snapshot"])]
            raw = store.read_range(snap.manifests[recipe.tensor_name], offset, length)
        elif recipe.kind == "replacement":
            repl_map = b.get("replacements_hex", {})
            key = str(chunk_index)
            if isinstance(repl_map, Mapping) and key in repl_map:
                raw = bytes.fromhex(str(repl_map[key]))
            else:
                if store is None or source_snapshots is None:
                    raise ProceduralModelError("store and source_snapshots required for unpatched replacement chunk.")
                snap = source_snapshots[str(b["base_snapshot"])]
                raw = store.read_range(snap.manifests[recipe.tensor_name], offset, length)
        else:
            raise ProceduralModelError(f"Unsupported recipe kind {recipe.kind!r}.")

        if hashlib.sha256(raw).hexdigest() != recipe.chunk_digests[chunk_index]:
            raise ProceduralModelError("Reconstructed tensor chunk failed SHA-256 commitment check.")
        _ = canonical_json(dict(b))
        _ = decode_f32_le(b"")
        return raw


__all__ = [
    "CompiledNeuralFSM",
    "FeedForwardBlock",
    "GenerativeWeightGround",
    "PiecewiseAffineRegion",
    "ProceduralModelError",
    "TensorRecipe",
    "compile_fsm_to_neural",
    "compile_multivariate_hinge_relu",
    "compile_piecewise_affine_relu",
    "evaluate_compiled_capsule",
    "evaluate_multivariate_capsule",
    "insert_identity_depth",
    "permute_hidden_units",
    "rescale_block_dyadic",
    "verify_compiled_capsule",
    "verify_multivariate_capsule",
    "verify_widening_certificate",
    "widen_block_dyadic",
]
