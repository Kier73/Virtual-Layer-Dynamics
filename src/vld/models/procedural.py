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
from typing import Any, Mapping, Sequence

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
    qx = Fraction(x) if not isinstance(x, float) else Fraction(str(x))
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
    data = parse_strict_json(capsule) if isinstance(capsule, (str, bytes)) else dict(capsule)
    if not isinstance(data, Mapping) or data.get("schema") != "vld.snc-relu-capsule.v1":
        raise ProceduralModelError("Unsupported capsule schema.")
    supplied_digest = data.get("capsule_digest")
    body = {k: v for k, v in data.items() if k != "capsule_digest"}
    if supplied_digest != domain_digest("vld.snc-relu-capsule.v1", body):
        raise ProceduralModelError("Capsule digest mismatch.")

    atlas = data["source_atlas"]
    if data.get("source_atlas_digest") != domain_digest("vld.semantic-atlas.v1", atlas):
        raise ProceduralModelError("Source atlas digest mismatch.")

    net = data["network"]
    if net.get("hidden_activation") != "relu":
        raise ProceduralModelError("Only exact ReLU hidden activation is verifiable over Q.")
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
    "compile_piecewise_affine_relu",
    "evaluate_compiled_capsule",
    "insert_identity_depth",
    "permute_hidden_units",
    "verify_compiled_capsule",
    "verify_widening_certificate",
    "widen_block_dyadic",
]
