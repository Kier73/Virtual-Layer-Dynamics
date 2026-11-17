r"""Core Virtual Ground specification, byte-lifting lenses, affine orbit navigation, and Pareto cost algebra.

Governing equations:
1. **Computational Ground:**
   $\mathcal{G} = (\mathcal{L}, \Sigma_{\mathcal{G}}, \Delta_{\mathcal{G}}, \mathcal{I}_{\mathcal{G}}, \mathcal{O}_{\mathcal{G}}, \mathcal{C}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}}, \nu_{\mathcal{G}})$

2. **Verified Byte Lens:**
   $\Lambda: (\mathbb{Z}/256\mathbb{Z})^N \to \Sigma_{\mathcal{G}}$ with exact round-trip invariant
   $\Lambda^{-1}(\Lambda(x)) = x$ on invertible representations.

3. **Dual-Derivation Affine Orbit Navigation ($O(\log n)$):**
   For $x_{t+1} = (a x_t + b) \bmod m$:
     - Functional monoid power: $(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1 \bmod m,\; (a_2 b_1 + b_2) \bmod m)$
     - Homogeneous matrix power:
         $\begin{pmatrix} x_n \\ 1 \end{pmatrix} \equiv \begin{pmatrix} a & b \\ 0 & 1 \end{pmatrix}^n \begin{pmatrix} x_0 \\ 1 \end{pmatrix} \pmod m$

4. **Guarded Shortcut Navigation Cost:**
   $\mathbf{C}_{\text{total}} = \mathbf{C}_{\text{lift}} + \mathbf{C}_g + \mathbf{C}_\sigma + p_{\text{exit}} \cdot \mathbf{C}_{\text{fallback}}$
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
from typing import Any, Sequence

from vld.canonical import domain_digest
from vld.triad import EpistemicStatus


class DualDerivationMismatchError(RuntimeError):
    """Raised if independent mathematical realizations disagree on an orbit jump."""


class UnderdeterminedGroundError(ValueError):
    """Raised when attempting to certify an unidentifiable Ground law from insufficient observations."""


class LensProjectionError(ValueError):
    """Raised when raw bytes fail domain or round-trip validation under a declared lens."""


@dataclass(frozen=True)
class VirtualGroundSpec:
    r"""Immutable, content-addressed Computational Ground specification $\mathcal{G}$."""

    ground_id: str
    version: str
    language: str
    state_domain: str
    transitions: tuple[str, ...]
    invariants: tuple[str, ...]
    observers: tuple[str, ...]
    cost_dimensions: tuple[str, ...] = ("latency_ms", "compute_ops", "bytes_moved", "verification_ops")
    evidence_scope: str = "declared-specification"

    def __post_init__(self) -> None:
        for field_name, scalar in (
            ("ground_id", self.ground_id),
            ("version", self.version),
            ("language", self.language),
            ("state_domain", self.state_domain),
            ("evidence_scope", self.evidence_scope),
        ):
            if not isinstance(scalar, str) or not scalar.strip():
                raise ValueError(f"{field_name} must be a non-empty str.")

        for seq_name in ("transitions", "invariants", "observers", "cost_dimensions"):
            seq_val = getattr(self, seq_name)
            if isinstance(seq_val, (str, bytes)) or not hasattr(seq_val, "__iter__"):
                raise ValueError(f"{seq_name} must be a sequence of non-empty strings, not {type(seq_val).__name__}.")
            items = tuple(seq_val)
            for item in items:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{seq_name} entries must be non-empty strings, got {item!r}.")
            object.__setattr__(self, seq_name, items)

        if not self.observers:
            raise ValueError("Ground must declare at least one observer projection.")
        if not self.cost_dimensions or len(set(self.cost_dimensions)) != len(self.cost_dimensions):
            raise ValueError("cost_dimensions must be non-empty and duplicate-free.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_dimensions": list(self.cost_dimensions),
            "evidence_scope": self.evidence_scope,
            "ground_id": self.ground_id,
            "invariants": list(self.invariants),
            "language": self.language,
            "observers": list(self.observers),
            "state_domain": self.state_domain,
            "transitions": list(self.transitions),
            "version": self.version,
        }

    @property
    def fingerprint(self) -> str:
        """Canonical domain-separated SHA-256 identity of this Ground specification."""
        return domain_digest("vld.ground-spec.v1", self.to_dict())


class LensKind(str, Enum):
    """Built-in reversible byte-to-Ground lenses over (Z/256Z)^N."""

    RAW_Z256 = "raw_z256"
    UINT_LE = "uint_le"
    UINT_BE = "uint_be"
    UTF8_EXACT = "utf8_exact"


@dataclass(frozen=True)
class LiftedByteProjection:
    """Verified result of lifting raw bytes through a declared ByteLens."""

    lens_kind: LensKind
    byte_length: int
    source_sha256: str
    value: Any
    receipt_digest: str


@dataclass(frozen=True)
class ByteLens:
    r"""Reversible coordinate lens $\Lambda: (\mathbb{Z}/256\mathbb{Z})^N \to \Sigma$."""

    kind: LensKind
    expected_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LensKind):
            raise ValueError(f"kind must be a LensKind, got {self.kind!r}.")
        if self.expected_length is not None:
            if not isinstance(self.expected_length, int) or isinstance(self.expected_length, bool) or self.expected_length < 0:
                raise ValueError("expected_length must be a non-negative int.")

    def lift(self, raw: bytes) -> LiftedByteProjection:
        """Lift raw bytes into the declared domain and verify exact round-trip invertibility."""
        if not isinstance(raw, bytes):
            raise LensProjectionError(f"ByteLens requires raw bytes, got {type(raw).__name__}")
        if self.expected_length is not None and len(raw) != self.expected_length:
            raise LensProjectionError(
                f"Expected {self.expected_length} byte(s) for {self.kind.value}, got {len(raw)}."
            )

        if self.kind is LensKind.RAW_Z256:
            value: Any = tuple(raw)
        elif self.kind is LensKind.UINT_LE:
            if len(raw) == 0:
                raise LensProjectionError("UINT_LE lens requires at least 1 byte.")
            value = int.from_bytes(raw, byteorder="little", signed=False)
        elif self.kind is LensKind.UINT_BE:
            if len(raw) == 0:
                raise LensProjectionError("UINT_BE lens requires at least 1 byte.")
            value = int.from_bytes(raw, byteorder="big", signed=False)
        elif self.kind is LensKind.UTF8_EXACT:
            try:
                value = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise LensProjectionError(f"Invalid UTF-8 byte sequence: {exc}") from exc
        else:
            raise LensProjectionError(f"Unsupported lens kind: {self.kind!r}")

        lowered = self.lower(value, byte_length=len(raw))
        if lowered != raw:
            raise LensProjectionError("Round-trip invertibility check failed.")

        source_sha256 = hashlib.sha256(raw).hexdigest()
        serializable_val = list(value) if isinstance(value, tuple) else value
        receipt_digest = domain_digest(
            "vld.byte-lens-receipt.v1",
            {
                "byte_length": len(raw),
                "expected_length": self.expected_length,
                "lens_kind": self.kind.value,
                "source_sha256": source_sha256,
                "value": serializable_val,
            },
        )
        return LiftedByteProjection(
            lens_kind=self.kind,
            byte_length=len(raw),
            source_sha256=source_sha256,
            value=value,
            receipt_digest=receipt_digest,
        )

    def lower(self, value: Any, *, byte_length: int | None = None) -> bytes:
        r"""Apply inverse map $\Lambda^{-1}: \Sigma \to (\mathbb{Z}/256\mathbb{Z})^N$."""
        if byte_length is not None and (not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length < 0):
            raise LensProjectionError("byte_length must be a non-negative int.")
        target_len = self.expected_length if self.expected_length is not None else byte_length
        if self.kind is LensKind.RAW_Z256:
            if not isinstance(value, (tuple, list)):
                raise LensProjectionError("RAW_Z256 lower expects sequence of ints in 0..255.")
            for item in value:
                if not isinstance(item, int) or isinstance(item, bool) or not (0 <= item <= 255):
                    raise LensProjectionError(f"Invalid Z/256Z element: {item!r}")
            out = bytes(value)
        elif self.kind in (LensKind.UINT_LE, LensKind.UINT_BE):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise LensProjectionError("Unsigned integer lens requires non-negative int.")
            if target_len is None or target_len <= 0:
                raise LensProjectionError("Integer lowering requires positive byte_length or expected_length.")
            order = "little" if self.kind is LensKind.UINT_LE else "big"
            try:
                out = value.to_bytes(target_len, byteorder=order, signed=False)
            except OverflowError as exc:
                raise LensProjectionError(f"Integer {value} overflows {target_len} byte(s).") from exc
        elif self.kind is LensKind.UTF8_EXACT:
            if not isinstance(value, str):
                raise LensProjectionError("UTF8_EXACT lower expects str.")
            try:
                out = value.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise LensProjectionError(f"Invalid Unicode string for UTF-8 encoding: {exc}") from exc
        else:
            raise LensProjectionError(f"Unsupported lens kind: {self.kind!r}")

        if target_len is not None and len(out) != target_len:
            raise LensProjectionError(f"Lowered length {len(out)} != expected {target_len}.")
        return out


@dataclass(frozen=True)
class OrbitJumpReceipt:
    """Verified receipt for an O(log n) affine orbit coordinate navigation."""

    law_digest: str
    initial_state: int
    coordinate: int
    value: int
    affine_multiplier: int
    affine_increment: int
    binary_steps: int


@dataclass(frozen=True)
class AffineOrbitLaw:
    r"""Admitted modular affine recurrence $x_{t+1} = (a x_t + b) \bmod m$."""

    multiplier: int
    increment: int
    modulus: int

    def __post_init__(self) -> None:
        for name, val in (("multiplier", self.multiplier), ("increment", self.increment), ("modulus", self.modulus)):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"{name} must be an exact int.")
        if self.modulus < 2:
            raise ValueError("modulus must be >= 2.")
        if not (0 <= self.multiplier < self.modulus) or not (0 <= self.increment < self.modulus):
            raise ValueError("multiplier and increment must be canonical residues in [0, modulus).")

    @classmethod
    def from_parameters(cls, multiplier: int, increment: int, modulus: int) -> AffineOrbitLaw:
        for name, val in (("multiplier", multiplier), ("increment", increment), ("modulus", modulus)):
            if not isinstance(val, int) or isinstance(val, bool):
                raise ValueError(f"{name} must be an exact int.")
        if modulus < 2:
            raise ValueError("modulus must be >= 2.")
        return cls(multiplier % modulus, increment % modulus, modulus)

    @property
    def digest(self) -> str:
        return domain_digest(
            "vld.affine-orbit-law.v1",
            {
                "increment": self.increment,
                "modulus": self.modulus,
                "multiplier": self.multiplier,
            },
        )

    def _compose_affine_jump(self, coordinate: int) -> tuple[int, int, int]:
        """Derivation A: binary exponentiation in the affine transformation monoid."""
        m = self.modulus
        acc_a, acc_b = 1, 0
        cur_a, cur_b = self.multiplier, self.increment
        n = coordinate
        steps = 0
        while n > 0:
            steps += 1
            if n & 1:
                acc_a = (cur_a * acc_a) % m
                acc_b = (cur_a * acc_b + cur_b) % m
            cur_b = (cur_a * cur_b + cur_b) % m
            cur_a = (cur_a * cur_a) % m
            n >>= 1
        return acc_a, acc_b, steps

    def _matrix_orbit_jump(self, x0: int, coordinate: int) -> int:
        """Derivation B: independent 2x2 homogeneous matrix exponentiation mod m."""
        m = self.modulus

        def mat_mul(p: tuple[int, int, int, int], q: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return (
                (p[0] * q[0] + p[1] * q[2]) % m,
                (p[0] * q[1] + p[1] * q[3]) % m,
                (p[2] * q[0] + p[3] * q[2]) % m,
                (p[2] * q[1] + p[3] * q[3]) % m,
            )

        res = (1, 0, 0, 1)
        base = (self.multiplier, self.increment, 0, 1)
        n = coordinate
        while n > 0:
            if n & 1:
                res = mat_mul(res, base)
            base = mat_mul(base, base)
            n >>= 1
        return (res[0] * (x0 % m) + res[1]) % m

    def navigate(self, initial_state: int, coordinate: int) -> OrbitJumpReceipt:
        r"""Evaluate $x_n$ in $O(\log n)$ steps with mandatory dual-derivation check."""
        if not isinstance(initial_state, int) or isinstance(initial_state, bool):
            raise ValueError("initial_state must be an int.")
        if not isinstance(coordinate, int) or isinstance(coordinate, bool) or coordinate < 0:
            raise ValueError("coordinate must be a non-negative int.")
        if not (0 <= initial_state < self.modulus):
            raise ValueError(f"initial_state {initial_state} outside [0, {self.modulus}).")

        jump_a, jump_b, steps = self._compose_affine_jump(coordinate)
        val_monoid = (jump_a * initial_state + jump_b) % self.modulus
        val_matrix = self._matrix_orbit_jump(initial_state, coordinate)
        if val_monoid != val_matrix:
            raise DualDerivationMismatchError(
                f"Dual derivation mismatch at coordinate {coordinate}: {val_monoid} != {val_matrix}"
            )
        return OrbitJumpReceipt(
            law_digest=self.digest,
            initial_state=initial_state,
            coordinate=coordinate,
            value=val_monoid,
            affine_multiplier=jump_a,
            affine_increment=jump_b,
            binary_steps=steps,
        )


@dataclass(frozen=True)
class ProgressionTraceProposal:
    """Result of inspecting an observed integer trace for modular progression structure."""

    status: EpistemicStatus
    step_delta: int | None
    inferred_modulus: int | None
    wrap_witnessed: bool
    law: AffineOrbitLaw | None
    reason: str


def propose_modular_progression(
    samples: Sequence[int],
    *,
    declared_modulus: int | None = None,
) -> ProgressionTraceProposal:
    r"""Propose or certify $x_{t+1} = (x_t + \Delta) \bmod m$ from finite observations.

    Identifiability & degrees-of-freedom rules:
      - If `declared_modulus` is omitted, BOTH a modular wrap ($x_t + \Delta \ge m \implies x_{t+1} = x_t + \Delta - m$)
        and a positive unwrapped step $\Delta > 0$ MUST be witnessed in `samples`, AND at least 4 samples
        ($N - 1 \ge 3$ transitions for 2 unknown parameters $(\Delta, m)$, giving $df = N - 3 \ge 1$
        predictive check) are required to reach `EpistemicStatus.CERTIFIED`.
      - A 3-sample wrap candidate ($df = 0$) or an unwrapped arithmetic progression (infinitely many
        compatible moduli $m > \max_t x_t$, plus $\mathbb{Z}$) is classified as `EpistemicStatus.HYPOTHESIS`
        (`law=None`), never silently promoted to a certified modular Ground.
    """
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required to assess progression structure.")
    for idx, val in enumerate(samples):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise ValueError(f"Sample at index {idx} must be a non-negative int, got {val!r}.")

    if declared_modulus is not None:
        if not isinstance(declared_modulus, int) or isinstance(declared_modulus, bool) or declared_modulus < 2:
            raise ValueError("declared_modulus must be an exact int >= 2.")
        if any(v >= declared_modulus for v in samples):
            return ProgressionTraceProposal(
                status=EpistemicStatus.AMBIGUOUS,
                step_delta=None,
                inferred_modulus=declared_modulus,
                wrap_witnessed=False,
                law=None,
                reason="Sample exceeds declared_modulus.",
            )
        step = (samples[1] - samples[0]) % declared_modulus
        if all((samples[i + 1] - samples[i]) % declared_modulus == step for i in range(len(samples) - 1)):
            wrapped = any(samples[i] + step >= declared_modulus for i in range(len(samples) - 1))
            law = AffineOrbitLaw(multiplier=1, increment=step, modulus=declared_modulus)
            return ProgressionTraceProposal(
                status=EpistemicStatus.CERTIFIED,
                step_delta=step,
                inferred_modulus=declared_modulus,
                wrap_witnessed=wrapped,
                law=law,
                reason="Consistent with caller-declared modulus.",
            )
        return ProgressionTraceProposal(
            status=EpistemicStatus.AMBIGUOUS,
            step_delta=None,
            inferred_modulus=declared_modulus,
            wrap_witnessed=False,
            law=None,
            reason="Samples violate constant modular step under declared_modulus.",
        )

    diffs = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]
    distinct_diffs = sorted(set(diffs))

    if len(distinct_diffs) == 1:
        delta = distinct_diffs[0]
        return ProgressionTraceProposal(
            status=EpistemicStatus.HYPOTHESIS,
            step_delta=delta,
            inferred_modulus=None,
            wrap_witnessed=False,
            law=None,
            reason="Unwrapped progression: modulus is unidentifiable without wrap witness or declared domain.",
        )

    if len(distinct_diffs) == 2 and distinct_diffs[0] < 0 < distinct_diffs[1]:
        neg_jump, pos_step = distinct_diffs[0], distinct_diffs[1]
        modulus = pos_step - neg_jump
        if modulus >= 2 and all(0 <= v < modulus for v in samples):
            if all((samples[i + 1] - samples[i]) % modulus == pos_step for i in range(len(samples) - 1)):
                if len(samples) < 4:
                    return ProgressionTraceProposal(
                        status=EpistemicStatus.HYPOTHESIS,
                        step_delta=pos_step,
                        inferred_modulus=modulus,
                        wrap_witnessed=True,
                        law=None,
                        reason="Wrap candidate has zero residual degrees of freedom (3 samples fit 2 parameters); at least 4 samples required for certification.",
                    )
                law = AffineOrbitLaw(multiplier=1, increment=pos_step, modulus=modulus)
                return ProgressionTraceProposal(
                    status=EpistemicStatus.CERTIFIED,
                    step_delta=pos_step,
                    inferred_modulus=modulus,
                    wrap_witnessed=True,
                    law=law,
                    reason="Unique step and modulus certified via witnessed modular wrap with positive residual degrees of freedom.",
                )

    return ProgressionTraceProposal(
        status=EpistemicStatus.AMBIGUOUS,
        step_delta=None,
        inferred_modulus=None,
        wrap_witnessed=False,
        law=None,
        reason="Samples do not match a single modular additive progression.",
    )


def certify_modular_progression(
    samples: Sequence[int],
    *,
    declared_modulus: int | None = None,
) -> AffineOrbitLaw:
    """Require an identifiable, certified modular progression law or raise UnderdeterminedGroundError."""
    proposal = propose_modular_progression(samples, declared_modulus=declared_modulus)
    if proposal.status is not EpistemicStatus.CERTIFIED or proposal.law is None:
        raise UnderdeterminedGroundError(
            f"Cannot certify modular Ground ({proposal.status.value}): {proposal.reason}"
        )
    return proposal.law


@dataclass(frozen=True)
class NavigationCostVector:
    r"""Multi-objective cost vector $\mathbf{C} = (L, E, B, R)$ without premature scalarization."""

    latency_ms: float
    compute_ops: float
    bytes_moved: float
    verification_ops: float

    def __post_init__(self) -> None:
        for name, val in (
            ("latency_ms", self.latency_ms),
            ("compute_ops", self.compute_ops),
            ("bytes_moved", self.bytes_moved),
            ("verification_ops", self.verification_ops),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be a finite real number >= 0, got {val!r}.")
            object.__setattr__(self, name, float(val))

    def __add__(self, other: object) -> NavigationCostVector:
        if not isinstance(other, NavigationCostVector):
            return NotImplemented
        return NavigationCostVector(
            latency_ms=self.latency_ms + other.latency_ms,
            compute_ops=self.compute_ops + other.compute_ops,
            bytes_moved=self.bytes_moved + other.bytes_moved,
            verification_ops=self.verification_ops + other.verification_ops,
        )

    def scale(self, factor: float) -> NavigationCostVector:
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor < 0.0:
            raise ValueError("Scale factor must be a finite real number >= 0.")
        return NavigationCostVector(
            latency_ms=self.latency_ms * factor,
            compute_ops=self.compute_ops * factor,
            bytes_moved=self.bytes_moved * factor,
            verification_ops=self.verification_ops * factor,
        )

    def pareto_dominates(self, other: NavigationCostVector) -> bool:
        """True iff self is <= other on all dimensions and strictly < on at least one."""
        dims_self = (self.latency_ms, self.compute_ops, self.bytes_moved, self.verification_ops)
        dims_other = (other.latency_ms, other.compute_ops, other.bytes_moved, other.verification_ops)
        no_worse = all(a <= b for a, b in zip(dims_self, dims_other))
        strictly_better = any(a < b for a, b in zip(dims_self, dims_other))
        return no_worse and strictly_better


def expected_shortcut_cost(
    *,
    lift_cost: NavigationCostVector,
    guard_cost: NavigationCostVector,
    shortcut_cost: NavigationCostVector,
    exit_probability: float,
    fallback_cost: NavigationCostVector,
) -> NavigationCostVector:
    r"""Evaluate $\mathbf{C}_{\text{VLD}} = \mathbf{C}_{\text{lift}} + \mathbf{C}_g + \mathbf{C}_\sigma + p_{\text{exit}} \mathbf{C}_{\text{exec}}$."""
    if (
        isinstance(exit_probability, bool)
        or not isinstance(exit_probability, (int, float))
        or not math.isfinite(exit_probability)
        or not (0.0 <= exit_probability <= 1.0)
    ):
        raise ValueError("exit_probability must be a finite real number in [0.0, 1.0].")
    return lift_cost + guard_cost + shortcut_cost + fallback_cost.scale(float(exit_probability))
