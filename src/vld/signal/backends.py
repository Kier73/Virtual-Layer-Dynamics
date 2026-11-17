r"""Multi-domain signal backends and observer-covariant algebras for `vld.signal`.

Governing equations:
1. **Cyclic Phasor Domain (`Signal8Backend`, $\mathbb{Z}/8\mathbb{Z}$):**
   Semantic symbol $k \in \mathbb{Z}/8\mathbb{Z}$ under odd-unit observer $u \in (\mathbb{Z}/8\mathbb{Z})^\times = \{1,3,5,7\}$
   is represented solely as the character phasor without storing $k$:
     $$\chi_u(k) = \exp\!\left(\frac{2\pi i u k}{8}\right), \qquad S = (z, u, \epsilon), \quad |z|=1,\; \epsilon \in [0, \pi]$$
   Observer reframing $u \to v$ raises $z$ to $m = v u^{-1} \bmod 8$:
     $$R_{u \to v}(z, u, \epsilon) = \left(\frac{z^m}{|z^m|},\; v,\; \min(\pi, m\epsilon)\right)$$

2. **Dyadic Modular Ring Domain (`ModularRingBackend`, $\mathbb{Z}/2^w\mathbb{Z}$):**
   Residue $x \in \mathbb{Z}/M\mathbb{Z}$ ($M = 2^w$) under unit observer $u \in (\mathbb{Z}/M\mathbb{Z})^\times$ is
   stored as $r = (u x) \bmod M$ with Lee-metric uncertainty radius $\delta \in [0, M/2]$.
   Observer-covariant ring multiplication satisfies:
     $$r_{a \cdot b} \equiv r_a \cdot r_b \cdot u^{-1} \equiv u (x_a x_b) \pmod M$$

3. **Exact Rational Scale Domain (`ExactRationalBackend`, $\mathbb{Q}$):**
   Value $x \in \mathbb{Q}$ under positive scale observer $u \in \mathbb{Q}_{>0}$ is stored as $q = u x$
   with exact rational semantic uncertainty $\epsilon \in \mathbb{Q}_{\ge 0}$ and zero rounding drift:
     $$q_{a + b} = q_a + R_{u_b \to u_a}(q_b), \qquad q_{a \cdot b} = \frac{q_a \cdot R_{u_b \to u_a}(q_b)}{u_a}$$
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Any, Final, Generic, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

from vld.triad import TruthDisposition

SIGNAL8_OBSERVERS: Final[tuple[int, ...]] = (1, 3, 5, 7)
TAU: Final[float] = 2.0 * math.pi
_EPS: Final[float] = 1e-12

V = TypeVar("V")


class SignalDomainError(ValueError):
    """Raised when a signal value, observer coordinate, or domain operation is invalid."""


class AmbiguousObservationError(SignalDomainError):
    """Raised when terminal symbol decoding overlaps multiple decision regions."""

    def __init__(self, candidates: tuple[int, ...], error_bound: float) -> None:
        self.candidates = candidates
        self.error_bound = error_bound
        super().__init__(
            f"Terminal observation is ambiguous across candidates={candidates} "
            f"(error_bound={error_bound:.6g})."
        )


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Explicit algebraic and physical-interface capabilities of a computational backend."""

    supports_reframe: bool
    supports_mul: bool
    supports_iq_calibration: bool
    exact_arithmetic: bool
    finite_symbol_cardinality: int | None


@dataclass(frozen=True, slots=True)
class GuardDecision:
    r"""Three-valued Kleene $\mathcal{K}_3$ guard evaluation with conservative geometric bounds."""

    disposition: TruthDisposition
    distance: float
    uncertainty: float
    tolerance: float
    lower_bound: float
    upper_bound: float

    @property
    def margin(self) -> float:
        return abs(self.distance - self.tolerance) - self.uncertainty


@runtime_checkable
class ComputationalBackend(Protocol[V]):
    """Formal backend contract decoupling machine control flow from value-domain algebra."""

    @property
    def domain_id(self) -> str: ...

    @property
    def observer_profiles(self) -> tuple[int, ...]: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def validate_value(self, value: V) -> None: ...

    def identity_value(self, observer: int) -> V: ...

    def reframe(self, value: V, target_observer: int) -> V: ...

    def add(self, left: V, right: V) -> V: ...

    def sub(self, left: V, right: V) -> V: ...

    def mul(self, left: V, right: V) -> V: ...

    def compare_equal(self, left: V, right: V, tolerance: float) -> GuardDecision: ...

    def compare_identity(self, value: V, tolerance: float) -> GuardDecision: ...

    def value_to_dict(self, value: V) -> dict[str, Any]: ...

    def value_from_dict(self, data: Mapping[str, Any]) -> V: ...


# ============================================================================
# 1. Signal8Backend: Unit Complex Phasors over Z/8Z
# ============================================================================


@dataclass(frozen=True, slots=True)
class SignalPhasorValue:
    """Calibrated unit-magnitude complex phasor with odd-unit observer and radian error bound."""

    phasor: complex
    observer: int = 1
    phase_error_bound: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.observer, bool) or not isinstance(self.observer, int):
            raise SignalDomainError("observer must be an integer.")
        if self.observer not in SIGNAL8_OBSERVERS:
            raise SignalDomainError(f"observer must be one of {SIGNAL8_OBSERVERS}, got {self.observer}.")
        if isinstance(self.phase_error_bound, bool) or not isinstance(self.phase_error_bound, (int, float)):
            raise SignalDomainError("phase_error_bound must be numeric.")
        err = float(self.phase_error_bound)
        if not math.isfinite(err) or not (0.0 <= err <= math.pi):
            raise SignalDomainError("phase_error_bound must be finite and in [0, pi].")
        z = complex(self.phasor)
        if not (math.isfinite(z.real) and math.isfinite(z.imag)):
            raise SignalDomainError("phasor real and imaginary components must be finite.")
        if abs(abs(z) - 1.0) > 1e-7:
            raise SignalDomainError("phasor must have unit magnitude.")
        object.__setattr__(self, "phasor", z)
        object.__setattr__(self, "phase_error_bound", err)

    def to_dict(self) -> dict[str, Any]:
        return {
            "imag": float(self.phasor.imag),
            "kind": "signal_phasor_z8",
            "observer": self.observer,
            "phase_error_bound": self.phase_error_bound,
            "real": float(self.phasor.real),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SignalPhasorValue:
        if "kind" in data and data["kind"] != "signal_phasor_z8":
            raise SignalDomainError(f"Expected kind='signal_phasor_z8', got {data['kind']!r}.")
        try:
            obs_raw = data["observer"]
            real_raw, imag_raw = data["real"], data["imag"]
            err_raw = data.get("phase_error_bound", 0.0)
            if any(isinstance(v, bool) for v in (obs_raw, real_raw, imag_raw, err_raw)):
                raise SignalDomainError("Boolean values are not valid SignalPhasorValue fields.")
            return cls(
                phasor=complex(float(real_raw), float(imag_raw)),
                observer=int(obs_raw),
                phase_error_bound=float(err_raw),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SignalDomainError(f"Invalid serialized SignalPhasorValue: {exc}") from exc


class Signal8Backend:
    r"""Finite cyclic phase character algebra over $\mathbb{Z}/8\mathbb{Z}$ with four odd-unit observers."""

    domain_id: str = "signal.phase.cyclic.z8.v1"
    observer_profiles: tuple[int, ...] = SIGNAL8_OBSERVERS
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_reframe=True,
        supports_mul=False,
        supports_iq_calibration=True,
        exact_arithmetic=False,
        finite_symbol_cardinality=8,
    )

    @staticmethod
    def _normalize(z: complex) -> complex:
        mag = abs(z)
        if not math.isfinite(mag) or mag <= _EPS:
            raise SignalDomainError("Cannot normalize zero-energy or non-finite phasor.")
        return z / mag

    def validate_value(self, value: SignalPhasorValue) -> None:
        if not isinstance(value, SignalPhasorValue):
            raise SignalDomainError(f"Expected SignalPhasorValue, got {type(value).__name__}.")
        if value.observer not in self.observer_profiles:
            raise SignalDomainError(f"Observer {value.observer} not permitted by {self.domain_id}.")

    def identity_value(self, observer: int = 1) -> SignalPhasorValue:
        if observer not in self.observer_profiles:
            raise SignalDomainError(f"Observer {observer} not in {self.observer_profiles}.")
        return SignalPhasorValue(1.0 + 0.0j, observer, 0.0)

    def encode_symbol(
        self,
        symbol: int,
        observer: int = 1,
        *,
        phase_error_bound: float = 0.0,
    ) -> SignalPhasorValue:
        if isinstance(symbol, bool) or not isinstance(symbol, int) or not (0 <= symbol < 8):
            raise SignalDomainError("Signal8 symbol must be an int in [0, 7].")
        if observer not in self.observer_profiles:
            raise SignalDomainError(f"Observer must be one of {self.observer_profiles}.")
        angle = TAU * ((observer * symbol) % 8) / 8.0
        return SignalPhasorValue(self._normalize(cmath.exp(1j * angle)), observer, phase_error_bound)

    def possible_symbols(self, value: SignalPhasorValue) -> tuple[int, ...]:
        self.validate_value(value)
        phase = math.atan2(value.phasor.imag, value.phasor.real)
        reach = math.pi / 8.0 + value.phase_error_bound + _EPS
        candidates: list[int] = []
        for sym in range(8):
            encoded = (value.observer * sym) % 8
            centre = TAU * encoded / 8.0
            delta = abs(math.atan2(math.sin(phase - centre), math.cos(phase - centre)))
            if delta <= reach:
                candidates.append(sym)
        return tuple(candidates)

    def observe_symbol(self, value: SignalPhasorValue) -> int:
        candidates = self.possible_symbols(value)
        if len(candidates) != 1:
            raise AmbiguousObservationError(candidates, value.phase_error_bound)
        return candidates[0]

    def ingest_iq(
        self,
        signal: Sequence[complex],
        pilot: Sequence[complex],
        observer: int = 1,
        *,
        phase_error_bound: float = 0.0,
    ) -> SignalPhasorValue:
        r"""Calibrate complex baseband I/Q samples against pilot via $\hat{z} = \frac{1}{N}\sum \frac{x_n \overline{p_n}}{|p_n|^2}$."""
        if len(signal) == 0 or len(signal) != len(pilot):
            raise SignalDomainError("signal and pilot must have equal positive length.")
        estimates: list[complex] = []
        for raw_x, raw_p in zip(signal, pilot):
            x, p = complex(raw_x), complex(raw_p)
            if not (math.isfinite(x.real) and math.isfinite(x.imag) and math.isfinite(p.real) and math.isfinite(p.imag)):
                raise SignalDomainError("I/Q samples must be finite complex numbers.")
            energy = abs(p) ** 2
            if energy <= _EPS:
                raise SignalDomainError("Pilot sample has zero energy.")
            estimates.append((x * p.conjugate()) / energy)
        mean_est = sum(estimates) / len(estimates)
        return SignalPhasorValue(self._normalize(mean_est), observer, phase_error_bound)

    def emit_iq(self, value: SignalPhasorValue, pilot: Sequence[complex]) -> tuple[complex, ...]:
        self.validate_value(value)
        if len(pilot) == 0:
            raise SignalDomainError("Pilot carrier sequence must be non-empty.")
        return tuple(complex(p) * value.phasor for p in pilot)

    def reframe(self, value: SignalPhasorValue, target_observer: int) -> SignalPhasorValue:
        self.validate_value(value)
        if target_observer not in self.observer_profiles:
            raise SignalDomainError(f"target_observer must be one of {self.observer_profiles}.")
        if value.observer == target_observer:
            return value
        exponent = (target_observer * pow(value.observer, -1, 8)) % 8
        z = self._normalize(value.phasor**exponent)
        err = min(math.pi, exponent * value.phase_error_bound)
        return SignalPhasorValue(z, target_observer, err)

    def _align(self, left: SignalPhasorValue, right: SignalPhasorValue) -> tuple[SignalPhasorValue, SignalPhasorValue]:
        self.validate_value(left)
        self.validate_value(right)
        if left.observer == right.observer:
            return left, right
        return left, self.reframe(right, left.observer)

    def add(self, left: SignalPhasorValue, right: SignalPhasorValue) -> SignalPhasorValue:
        a, b = self._align(left, right)
        return SignalPhasorValue(
            self._normalize(a.phasor * b.phasor),
            a.observer,
            min(math.pi, a.phase_error_bound + b.phase_error_bound),
        )

    def sub(self, left: SignalPhasorValue, right: SignalPhasorValue) -> SignalPhasorValue:
        a, b = self._align(left, right)
        return SignalPhasorValue(
            self._normalize(a.phasor * b.phasor.conjugate()),
            a.observer,
            min(math.pi, a.phase_error_bound + b.phase_error_bound),
        )

    def mul(self, left: SignalPhasorValue, right: SignalPhasorValue) -> SignalPhasorValue:
        raise SignalDomainError(
            "Signal8Backend represents the additive character group of Z/8Z; ring MUL is not a native character product."
        )

    def compare_equal(self, left: SignalPhasorValue, right: SignalPhasorValue, tolerance: float) -> GuardDecision:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise SignalDomainError("tolerance must be numeric.")
        tol = float(tolerance)
        if not math.isfinite(tol) or not (0.0 <= tol <= math.pi):
            raise SignalDomainError("tolerance must lie in [0, pi].")
        a, b = self._align(left, right)
        rel = a.phasor * b.phasor.conjugate()
        distance = abs(math.atan2(rel.imag, rel.real))
        uncertainty = min(math.pi, a.phase_error_bound + b.phase_error_bound)
        lower = max(0.0, distance - uncertainty)
        upper = min(math.pi, distance + uncertainty)
        if upper <= tol + _EPS:
            disp = TruthDisposition.TRUE
        elif lower > tol + _EPS:
            disp = TruthDisposition.FALSE
        else:
            disp = TruthDisposition.INDETERMINATE
        return GuardDecision(disp, distance, uncertainty, tol, lower, upper)

    def compare_identity(self, value: SignalPhasorValue, tolerance: float) -> GuardDecision:
        self.validate_value(value)
        return self.compare_equal(value, self.identity_value(value.observer), tolerance)

    def value_to_dict(self, value: SignalPhasorValue) -> dict[str, Any]:
        self.validate_value(value)
        return value.to_dict()

    def value_from_dict(self, data: Mapping[str, Any]) -> SignalPhasorValue:
        val = SignalPhasorValue.from_dict(data)
        self.validate_value(val)
        return val


# ============================================================================
# 2. ModularRingBackend: Z/2^w Z Ring with Odd-Unit Observers & MUL Support
# ============================================================================


@dataclass(frozen=True, slots=True)
class ModularResidueValue:
    r"""Observer-scaled residue $r \equiv u \cdot x \pmod{2^w}$ with Lee-metric error radius."""

    residue: int
    observer: int
    modulus: int
    error_radius: int = 0

    def __post_init__(self) -> None:
        for name, val in (
            ("residue", self.residue),
            ("observer", self.observer),
            ("modulus", self.modulus),
            ("error_radius", self.error_radius),
        ):
            if isinstance(val, bool) or not isinstance(val, int):
                raise SignalDomainError(f"{name} must be an exact int.")
        if self.modulus < 4 or (self.modulus & (self.modulus - 1)) != 0:
            raise SignalDomainError("modulus must be a power of two >= 4.")
        if not (0 <= self.residue < self.modulus):
            raise SignalDomainError(f"residue {self.residue} outside [0, {self.modulus}).")
        if math.gcd(self.observer, self.modulus) != 1 or not (1 <= self.observer < self.modulus):
            raise SignalDomainError(f"observer {self.observer} must be an odd unit in [1, {self.modulus}).")
        if not (0 <= self.error_radius <= self.modulus // 2):
            raise SignalDomainError(f"error_radius must lie in [0, {self.modulus // 2}].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_radius": self.error_radius,
            "kind": "modular_residue",
            "modulus": self.modulus,
            "observer": self.observer,
            "residue": self.residue,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModularResidueValue:
        if "kind" in data and data["kind"] != "modular_residue":
            raise SignalDomainError(f"Expected kind='modular_residue', got {data['kind']!r}.")
        try:
            return cls(
                residue=data["residue"],
                observer=data["observer"],
                modulus=data["modulus"],
                error_radius=data.get("error_radius", 0),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SignalDomainError(f"Invalid serialized ModularResidueValue: {exc}") from exc


class ModularRingBackend:
    r"""Exact dyadic ring backend over $\mathbb{Z}/2^w\mathbb{Z}$ supporting `ADD`, `SUB`, `MUL`, and `REFRAME`."""

    def __init__(self, bit_width: int = 8, observer_profiles: Sequence[int] | None = None) -> None:
        if isinstance(bit_width, bool) or not isinstance(bit_width, int) or not (2 <= bit_width <= 64):
            raise SignalDomainError("bit_width must be an integer in [2, 64].")
        self.bit_width = bit_width
        self.modulus = 1 << bit_width
        default_obs = (1, 3) if bit_width == 2 else (1, 3, 5, 7)
        obs = tuple(observer_profiles) if observer_profiles is not None else default_obs
        if not obs or len(set(obs)) != len(obs):
            raise SignalDomainError("observer_profiles must be non-empty and unique.")
        for u in obs:
            if isinstance(u, bool) or not isinstance(u, int) or not (1 <= u < self.modulus) or (u % 2 == 0):
                raise SignalDomainError(f"Invalid odd-unit observer {u!r} for modulus {self.modulus}.")
        self._observers = obs
        self._domain_id = f"signal.ring.z2pow{bit_width}.v1"
        self._capabilities = BackendCapabilities(
            supports_reframe=True,
            supports_mul=True,
            supports_iq_calibration=False,
            exact_arithmetic=True,
            finite_symbol_cardinality=self.modulus,
        )

    @property
    def domain_id(self) -> str:
        return self._domain_id

    @property
    def observer_profiles(self) -> tuple[int, ...]:
        return self._observers

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @staticmethod
    def _lee_norm(x: int, modulus: int) -> int:
        r = x % modulus
        return min(r, modulus - r)

    def validate_value(self, value: ModularResidueValue) -> None:
        if not isinstance(value, ModularResidueValue):
            raise SignalDomainError(f"Expected ModularResidueValue, got {type(value).__name__}.")
        if value.modulus != self.modulus:
            raise SignalDomainError(f"Modulus mismatch: {value.modulus} != {self.modulus}.")
        if value.observer not in self._observers:
            raise SignalDomainError(f"Observer {value.observer} not declared in {self._observers}.")

    def identity_value(self, observer: int = 1) -> ModularResidueValue:
        if observer not in self._observers:
            raise SignalDomainError(f"Observer {observer} not declared in {self._observers}.")
        return ModularResidueValue(0, observer, self.modulus, 0)

    def encode_symbol(self, symbol: int, observer: int = 1, *, error_radius: int = 0) -> ModularResidueValue:
        if isinstance(symbol, bool) or not isinstance(symbol, int) or not (0 <= symbol < self.modulus):
            raise SignalDomainError(f"Symbol must be an int in [0, {self.modulus}).")
        if observer not in self._observers:
            raise SignalDomainError(f"Observer {observer} not declared in {self._observers}.")
        return ModularResidueValue((observer * symbol) % self.modulus, observer, self.modulus, error_radius)

    def observe_symbol(self, value: ModularResidueValue) -> int:
        self.validate_value(value)
        if value.error_radius > 0:
            inv_u = pow(value.observer, -1, self.modulus)
            center = (value.residue * inv_u) % self.modulus
            candidates = tuple(
                sorted({(center + delta) % self.modulus for delta in range(-value.error_radius, value.error_radius + 1)})
            )
            raise AmbiguousObservationError(candidates, TAU * value.error_radius / self.modulus)
        inv_u = pow(value.observer, -1, self.modulus)
        return (value.residue * inv_u) % self.modulus

    def reframe(self, value: ModularResidueValue, target_observer: int) -> ModularResidueValue:
        self.validate_value(value)
        if target_observer not in self._observers:
            raise SignalDomainError(f"target_observer {target_observer} not declared in {self._observers}.")
        if value.observer == target_observer:
            return value
        factor = (target_observer * pow(value.observer, -1, self.modulus)) % self.modulus
        new_residue = (value.residue * factor) % self.modulus
        # Semantic uncertainty delta on x is invariant under unit bijection!
        return ModularResidueValue(new_residue, target_observer, self.modulus, value.error_radius)

    def _align(self, left: ModularResidueValue, right: ModularResidueValue) -> tuple[ModularResidueValue, ModularResidueValue]:
        self.validate_value(left)
        self.validate_value(right)
        if left.observer == right.observer:
            return left, right
        return left, self.reframe(right, left.observer)

    def add(self, left: ModularResidueValue, right: ModularResidueValue) -> ModularResidueValue:
        a, b = self._align(left, right)
        return ModularResidueValue(
            (a.residue + b.residue) % self.modulus,
            a.observer,
            self.modulus,
            min(self.modulus // 2, a.error_radius + b.error_radius),
        )

    def sub(self, left: ModularResidueValue, right: ModularResidueValue) -> ModularResidueValue:
        a, b = self._align(left, right)
        return ModularResidueValue(
            (a.residue - b.residue) % self.modulus,
            a.observer,
            self.modulus,
            min(self.modulus // 2, a.error_radius + b.error_radius),
        )

    def mul(self, left: ModularResidueValue, right: ModularResidueValue) -> ModularResidueValue:
        r"""Compute observer-covariant ring product $r_{a b} \equiv r_a r_b u^{-1} \pmod{2^w}$."""
        a, b = self._align(left, right)
        inv_u = pow(a.observer, -1, self.modulus)
        prod_residue = (a.residue * b.residue * inv_u) % self.modulus
        if a.error_radius == 0 and b.error_radius == 0:
            err = 0
        else:
            xa_norm = self._lee_norm((a.residue * inv_u) % self.modulus, self.modulus)
            xb_norm = self._lee_norm((b.residue * inv_u) % self.modulus, self.modulus)
            err = min(
                self.modulus // 2,
                xa_norm * b.error_radius + xb_norm * a.error_radius + a.error_radius * b.error_radius,
            )
        return ModularResidueValue(prod_residue, a.observer, self.modulus, err)

    def compare_equal(self, left: ModularResidueValue, right: ModularResidueValue, tolerance: float) -> GuardDecision:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise SignalDomainError("tolerance must be numeric.")
        tol = float(tolerance)
        if not math.isfinite(tol) or not (0.0 <= tol <= math.pi):
            raise SignalDomainError("tolerance must lie in [0, pi].")
        a, b = self._align(left, right)
        inv_u = pow(a.observer, -1, self.modulus)
        semantic_diff = ((a.residue - b.residue) * inv_u) % self.modulus
        lee_dist = self._lee_norm(semantic_dist := semantic_diff, self.modulus)
        _ = semantic_dist
        scale = TAU / self.modulus
        distance = min(math.pi, lee_dist * scale)
        uncertainty = min(math.pi, (a.error_radius + b.error_radius) * scale)
        lower = max(0.0, distance - uncertainty)
        upper = min(math.pi, distance + uncertainty)
        if upper <= tol + _EPS:
            disp = TruthDisposition.TRUE
        elif lower > tol + _EPS:
            disp = TruthDisposition.FALSE
        else:
            disp = TruthDisposition.INDETERMINATE
        return GuardDecision(disp, distance, uncertainty, tol, lower, upper)

    def compare_identity(self, value: ModularResidueValue, tolerance: float) -> GuardDecision:
        self.validate_value(value)
        return self.compare_equal(value, self.identity_value(value.observer), tolerance)

    def value_to_dict(self, value: ModularResidueValue) -> dict[str, Any]:
        self.validate_value(value)
        return value.to_dict()

    def value_from_dict(self, data: Mapping[str, Any]) -> ModularResidueValue:
        val = ModularResidueValue.from_dict(data)
        self.validate_value(val)
        return val


# ============================================================================
# 3. ExactRationalBackend: Infinite Characteristic-0 Field Q
# ============================================================================


@dataclass(frozen=True, slots=True)
class RationalSignalValue:
    r"""Exact rational state $q = u \cdot x \in \mathbb{Q}$ with semantic uncertainty $\epsilon \in \mathbb{Q}_{\ge 0}$."""

    scaled_value: Fraction
    observer: int = 1
    error_bound: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        if isinstance(self.observer, bool) or not isinstance(self.observer, int) or self.observer <= 0:
            raise SignalDomainError("Rational observer scale must be a positive integer.")
        if isinstance(self.scaled_value, bool) or isinstance(self.scaled_value, float):
            raise SignalDomainError("scaled_value must be an exact Fraction or int, not float.")
        if isinstance(self.error_bound, bool) or isinstance(self.error_bound, float):
            raise SignalDomainError("error_bound must be an exact Fraction or int, not float.")
        q = Fraction(self.scaled_value)
        eb = Fraction(self.error_bound)
        if eb < 0:
            raise SignalDomainError("error_bound must be non-negative.")
        object.__setattr__(self, "scaled_value", q)
        object.__setattr__(self, "error_bound", eb)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_den": self.error_bound.denominator,
            "error_num": self.error_bound.numerator,
            "kind": "rational_signal",
            "observer": self.observer,
            "val_den": self.scaled_value.denominator,
            "val_num": self.scaled_value.numerator,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RationalSignalValue:
        if "kind" in data and data["kind"] != "rational_signal":
            raise SignalDomainError(f"Expected kind='rational_signal', got {data['kind']!r}.")
        try:
            vn, vd = data["val_num"], data["val_den"]
            obs_raw = data["observer"]
            en, ed = data.get("error_num", 0), data.get("error_den", 1)
            if any(isinstance(v, bool) or not isinstance(v, int) for v in (vn, vd, obs_raw, en, ed)):
                raise SignalDomainError("RationalSignalValue dictionary fields must be exact integers.")
            return cls(
                scaled_value=Fraction(vn, vd),
                observer=obs_raw,
                error_bound=Fraction(en, ed),
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise SignalDomainError(f"Invalid serialized RationalSignalValue: {exc}") from exc


class ExactRationalBackend:
    r"""Exact rational field backend over $\mathbb{Q}$ with zero floating-point drift."""

    domain_id: str = "signal.rational.q.v1"
    capabilities: BackendCapabilities = BackendCapabilities(
        supports_reframe=True,
        supports_mul=True,
        supports_iq_calibration=False,
        exact_arithmetic=True,
        finite_symbol_cardinality=None,
    )

    def __init__(self, observer_profiles: Sequence[int] = (1, 2, 4, 8)) -> None:
        obs = tuple(observer_profiles)
        if not obs or len(set(obs)) != len(obs):
            raise SignalDomainError("observer_profiles must be non-empty and unique.")
        for u in obs:
            if isinstance(u, bool) or not isinstance(u, int) or u <= 0:
                raise SignalDomainError("Rational observer profile must be a positive integer.")
        self._observers = obs

    @property
    def observer_profiles(self) -> tuple[int, ...]:
        return self._observers

    def validate_value(self, value: RationalSignalValue) -> None:
        if not isinstance(value, RationalSignalValue):
            raise SignalDomainError(f"Expected RationalSignalValue, got {type(value).__name__}.")
        if value.observer not in self._observers:
            raise SignalDomainError(f"Observer {value.observer} not declared in {self._observers}.")

    def identity_value(self, observer: int = 1) -> RationalSignalValue:
        if observer not in self._observers:
            raise SignalDomainError(f"Observer {observer} not declared in {self._observers}.")
        return RationalSignalValue(Fraction(0), observer, Fraction(0))

    def encode_rational(
        self,
        value: int | Fraction | str,
        observer: int = 1,
        *,
        error_bound: int | Fraction = Fraction(0),
    ) -> RationalSignalValue:
        if isinstance(value, (bool, float)):
            raise SignalDomainError("encode_rational requires exact int, Fraction, or rational str.")
        if observer not in self._observers:
            raise SignalDomainError(f"Observer {observer} not declared in {self._observers}.")
        q = Fraction(value) * observer
        return RationalSignalValue(q, observer, Fraction(error_bound))

    def observe_rational(self, value: RationalSignalValue) -> Fraction:
        self.validate_value(value)
        if value.error_bound != 0:
            raise SignalDomainError(f"Cannot point-observe rational with non-zero uncertainty {value.error_bound}.")
        return value.scaled_value / value.observer

    def reframe(self, value: RationalSignalValue, target_observer: int) -> RationalSignalValue:
        self.validate_value(value)
        if target_observer not in self._observers:
            raise SignalDomainError(f"target_observer {target_observer} not declared in {self._observers}.")
        if value.observer == target_observer:
            return value
        rescaled = value.scaled_value * Fraction(target_observer, value.observer)
        return RationalSignalValue(rescaled, target_observer, value.error_bound)

    def _align(self, left: RationalSignalValue, right: RationalSignalValue) -> tuple[RationalSignalValue, RationalSignalValue]:
        self.validate_value(left)
        self.validate_value(right)
        if left.observer == right.observer:
            return left, right
        return left, self.reframe(right, left.observer)

    def add(self, left: RationalSignalValue, right: RationalSignalValue) -> RationalSignalValue:
        a, b = self._align(left, right)
        return RationalSignalValue(a.scaled_value + b.scaled_value, a.observer, a.error_bound + b.error_bound)

    def sub(self, left: RationalSignalValue, right: RationalSignalValue) -> RationalSignalValue:
        a, b = self._align(left, right)
        return RationalSignalValue(a.scaled_value - b.scaled_value, a.observer, a.error_bound + b.error_bound)

    def mul(self, left: RationalSignalValue, right: RationalSignalValue) -> RationalSignalValue:
        a, b = self._align(left, right)
        xa = abs(a.scaled_value) / a.observer
        xb = abs(b.scaled_value) / b.observer
        prod_err = xa * b.error_bound + xb * a.error_bound + a.error_bound * b.error_bound
        return RationalSignalValue((a.scaled_value * b.scaled_value) / a.observer, a.observer, prod_err)

    def compare_equal(self, left: RationalSignalValue, right: RationalSignalValue, tolerance: float) -> GuardDecision:
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise SignalDomainError("tolerance must be numeric.")
        tol_float = float(tolerance)
        if not math.isfinite(tol_float) or tol_float < 0.0:
            raise SignalDomainError("tolerance must be finite and non-negative.")
        tol_frac = Fraction(tolerance) if isinstance(tolerance, int) else Fraction(tol_float)
        a, b = self._align(left, right)
        dist_frac = abs(a.scaled_value - b.scaled_value) / a.observer
        unc_frac = a.error_bound + b.error_bound
        lower_frac = max(Fraction(0), dist_frac - unc_frac)
        upper_frac = dist_frac + unc_frac
        if upper_frac <= tol_frac:
            disp = TruthDisposition.TRUE
        elif lower_frac > tol_frac:
            disp = TruthDisposition.FALSE
        else:
            disp = TruthDisposition.INDETERMINATE
        return GuardDecision(
            disposition=disp,
            distance=float(dist_frac),
            uncertainty=float(unc_frac),
            tolerance=tol_float,
            lower_bound=float(lower_frac),
            upper_bound=float(upper_frac),
        )

    def compare_identity(self, value: RationalSignalValue, tolerance: float) -> GuardDecision:
        self.validate_value(value)
        return self.compare_equal(value, self.identity_value(value.observer), tolerance)

    def value_to_dict(self, value: RationalSignalValue) -> dict[str, Any]:
        self.validate_value(value)
        return value.to_dict()

    def value_from_dict(self, data: Mapping[str, Any]) -> RationalSignalValue:
        val = RationalSignalValue.from_dict(data)
        self.validate_value(val)
        return val
