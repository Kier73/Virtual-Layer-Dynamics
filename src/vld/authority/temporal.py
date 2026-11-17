r"""Phase 3 — Scheduler-Scoped Temporal Coordinates, Vector Clocks & Decoupled Provenance (`vld.authority.temporal`).

Governing equations:
1. **Lossless Temporal Numeric Envelopes:**
   - Arbitrary-precision integers serialize as canonical base-10 strings (`"0"` or `"-?[1-9][0-9]*"`),
     never passing through IEEE-754 JSON numbers.
   - Finite binary64 floats serialize as exact 16-hex big-endian IEEE-754 bit patterns (`struct.pack(">d", x)`),
     preserving `-0.0` and subnormal magnitudes losslessly.
2. **Scheduler-Scoped Vector Clocks (Resolving `G13`):**
   A temporal coordinate $\tau = (\text{scheduler\_id}, \text{epoch}, V)$ scopes causal vector clock
   $V: \mathcal{P} \to \mathbb{N}_0$ to an explicit scheduler domain. Comparing $\tau_1, \tau_2$ across
   distinct schedulers ($\text{scheduler\_id}_1 \neq \text{scheduler\_id}_2$) without a calibrated
   `ClockBridgeReceipt` raises `CrossSchedulerComparisonError` rather than silently misinterpreting ticks:
     $$V_1 \le V_2 \iff \forall p \in \mathcal{P},\; V_1(p) \le V_2(p)$$
3. **Observation vs. Interpretation Separation (`G21`):**
   `EncounterEnvelope` binds raw byte hash, byte length, observer identity, and temporal coordinate with
   zero semantic commitment. `InterpretationEnvelope` references `encounter_digest` separately so
   hypothesis revision or Ground demotion never mutates raw observation provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import math
import re
import struct
from typing import Any, Mapping

from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.ground import LensKind
from vld.triad import EpistemicStatus

_DECIMAL_RE = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_HEX64_BITS_RE = re.compile(r"[0-9a-f]{16}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class TemporalError(ValueError):
    """Raised when a temporal coordinate, lossless numeric envelope, or provenance record is invalid."""


class CrossSchedulerComparisonError(TemporalError):
    """Raised when comparing temporal coordinates across distinct schedulers without a calibrated bridge (`G13`)."""


class CausalOrder(str, Enum):
    """Partial-order disposition between two causal vector clocks within the same scheduler scope."""

    BEFORE = "before"
    AFTER = "after"
    EQUAL = "equal"
    CONCURRENT = "concurrent"


def encode_temporal_int(value: int) -> dict[str, str]:
    """Encode an arbitrary-precision Python integer as a lossless canonical decimal envelope."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalError("Temporal integer must be an exact int, not bool/float.")
    return {
        "encoding": "canonical-decimal-string",
        "kind": "integer",
        "schema": "vld.temporal-numeric.v2",
        "value": str(value),
    }


def decode_temporal_int(envelope: Mapping[str, Any], *, minimum: int | None = None) -> int:
    """Decode a canonical decimal integer envelope with strict schema validation."""
    if not isinstance(envelope, Mapping) or set(envelope.keys()) != {"encoding", "kind", "schema", "value"}:
        raise TemporalError("Invalid temporal integer envelope keyset.")
    if (
        envelope["schema"] != "vld.temporal-numeric.v2"
        or envelope["kind"] != "integer"
        or envelope["encoding"] != "canonical-decimal-string"
    ):
        raise TemporalError("Temporal integer envelope schema/encoding mismatch.")
    tok = envelope["value"]
    if not isinstance(tok, str) or not tok.isascii() or _DECIMAL_RE.fullmatch(tok) is None:
        raise TemporalError(f"Non-canonical decimal integer token: {tok!r}")
    val = int(tok, 10)
    if minimum is not None and val < minimum:
        raise TemporalError(f"Decoded temporal integer {val} < minimum {minimum}.")
    return val


def encode_temporal_f64(value: float) -> dict[str, str]:
    """Encode a finite IEEE 754 binary64 float as its exact 16-hex big-endian bit pattern."""
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise TemporalError("Temporal binary64 value must be a finite Python float.")
    return {
        "encoding": "ieee754-binary64-be-hex",
        "kind": "binary64",
        "schema": "vld.temporal-numeric.v2",
        "value": struct.pack(">d", value).hex(),
    }


def decode_temporal_f64(envelope: Mapping[str, Any]) -> float:
    """Decode an exact IEEE 754 binary64 envelope preserving `-0.0` and subnormals."""
    if not isinstance(envelope, Mapping) or set(envelope.keys()) != {"encoding", "kind", "schema", "value"}:
        raise TemporalError("Invalid temporal binary64 envelope keyset.")
    if (
        envelope["schema"] != "vld.temporal-numeric.v2"
        or envelope["kind"] != "binary64"
        or envelope["encoding"] != "ieee754-binary64-be-hex"
    ):
        raise TemporalError("Temporal binary64 envelope schema/encoding mismatch.")
    tok = envelope["value"]
    if not isinstance(tok, str) or _HEX64_BITS_RE.fullmatch(tok) is None:
        raise TemporalError("Temporal binary64 value must be 16 lowercase hex digits.")
    val = struct.unpack(">d", bytes.fromhex(tok))[0]
    if not math.isfinite(val):
        raise TemporalError("Decoded binary64 value is non-finite.")
    return val


@dataclass(frozen=True, slots=True)
class VectorClock:
    r"""Immutable non-zero-sparse causal vector clock $V: \mathcal{P} \to \mathbb{N}_0$."""

    entries: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        cleaned: dict[str, int] = {}
        raw_items = self.entries.items() if isinstance(self.entries, Mapping) else self.entries
        for k, v in raw_items:
            if not isinstance(k, str) or not k.strip():
                raise TemporalError("VectorClock locus key must be a non-empty string.")
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise TemporalError("VectorClock tick count must be a non-negative integer.")
            if k in cleaned:
                raise TemporalError(f"Duplicate locus key {k!r} in VectorClock.")
            if v > 0:
                cleaned[k] = v
        object.__setattr__(self, "entries", tuple(sorted(cleaned.items())))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, int]) -> VectorClock:
        return cls(tuple(mapping.items()))

    def get(self, locus_id: str) -> int:
        for k, v in self.entries:
            if k == locus_id:
                return v
        return 0

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {k: encode_temporal_int(v) for k, v in self.entries}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VectorClock:
        if not isinstance(payload, Mapping):
            raise TemporalError("VectorClock payload must be a mapping.")
        return cls(tuple((k, decode_temporal_int(v, minimum=0)) for k, v in payload.items()))

    def advance(self, locus_id: str, delta: int = 1) -> VectorClock:
        if not isinstance(locus_id, str) or not locus_id.strip():
            raise TemporalError("locus_id must be a non-empty string.")
        if isinstance(delta, bool) or not isinstance(delta, int) or delta <= 0:
            raise TemporalError("delta must be a positive integer.")
        d = dict(self.entries)
        d[locus_id] = d.get(locus_id, 0) + delta
        return VectorClock.from_mapping(d)

    def merge(self, other: VectorClock) -> VectorClock:
        if not isinstance(other, VectorClock):
            raise TemporalError("Can only merge with another VectorClock.")
        d = dict(self.entries)
        for k, v in other.entries:
            if v > d.get(k, 0):
                d[k] = v
        return VectorClock.from_mapping(d)

    def compare(self, other: VectorClock) -> CausalOrder:
        if not isinstance(other, VectorClock):
            raise TemporalError("Can only compare with another VectorClock.")
        a = dict(self.entries)
        b = dict(other.entries)
        keys = set(a.keys()) | set(b.keys())
        le = all(a.get(k, 0) <= b.get(k, 0) for k in keys)
        ge = all(a.get(k, 0) >= b.get(k, 0) for k in keys)
        if le and ge:
            return CausalOrder.EQUAL
        if le:
            return CausalOrder.BEFORE
        if ge:
            return CausalOrder.AFTER
        return CausalOrder.CONCURRENT


@dataclass(frozen=True, slots=True)
class ClockBridgeReceipt:
    """Explicit causal synchronization witness permitting bounded cross-scheduler comparison."""

    source_scheduler_id: str
    source_epoch: int
    source_anchor: VectorClock
    target_scheduler_id: str
    target_epoch: int
    target_anchor: VectorClock

    def __post_init__(self) -> None:
        if not self.source_scheduler_id.strip() or not self.target_scheduler_id.strip():
            raise TemporalError("Bridge scheduler IDs must be non-empty.")
        if self.source_scheduler_id == self.target_scheduler_id and self.source_epoch == self.target_epoch:
            raise TemporalError("ClockBridgeReceipt requires distinct scheduler scopes or epochs.")
        if isinstance(self.source_epoch, bool) or self.source_epoch < 0 or isinstance(self.target_epoch, bool) or self.target_epoch < 0:
            raise TemporalError("Bridge epochs must be non-negative integers.")

    @property
    def bridge_digest(self) -> str:
        return domain_digest(
            "vld.clock-bridge.v1",
            {
                "source_anchor": self.source_anchor.to_dict(),
                "source_epoch": encode_temporal_int(self.source_epoch),
                "source_scheduler_id": self.source_scheduler_id,
                "target_anchor": self.target_anchor.to_dict(),
                "target_epoch": encode_temporal_int(self.target_epoch),
                "target_scheduler_id": self.target_scheduler_id,
            },
        )


@dataclass(frozen=True, slots=True)
class TemporalCoordinate:
    """Scheduler-scoped causal coordinate preventing silent cross-scheduler tick confusion (`G13`)."""

    scheduler_id: str
    epoch: int
    clock: VectorClock
    wall_hint_f64: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheduler_id, str) or not self.scheduler_id.strip():
            raise TemporalError("scheduler_id must be a non-empty string.")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise TemporalError("epoch must be a non-negative integer.")
        if not isinstance(self.clock, VectorClock):
            raise TemporalError("clock must be a VectorClock.")
        if self.wall_hint_f64 is not None:
            if isinstance(self.wall_hint_f64, bool) or not isinstance(self.wall_hint_f64, float) or not math.isfinite(self.wall_hint_f64):
                raise TemporalError("wall_hint_f64 must be a finite float when provided.")

    def advance(self, locus_id: str, delta: int = 1, *, wall_hint_f64: float | None = None) -> TemporalCoordinate:
        return TemporalCoordinate(
            scheduler_id=self.scheduler_id,
            epoch=self.epoch,
            clock=self.clock.advance(locus_id, delta),
            wall_hint_f64=wall_hint_f64,
        )

    def receive_merge(self, incoming: TemporalCoordinate, locus_id: str) -> TemporalCoordinate:
        """Merge causal history within the same `(scheduler_id, epoch)` and advance receiver locus."""
        if self.scheduler_id != incoming.scheduler_id or self.epoch != incoming.epoch:
            raise CrossSchedulerComparisonError(
                f"Cannot directly merge coordinates across scopes ({self.scheduler_id}@{self.epoch} vs "
                f"{incoming.scheduler_id}@{incoming.epoch}) without a ClockBridgeReceipt."
            )
        merged = self.clock.merge(incoming.clock).advance(locus_id, 1)
        return TemporalCoordinate(scheduler_id=self.scheduler_id, epoch=self.epoch, clock=merged)

    def compare(self, other: TemporalCoordinate, *, bridge: ClockBridgeReceipt | None = None) -> CausalOrder:
        """Compare causal order, enforcing `G13` scheduler-scope isolation unless bridged."""
        if not isinstance(other, TemporalCoordinate):
            raise TemporalError("Expected TemporalCoordinate.")
        if self.scheduler_id == other.scheduler_id and self.epoch == other.epoch:
            return self.clock.compare(other.clock)

        if bridge is None:
            raise CrossSchedulerComparisonError(
                f"Cross-scheduler/epoch comparison refused ({self.scheduler_id}@{self.epoch} vs "
                f"{other.scheduler_id}@{other.epoch}): supply an explicit ClockBridgeReceipt."
            )

        # Check transitive happens-before across source_anchor -> target_anchor!
        if (
            self.scheduler_id == bridge.source_scheduler_id
            and self.epoch == bridge.source_epoch
            and other.scheduler_id == bridge.target_scheduler_id
            and other.epoch == bridge.target_epoch
        ):
            c_left = self.clock.compare(bridge.source_anchor)
            c_right = bridge.target_anchor.compare(other.clock)
            if c_left in (CausalOrder.BEFORE, CausalOrder.EQUAL) and c_right in (CausalOrder.BEFORE, CausalOrder.EQUAL):
                return CausalOrder.BEFORE
            return CausalOrder.CONCURRENT

        if (
            other.scheduler_id == bridge.source_scheduler_id
            and other.epoch == bridge.source_epoch
            and self.scheduler_id == bridge.target_scheduler_id
            and self.epoch == bridge.target_epoch
        ):
            c_left = other.clock.compare(bridge.source_anchor)
            c_right = bridge.target_anchor.compare(self.clock)
            if c_left in (CausalOrder.BEFORE, CausalOrder.EQUAL) and c_right in (CausalOrder.BEFORE, CausalOrder.EQUAL):
                return CausalOrder.AFTER
            return CausalOrder.CONCURRENT

        raise CrossSchedulerComparisonError("Supplied ClockBridgeReceipt does not match coordinate scopes.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock": self.clock.to_dict(),
            "epoch": encode_temporal_int(self.epoch),
            "scheduler_id": self.scheduler_id,
            "schema": "vld.temporal-coordinate.v2",
            "wall_hint": None if self.wall_hint_f64 is None else encode_temporal_f64(self.wall_hint_f64),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TemporalCoordinate:
        req = {"clock", "epoch", "scheduler_id", "schema", "wall_hint"}
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.temporal-coordinate.v2":
            raise TemporalError("Invalid TemporalCoordinate v2 payload.")
        wh = None if data["wall_hint"] is None else decode_temporal_f64(data["wall_hint"])
        return cls(
            scheduler_id=data["scheduler_id"],
            epoch=decode_temporal_int(data["epoch"], minimum=0),
            clock=VectorClock.from_dict(data["clock"]),
            wall_hint_f64=wh,
        )

    @property
    def digest(self) -> str:
        return domain_digest("vld.temporal-coordinate.v2", self.to_dict())


@dataclass(frozen=True, slots=True)
class EncounterEnvelope:
    """Raw observation receipt (`G21`) strictly decoupled from downstream Ground interpretation."""

    encounter_id: str
    observer_id: str
    temporal_coord: TemporalCoordinate
    raw_sha256: str
    byte_length: int
    channel_meta: Mapping[str, Any]
    encounter_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.encounter_id, str) or not self.encounter_id.strip():
            raise TemporalError("encounter_id must be a non-empty string.")
        if not isinstance(self.observer_id, str) or not self.observer_id.strip():
            raise TemporalError("observer_id must be a non-empty string.")
        if not isinstance(self.temporal_coord, TemporalCoordinate):
            raise TemporalError("temporal_coord must be a TemporalCoordinate.")
        if not isinstance(self.raw_sha256, str) or _SHA256_RE.fullmatch(self.raw_sha256) is None:
            raise TemporalError("raw_sha256 must be a 64-char lowercase hex digest.")
        if isinstance(self.byte_length, bool) or not isinstance(self.byte_length, int) or self.byte_length < 0:
            raise TemporalError("byte_length must be a non-negative integer.")
        clean_meta = parse_strict_json(canonical_json(dict(self.channel_meta)))
        if not isinstance(clean_meta, dict):
            raise TemporalError("channel_meta must be a JSON object.")
        object.__setattr__(self, "channel_meta", clean_meta)
        computed = domain_digest("vld.encounter-envelope.v1", self._body())
        if self.encounter_digest and self.encounter_digest != computed:
            raise TemporalError("EncounterEnvelope digest mismatch.")
        object.__setattr__(self, "encounter_digest", computed)

    @classmethod
    def capture(
        cls,
        *,
        encounter_id: str,
        observer_id: str,
        temporal_coord: TemporalCoordinate,
        raw_bytes: bytes,
        channel_meta: Mapping[str, Any] | None = None,
    ) -> EncounterEnvelope:
        if not isinstance(raw_bytes, bytes):
            raise TemporalError("raw_bytes must be bytes.")
        return cls(
            encounter_id=encounter_id,
            observer_id=observer_id,
            temporal_coord=temporal_coord,
            raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            byte_length=len(raw_bytes),
            channel_meta=dict(channel_meta or {}),
        )

    def _body(self) -> dict[str, Any]:
        return {
            "byte_length": encode_temporal_int(self.byte_length),
            "channel_meta": dict(self.channel_meta),
            "encounter_id": self.encounter_id,
            "observer_id": self.observer_id,
            "raw_sha256": self.raw_sha256,
            "schema": "vld.encounter-envelope.v1",
            "temporal_coord": self.temporal_coord.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "encounter_digest": self.encounter_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EncounterEnvelope:
        req = {
            "byte_length",
            "channel_meta",
            "encounter_digest",
            "encounter_id",
            "observer_id",
            "raw_sha256",
            "schema",
            "temporal_coord",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.encounter-envelope.v1":
            raise TemporalError("Invalid EncounterEnvelope payload.")
        return cls(
            encounter_id=data["encounter_id"],
            observer_id=data["observer_id"],
            temporal_coord=TemporalCoordinate.from_dict(data["temporal_coord"]),
            raw_sha256=data["raw_sha256"],
            byte_length=decode_temporal_int(data["byte_length"], minimum=0),
            channel_meta=data["channel_meta"],
            encounter_digest=data["encounter_digest"],
        )


@dataclass(frozen=True, slots=True)
class InterpretationEnvelope:
    """Downstream semantic projection linked to an `EncounterEnvelope` by immutable digest."""

    encounter_digest: str
    ground_fingerprint: str
    lens_kind: LensKind
    epistemic_status: EpistemicStatus
    interpreted_value: Any
    interpreter_id: str
    interpretation_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.encounter_digest, str) or _SHA256_RE.fullmatch(self.encounter_digest) is None:
            raise TemporalError("encounter_digest must be a 64-char SHA-256 hex digest.")
        if not isinstance(self.ground_fingerprint, str) or _SHA256_RE.fullmatch(self.ground_fingerprint) is None:
            raise TemporalError("ground_fingerprint must be a 64-char SHA-256 hex digest.")
        if not isinstance(self.lens_kind, LensKind):
            object.__setattr__(self, "lens_kind", LensKind(self.lens_kind))
        if not isinstance(self.epistemic_status, EpistemicStatus):
            object.__setattr__(self, "epistemic_status", EpistemicStatus(self.epistemic_status))
        if not isinstance(self.interpreter_id, str) or not self.interpreter_id.strip():
            raise TemporalError("interpreter_id must be a non-empty string.")
        clean_val = parse_strict_json(canonical_json(self.interpreted_value))
        object.__setattr__(self, "interpreted_value", clean_val)
        computed = domain_digest("vld.interpretation-envelope.v1", self._body())
        if self.interpretation_digest and self.interpretation_digest != computed:
            raise TemporalError("InterpretationEnvelope digest mismatch.")
        object.__setattr__(self, "interpretation_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "encounter_digest": self.encounter_digest,
            "epistemic_status": self.epistemic_status.value,
            "ground_fingerprint": self.ground_fingerprint,
            "interpreted_value": self.interpreted_value,
            "interpreter_id": self.interpreter_id,
            "lens_kind": self.lens_kind.value,
            "schema": "vld.interpretation-envelope.v1",
        }

    def verify_against_encounter(self, encounter: EncounterEnvelope) -> bool:
        return self.encounter_digest == encounter.encounter_digest

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "interpretation_digest": self.interpretation_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InterpretationEnvelope:
        req = {
            "encounter_digest",
            "epistemic_status",
            "ground_fingerprint",
            "interpretation_digest",
            "interpreted_value",
            "interpreter_id",
            "lens_kind",
            "schema",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.interpretation-envelope.v1":
            raise TemporalError("Invalid InterpretationEnvelope payload.")
        return cls(
            encounter_digest=data["encounter_digest"],
            ground_fingerprint=data["ground_fingerprint"],
            lens_kind=LensKind(data["lens_kind"]),
            epistemic_status=EpistemicStatus(data["epistemic_status"]),
            interpreted_value=data["interpreted_value"],
            interpreter_id=data["interpreter_id"],
            interpretation_digest=data["interpretation_digest"],
        )


__all__ = [
    "CausalOrder",
    "ClockBridgeReceipt",
    "CrossSchedulerComparisonError",
    "EncounterEnvelope",
    "InterpretationEnvelope",
    "TemporalCoordinate",
    "TemporalError",
    "VectorClock",
    "decode_temporal_f64",
    "decode_temporal_int",
    "encode_temporal_f64",
    "encode_temporal_int",
]
