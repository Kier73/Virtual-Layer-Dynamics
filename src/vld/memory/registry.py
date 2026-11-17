r"""Versioned Structural Memory Registry with zero-intermediate-state reads and counterexample demotion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import threading
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest
from vld.journal import HashChainedJournal
from vld.memory.certifier import CertifiedLaw, HorizonExceededError, LawCertificationRefused, certify_sequence_law


class StructuralMemoryError(ValueError):
    """Raised on malformed, unauthorized, or invalidated Structural Memory operations."""


class CertificateStatus(str, Enum):
    """Lifecycle disposition of a Structural Memory revision certificate."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED_BY_COUNTEREXAMPLE = "invalidated_by_counterexample"


@dataclass(frozen=True, slots=True)
class StructuralMemorySpec:
    """Immutable revision-pinned Structural Memory specification."""

    memory_id: str
    revision: int
    domain: str
    law: CertifiedLaw
    context: Mapping[str, Any]
    parent_spec_digest: str | None = None

    SCHEMA = "vld.structural-memory-spec.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise StructuralMemoryError("memory_id must be a non-empty string.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise StructuralMemoryError("revision must be an int >= 1.")
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise StructuralMemoryError("domain must be a non-empty string.")
        if not isinstance(self.law, CertifiedLaw):
            raise StructuralMemoryError("law must be a CertifiedLaw instance.")

    @property
    def context_digest(self) -> str:
        return hashlib.sha256(canonical_json(dict(self.context)).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": dict(self.context),
            "domain": self.domain,
            "law": self.law.to_dict(),
            "memory_id": self.memory_id,
            "parent_spec_digest": self.parent_spec_digest,
            "revision": self.revision,
            "schema": self.SCHEMA,
        }

    @property
    def digest(self) -> str:
        return domain_digest(self.SCHEMA, self.to_dict())


@dataclass(frozen=True, slots=True)
class StructuralMemoryCertificate:
    """Evidence certificate binding a `StructuralMemorySpec` to its witness digest."""

    spec_digest: str
    context_digest: str
    witness_digest: str
    witness_count: int
    verdict: str
    first_bad: int | None

    SCHEMA = "vld.structural-memory-cert.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_digest": self.context_digest,
            "first_bad": self.first_bad,
            "schema": self.SCHEMA,
            "spec_digest": self.spec_digest,
            "verdict": self.verdict,
            "witness_count": self.witness_count,
            "witness_digest": self.witness_digest,
        }

    @property
    def digest(self) -> str:
        return domain_digest(self.SCHEMA, self.to_dict())


@dataclass(frozen=True, slots=True)
class StructuralReadReceipt:
    """Verifiable read receipt proving zero intermediate state expansion."""

    memory_id: str
    revision: int
    spec_digest: str
    certificate_digest: str
    context_digest: str
    coordinate_start: int
    length: int
    values: tuple[int, ...]
    expanded_intermediate_states: int
    journal_event_digest: str

    SCHEMA = "vld.structural-read-receipt.v1"

    def body_without_digest(self) -> dict[str, Any]:
        return {
            "certificate_digest": self.certificate_digest,
            "context_digest": self.context_digest,
            "coordinate_start": self.coordinate_start,
            "expanded_intermediate_states": self.expanded_intermediate_states,
            "journal_event_digest": self.journal_event_digest,
            "length": self.length,
            "memory_id": self.memory_id,
            "revision": self.revision,
            "schema": self.SCHEMA,
            "spec_digest": self.spec_digest,
            "values": list(self.values),
        }

    @property
    def receipt_digest(self) -> str:
        return domain_digest(self.SCHEMA, self.body_without_digest())


@dataclass
class _RevisionRecord:
    spec: StructuralMemorySpec
    certificate: StructuralMemoryCertificate
    status: CertificateStatus
    invalidated_at_coordinate: int | None = None
    counterexample_value: int | None = None


class StructuralMemoryRegistry:
    """Append-only revisioned Structural Memory registry with counterexample invalidation."""

    def __init__(self, journal: HashChainedJournal | None = None) -> None:
        self.journal = journal if journal is not None else HashChainedJournal()
        self._lock = threading.RLock()
        self._revisions: dict[str, list[_RevisionRecord]] = {}
        self._active_revision: dict[str, int | None] = {}
        self._op_seq = 0

    def _next_req_id(self, prefix: str) -> str:
        with self._lock:
            self._op_seq += 1
            return f"{prefix}-{self._op_seq:06d}"

    def certify_and_admit(
        self,
        memory_id: str,
        samples: Sequence[int],
        *,
        domain: str = "Z",
        declared_modulus: int | None = None,
        allow_unbounded_poly: bool = False,
        prefer_gf2: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[StructuralMemorySpec, StructuralMemoryCertificate]:
        law = certify_sequence_law(
            samples,
            declared_modulus=declared_modulus,
            allow_unbounded_poly=allow_unbounded_poly,
            prefer_gf2=prefer_gf2,
        )
        return self.admit_law(memory_id, law, witnesses=samples, domain=domain, context=context)

    def admit_law(
        self,
        memory_id: str,
        law: CertifiedLaw,
        *,
        witnesses: Sequence[int],
        domain: str = "Z",
        context: Mapping[str, Any] | None = None,
    ) -> tuple[StructuralMemorySpec, StructuralMemoryCertificate]:
        if len(witnesses) < 1:
            raise StructuralMemoryError("At least 1 witness value is required to bind a certificate.")
        for t, w in enumerate(witnesses):
            if law.verdict == "PREFIX-GROUND" and law.first_bad is not None and t >= law.first_bad:
                break
            if law.evaluate(t) != w:
                raise StructuralMemoryError(f"Supplied law disagrees with witness at coordinate {t}.")

        with self._lock:
            history = self._revisions.setdefault(memory_id, [])
            rev_num = len(history) + 1
            parent_digest = history[-1].spec.digest if history else None
            for prior in history:
                if prior.status is CertificateStatus.ACTIVE:
                    prior.status = CertificateStatus.SUPERSEDED

            spec = StructuralMemorySpec(
                memory_id=memory_id,
                revision=rev_num,
                domain=domain,
                law=law,
                context=dict(context or {}),
                parent_spec_digest=parent_digest,
            )
            witness_digest = domain_digest("vld.structural-witnesses.v1", list(witnesses))
            cert = StructuralMemoryCertificate(
                spec_digest=spec.digest,
                context_digest=spec.context_digest,
                witness_digest=witness_digest,
                witness_count=len(witnesses),
                verdict=law.verdict,
                first_bad=law.first_bad,
            )
            history.append(_RevisionRecord(spec=spec, certificate=cert, status=CertificateStatus.ACTIVE))
            self._active_revision[memory_id] = rev_num

            self.journal.record(
                request_id=self._next_req_id(f"sm-admit-{memory_id}"),
                ground_fingerprint=spec.digest,
                operation="structural_memory.admit",
                payload={
                    "memory_id": memory_id,
                    "parent_spec_digest": parent_digest,
                    "revision": rev_num,
                    "verdict": law.verdict,
                },
                result_fn=lambda: {
                    "certificate_digest": cert.digest,
                    "family": law.family,
                    "spec_digest": spec.digest,
                },
            )
            return spec, cert

    def active_record(self, memory_id: str) -> tuple[StructuralMemorySpec, StructuralMemoryCertificate, CertificateStatus]:
        with self._lock:
            if memory_id not in self._revisions:
                raise StructuralMemoryError(f"Unknown memory_id {memory_id!r}.")
            active_rev = self._active_revision.get(memory_id)
            if active_rev is None:
                raise StructuralMemoryError(f"Memory {memory_id!r} has been demoted with no active Ground revision.")
            rec = self._revisions[memory_id][active_rev - 1]
            return rec.spec, rec.certificate, rec.status

    def revision_history(self, memory_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            if memory_id not in self._revisions:
                raise StructuralMemoryError(f"Unknown memory_id {memory_id!r}.")
            return tuple(
                {
                    "certificate_digest": r.certificate.digest,
                    "counterexample_value": r.counterexample_value,
                    "family": r.spec.law.family,
                    "first_bad": r.spec.law.first_bad,
                    "invalidated_at_coordinate": r.invalidated_at_coordinate,
                    "revision": r.spec.revision,
                    "spec_digest": r.spec.digest,
                    "status": r.status.value,
                    "verdict": r.spec.law.verdict,
                }
                for r in self._revisions[memory_id]
            )

    def read(self, memory_id: str, coordinate: int, *, revision: int | None = None) -> tuple[int, StructuralReadReceipt]:
        vals, receipt = self.read_range(memory_id, coordinate, 1, revision=revision)
        return vals[0], receipt

    def read_range(
        self,
        memory_id: str,
        coordinate_start: int,
        length: int,
        *,
        revision: int | None = None,
    ) -> tuple[tuple[int, ...], StructuralReadReceipt]:
        if isinstance(coordinate_start, bool) or not isinstance(coordinate_start, int) or coordinate_start < 0:
            raise StructuralMemoryError("coordinate_start must be a non-negative integer.")
        if isinstance(length, bool) or not isinstance(length, int) or length < 1 or length > 65_536:
            raise StructuralMemoryError("length must be an int in [1, 65536].")

        with self._lock:
            if memory_id not in self._revisions:
                raise StructuralMemoryError(f"Unknown memory_id {memory_id!r}.")
            if revision is None:
                rev_idx = self._active_revision.get(memory_id)
                if rev_idx is None:
                    raise StructuralMemoryError(f"Memory {memory_id!r} has no active Ground law.")
            else:
                if not (1 <= revision <= len(self._revisions[memory_id])):
                    raise StructuralMemoryError(f"Unknown revision {revision} for {memory_id!r}.")
                rev_idx = revision

            rec = self._revisions[memory_id][rev_idx - 1]
            if rec.status is CertificateStatus.INVALIDATED_BY_COUNTEREXAMPLE:
                raise StructuralMemoryError(
                    f"Revision {rec.spec.revision} of {memory_id!r} was invalidated by counterexample at coordinate {rec.invalidated_at_coordinate}."
                )

            values = tuple(rec.spec.law.evaluate(coordinate_start + offset) for offset in range(length))
            j_rec, _ = self.journal.record(
                request_id=self._next_req_id(f"sm-read-{memory_id}"),
                ground_fingerprint=rec.spec.digest,
                operation="structural_memory.read_range",
                payload={
                    "coordinate_start": coordinate_start,
                    "length": length,
                    "memory_id": memory_id,
                    "revision": rec.spec.revision,
                },
                result_fn=lambda: {
                    "certificate_digest": rec.certificate.digest,
                    "expanded_intermediate_states": 0,
                    "values_digest": domain_digest("vld.read-values.v1", list(values)),
                },
            )
            receipt = StructuralReadReceipt(
                memory_id=memory_id,
                revision=rec.spec.revision,
                spec_digest=rec.spec.digest,
                certificate_digest=rec.certificate.digest,
                context_digest=rec.spec.context_digest,
                coordinate_start=coordinate_start,
                length=length,
                values=values,
                expanded_intermediate_states=0,
                journal_event_digest=j_rec.event_digest,
            )
            return values, receipt

    def locate_value(self, memory_id: str, target: int) -> int | None:
        spec, _, _ = self.active_record(memory_id)
        return spec.law.locate_value(target)

    def inject_counterexample(
        self,
        memory_id: str,
        coordinate: int,
        observed_value: int,
    ) -> dict[str, Any]:
        """Inject a held-out counterexample, demote active Ground revision, and preserve full audit lineage."""
        if isinstance(coordinate, bool) or not isinstance(coordinate, int) or coordinate < 0:
            raise StructuralMemoryError("coordinate must be a non-negative integer.")
        if isinstance(observed_value, bool) or not isinstance(observed_value, int):
            raise StructuralMemoryError("observed_value must be an exact int.")

        with self._lock:
            if memory_id not in self._revisions:
                raise StructuralMemoryError(f"Unknown memory_id {memory_id!r}.")
            active_rev = self._active_revision.get(memory_id)
            if active_rev is None:
                raise StructuralMemoryError(f"Memory {memory_id!r} already has no active law.")
            rec = self._revisions[memory_id][active_rev - 1]
            try:
                predicted = rec.spec.law.evaluate(coordinate)
            except HorizonExceededError as exc:
                raise StructuralMemoryError("Coordinate already lies beyond active PREFIX-GROUND horizon.") from exc

            if predicted == observed_value:
                raise StructuralMemoryError(
                    f"Observation {observed_value} at coordinate {coordinate} agrees with active law; not a counterexample."
                )

            # Invalidate current revision in-place without deleting it from history!
            rec.status = CertificateStatus.INVALIDATED_BY_COUNTEREXAMPLE
            rec.invalidated_at_coordinate = coordinate
            rec.counterexample_value = observed_value

            new_rev_num: int | None = None
            new_spec_digest: str | None = None
            if coordinate >= 1:
                bounded_law = CertifiedLaw(
                    family=rec.spec.law.family,
                    params=rec.spec.law.params,
                    order=rec.spec.law.order,
                    verdict="PREFIX-GROUND",
                    first_bad=coordinate if rec.spec.law.first_bad is None else min(rec.spec.law.first_bad, coordinate),
                )
                new_rev_num = len(self._revisions[memory_id]) + 1
                new_spec = StructuralMemorySpec(
                    memory_id=memory_id,
                    revision=new_rev_num,
                    domain=rec.spec.domain,
                    law=bounded_law,
                    context=rec.spec.context,
                    parent_spec_digest=rec.spec.digest,
                )
                new_cert = StructuralMemoryCertificate(
                    spec_digest=new_spec.digest,
                    context_digest=new_spec.context_digest,
                    witness_digest=rec.certificate.witness_digest,
                    witness_count=min(rec.certificate.witness_count, coordinate),
                    verdict="PREFIX-GROUND",
                    first_bad=bounded_law.first_bad,
                )
                self._revisions[memory_id].append(
                    _RevisionRecord(spec=new_spec, certificate=new_cert, status=CertificateStatus.ACTIVE)
                )
                self._active_revision[memory_id] = new_rev_num
                new_spec_digest = new_spec.digest
            else:
                self._active_revision[memory_id] = None

            j_rec, _ = self.journal.record(
                request_id=self._next_req_id(f"sm-counterexample-{memory_id}"),
                ground_fingerprint=rec.spec.digest,
                operation="structural_memory.counterexample_demote",
                payload={
                    "coordinate": coordinate,
                    "invalidated_revision": rec.spec.revision,
                    "memory_id": memory_id,
                    "observed_value": observed_value,
                    "predicted_value": predicted,
                },
                result_fn=lambda: {
                    "demoted_to": "PREFIX-GROUND" if new_rev_num is not None else "DEMOTED_NONE",
                    "new_revision": new_rev_num,
                    "new_spec_digest": new_spec_digest,
                },
            )
            return {
                "demoted_to": "PREFIX-GROUND" if new_rev_num is not None else "DEMOTED_NONE",
                "invalidated_revision": rec.spec.revision,
                "journal_event_digest": j_rec.event_digest,
                "new_revision": new_rev_num,
                "new_spec_digest": new_spec_digest,
            }

    def verify_read_receipt(
        self,
        receipt: StructuralReadReceipt,
        *,
        require_active_revision: bool = False,
    ) -> tuple[bool, str]:
        """Verify receipt cryptographic integrity, journal commitment, value conservation, and epistemic validity."""
        with self._lock:
            if receipt.memory_id not in self._revisions:
                return False, "unknown_memory_id"
            history = self._revisions[receipt.memory_id]
            if not (1 <= receipt.revision <= len(history)):
                return False, "unknown_revision"
            rec = history[receipt.revision - 1]
            if rec.spec.digest != receipt.spec_digest or rec.certificate.digest != receipt.certificate_digest:
                return False, "certificate_or_spec_digest_mismatch"
            if rec.spec.context_digest != receipt.context_digest:
                return False, "context_digest_mismatch"
            if (
                not isinstance(receipt.values, (tuple, list))
                or receipt.expanded_intermediate_states != 0
                or receipt.coordinate_start < 0
                or receipt.length < 1
                or len(receipt.values) != receipt.length
            ):
                return False, "malformed_receipt_bounds_or_values"

            if rec.status is CertificateStatus.INVALIDATED_BY_COUNTEREXAMPLE:
                assert rec.invalidated_at_coordinate is not None
                read_end_exclusive = receipt.coordinate_start + receipt.length
                if read_end_exclusive > rec.invalidated_at_coordinate:
                    return (
                        False,
                        f"invalidated_by_counterexample_at_{rec.invalidated_at_coordinate}",
                    )
                if require_active_revision:
                    return False, "revision_invalidated_by_counterexample"

            if require_active_revision and rec.status is not CertificateStatus.ACTIVE:
                return False, f"revision_status_{rec.status.value}"

            if (
                rec.spec.law.verdict == "PREFIX-GROUND"
                and rec.spec.law.first_bad is not None
                and receipt.coordinate_start + receipt.length > rec.spec.law.first_bad
            ):
                return False, "exceeds_prefix_horizon"

            expected_values_digest = domain_digest("vld.read-values.v1", list(receipt.values))
            expected_payload = {
                "coordinate_start": receipt.coordinate_start,
                "length": receipt.length,
                "memory_id": receipt.memory_id,
                "revision": receipt.revision,
            }
            matched_event = None
            for ev in self.journal.events:
                if ev.event_digest == receipt.journal_event_digest:
                    matched_event = ev
                    break
            if matched_event is None:
                return False, "missing_journal_event"
            if (
                matched_event.operation != "structural_memory.read_range"
                or matched_event.ground_fingerprint != receipt.spec_digest
                or matched_event.payload != expected_payload
                or matched_event.result.get("certificate_digest") != receipt.certificate_digest
                or matched_event.result.get("values_digest") != expected_values_digest
            ):
                return False, "journal_receipt_payload_or_values_mismatch"

            for offset, val in enumerate(receipt.values):
                if rec.spec.law.evaluate(receipt.coordinate_start + offset) != val:
                    return False, "law_evaluation_value_mismatch"

            return True, "valid"


__all__ = [
    "CertificateStatus",
    "HorizonExceededError",
    "LawCertificationRefused",
    "StructuralMemoryCertificate",
    "StructuralMemoryError",
    "StructuralMemoryRegistry",
    "StructuralMemorySpec",
    "StructuralReadReceipt",
]
