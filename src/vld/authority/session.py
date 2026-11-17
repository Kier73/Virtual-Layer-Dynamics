r"""Phase 3 — M3 Quorum-Fenced Computational Sessions & Portable Virtual Processor Continuity (`vld.authority.session`).

Mathematical contract:
1. **Receiver-Owned CAS Commit over M1 State Pages:**
   State revision $r \to r+1$ commits iff `(expected_revision == r) and (expected_state_digest == H(s_r))`,
   storing immutable state bytes as an M1 `PagedByteManifest` and anchoring its manifest digest in M3
   strict-majority ($2q > N$) `QuorumLineageCoordinator`.
2. **Exact-Once Crash Continuity Across `os._exit` & Host Replacement (`G22`):**
   Before invoking quorum sealing for slot $r+1$, the processor atomically stages a write-ahead
   pending transition record binding `(operation_id, r+1, next_manifest)`. Upon hard-kill at *any*
   instruction boundary (pre-quorum or post-quorum) and rehoming onto a fresh host under fenced
   term $t' > t$, recovery reconciles pending intent with the quorum high-water mark so semantic
   transitions execute **at most once** while stale zombie hosts in term $t < t'$ are deterministically refused.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from vld.authority.quorum import (
    QuorumCommitCertificate,
    QuorumError,
    QuorumLineageCoordinator,
    RollbackDetectedError,
    StaleTermError,
)
from vld.authority.temporal import TemporalCoordinate, VectorClock
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.journal import GENESIS_PREV_DIGEST
from vld.memory import PagedByteManifest, PagedObjectStore

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SessionError(ValueError):
    """Base error for M3 computational session validation or persistence."""


class SessionConflictError(SessionError):
    """Raised on CAS state-digest / revision mismatch or conflicting `operation_id` replay."""


class StaleEpochError(SessionError):
    """Raised when a stale host/processor attempts to mutate state under a superseded fencing term."""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = canonical_bytes(dict(payload))
    with tmp.open("wb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class SessionCheckpoint:
    """Immutable M3 session revision descriptor sealed by a $2q > N$ `QuorumCommitCertificate`."""

    session_id: str
    processor_id: str
    host_id: str
    ground_fingerprint: str
    revision: int
    term: int
    predecessor_checkpoint_digest: str
    state_manifest: PagedByteManifest
    operation_id: str
    input_digest: str
    temporal_coord: TemporalCoordinate
    quorum_certificate: QuorumCommitCertificate
    checkpoint_digest: str = ""

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.processor_id.strip() or not self.host_id.strip():
            raise SessionError("session_id, processor_id, and host_id must be non-empty.")
        for fname in ("ground_fingerprint", "predecessor_checkpoint_digest", "input_digest"):
            val = getattr(self, fname)
            if not isinstance(val, str) or _SHA256_RE.fullmatch(val) is None:
                raise SessionError(f"{fname} must be a 64-char SHA-256 hex digest.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise SessionError("revision must be a positive integer.")
        if isinstance(self.term, bool) or not isinstance(self.term, int) or self.term <= 0:
            raise SessionError("term must be a positive integer.")
        if self.quorum_certificate.proposal.slot != self.revision:
            raise SessionError("QuorumCommitCertificate slot disagrees with checkpoint revision.")
        if self.quorum_certificate.proposal.state_manifest_digest != self.state_manifest.manifest_digest:
            raise SessionError("QuorumCommitCertificate state_manifest_digest disagrees with M1 manifest.")
        computed = domain_digest("vld.session-checkpoint.v1", self._body())
        if self.checkpoint_digest and self.checkpoint_digest != computed:
            raise SessionError("SessionCheckpoint digest mismatch.")
        object.__setattr__(self, "checkpoint_digest", computed)

    @property
    def state_digest(self) -> str:
        return self.state_manifest.content_digest

    def _body(self) -> dict[str, Any]:
        return {
            "ground_fingerprint": self.ground_fingerprint,
            "host_id": self.host_id,
            "input_digest": self.input_digest,
            "operation_id": self.operation_id,
            "predecessor_checkpoint_digest": self.predecessor_checkpoint_digest,
            "processor_id": self.processor_id,
            "quorum_certificate": self.quorum_certificate.to_dict(),
            "revision": self.revision,
            "schema": "vld.session-checkpoint.v1",
            "session_id": self.session_id,
            "state_manifest": self.state_manifest.to_dict(),
            "temporal_coord": self.temporal_coord.to_dict(),
            "term": self.term,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "checkpoint_digest": self.checkpoint_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionCheckpoint:
        req = {
            "checkpoint_digest",
            "ground_fingerprint",
            "host_id",
            "input_digest",
            "operation_id",
            "predecessor_checkpoint_digest",
            "processor_id",
            "quorum_certificate",
            "revision",
            "schema",
            "session_id",
            "state_manifest",
            "temporal_coord",
            "term",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.session-checkpoint.v1":
            raise SessionError("Invalid SessionCheckpoint payload.")
        return cls(
            session_id=data["session_id"],
            processor_id=data["processor_id"],
            host_id=data["host_id"],
            ground_fingerprint=data["ground_fingerprint"],
            revision=data["revision"],
            term=data["term"],
            predecessor_checkpoint_digest=data["predecessor_checkpoint_digest"],
            state_manifest=PagedByteManifest.from_dict(data["state_manifest"]),
            operation_id=data["operation_id"],
            input_digest=data["input_digest"],
            temporal_coord=TemporalCoordinate.from_dict(data["temporal_coord"]),
            quorum_certificate=QuorumCommitCertificate.from_dict(data["quorum_certificate"]),
            checkpoint_digest=data["checkpoint_digest"],
        )


class ComputationalSession:
    """Durable M3 session state machine combining M1 byte pages, CAS commits, and $2q > N$ quorum fencing."""

    def __init__(
        self,
        *,
        session_id: str,
        processor_id: str,
        host_id: str,
        ground_fingerprint: str,
        store: PagedObjectStore,
        coordinator: QuorumLineageCoordinator,
        session_dir: str | Path,
        term: int = 1,
        scheduler_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.processor_id = processor_id
        self.host_id = host_id
        self.ground_fingerprint = ground_fingerprint
        self.store = store
        self.coordinator = coordinator
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.term = term
        self.scheduler_id = scheduler_id or f"sched:{session_id}"
        self._checkpoints_by_rev: dict[int, SessionCheckpoint] = {}
        self._ops_index: dict[str, tuple[str, int]] = {}  # operation_id -> (input_digest, revision)
        self._load_disk_checkpoints()

    def _ckpt_path(self, revision: int) -> Path:
        return self.session_dir / f"checkpoint_{revision:06d}.json"

    def _latest_path(self) -> Path:
        return self.session_dir / "latest.json"

    def _pending_path(self) -> Path:
        return self.session_dir / "pending_transition.json"

    def _load_disk_checkpoints(self) -> None:
        for p in sorted(self.session_dir.glob("checkpoint_*.json")):
            ckpt = SessionCheckpoint.from_dict(parse_strict_json(p.read_bytes()))
            self._checkpoints_by_rev[ckpt.revision] = ckpt
            self._ops_index[ckpt.operation_id] = (ckpt.input_digest, ckpt.revision)

    def _persist_checkpoint(self, ckpt: SessionCheckpoint) -> None:
        _atomic_write_json(self._ckpt_path(ckpt.revision), ckpt.to_dict())
        _atomic_write_json(self._latest_path(), ckpt.to_dict())
        self._checkpoints_by_rev[ckpt.revision] = ckpt
        self._ops_index[ckpt.operation_id] = (ckpt.input_digest, ckpt.revision)

    @property
    def current_checkpoint(self) -> SessionCheckpoint | None:
        self._load_disk_checkpoints()
        if not self._checkpoints_by_rev:
            return None
        max_rev = max(self._checkpoints_by_rev.keys())
        ckpt = self._checkpoints_by_rev[max_rev]
        # Verify against quorum high-water mark so rolled-back disk files are caught!
        self.coordinator.verify_fresh_checkpoint(
            slot=ckpt.revision,
            state_manifest_digest=ckpt.state_manifest.manifest_digest,
        )
        return ckpt

    def verify_latest_pointer(self) -> SessionCheckpoint:
        """Verify `latest.json` against both M1 store content and M3 quorum high-water mark."""
        lp = self._latest_path()
        if not lp.exists():
            raise RollbackDetectedError("Session latest.json is missing.")
        ckpt = SessionCheckpoint.from_dict(parse_strict_json(lp.read_bytes()))
        self.coordinator.verify_fresh_checkpoint(
            slot=ckpt.revision,
            state_manifest_digest=ckpt.state_manifest.manifest_digest,
        )
        raw = self.store.read_bytes(ckpt.state_manifest)
        if hashlib.sha256(raw).hexdigest() != ckpt.state_digest:
            raise SessionError("Corrupted M1 state bytes for SessionCheckpoint.")
        return ckpt

    def read_state_bytes(self) -> bytes:
        ckpt = self.current_checkpoint
        if ckpt is None:
            raise SessionError("Session has not been initialized.")
        return self.store.read_bytes(ckpt.state_manifest)

    def read_state_json(self) -> Any:
        return parse_strict_json(self.read_state_bytes())

    def initialize(
        self,
        initial_state: bytes | Mapping[str, Any],
        *,
        active_member_ids: Sequence[str] | None = None,
    ) -> SessionCheckpoint:
        """Initialize revision 1 of the session if not already initialized."""
        if self._checkpoints_by_rev:
            raise SessionConflictError("Session is already initialized.")
        raw = canonical_bytes(dict(initial_state)) if isinstance(initial_state, Mapping) else bytes(initial_state)
        manifest = self.store.put_bytes(raw, media_type="application/json")
        coord = TemporalCoordinate(
            scheduler_id=self.scheduler_id,
            epoch=self.term,
            clock=VectorClock().advance(self.processor_id, 1),
        )
        input_digest = hashlib.sha256(raw).hexdigest()
        try:
            self.coordinator.fence_term(self.term, active_member_ids=active_member_ids)
            cert = self.coordinator.propose_and_seal(
                slot=1,
                term=self.term,
                predecessor_manifest_digest=GENESIS_PREV_DIGEST,
                state_manifest_digest=manifest.manifest_digest,
                proposer_id=self.host_id,
                operation_id="session.initialize",
                active_member_ids=active_member_ids,
            )
        except StaleTermError as exc:
            raise StaleEpochError(str(exc)) from exc

        ckpt = SessionCheckpoint(
            session_id=self.session_id,
            processor_id=self.processor_id,
            host_id=self.host_id,
            ground_fingerprint=self.ground_fingerprint,
            revision=1,
            term=self.term,
            predecessor_checkpoint_digest=GENESIS_PREV_DIGEST,
            state_manifest=manifest,
            operation_id="session.initialize",
            input_digest=input_digest,
            temporal_coord=coord,
            quorum_certificate=cert,
        )
        self._persist_checkpoint(ckpt)
        return ckpt

    def reconcile_pending(self, *, active_member_ids: Sequence[str] | None = None) -> SessionCheckpoint | None:
        """Complete or finalize any crash-interrupted staged transition under the active fencing term."""
        self._load_disk_checkpoints()
        pp = self._pending_path()
        if not pp.exists():
            return self.current_checkpoint

        pending = parse_strict_json(pp.read_bytes())
        slot = int(pending["slot"])
        next_manifest = PagedByteManifest.from_dict(pending["state_manifest"])
        t_coord = TemporalCoordinate.from_dict(pending["temporal_coord"])
        # Re-scope temporal coordinate epoch if recovered under a higher term
        if t_coord.epoch != self.term:
            t_coord = TemporalCoordinate(
                scheduler_id=t_coord.scheduler_id,
                epoch=self.term,
                clock=t_coord.clock.advance(self.processor_id, 1),
            )

        hw_cert = self.coordinator.high_water_certificate
        try:
            self.coordinator.fence_term(self.term, active_member_ids=active_member_ids)
            if hw_cert is not None and hw_cert.proposal.slot == slot:
                if hw_cert.proposal.state_manifest_digest != next_manifest.manifest_digest:
                    raise SessionConflictError("Staged pending transition conflicts with quorum-sealed slot.")
                cert = hw_cert
            elif (hw_cert is None and slot == 1) or (hw_cert is not None and hw_cert.proposal.slot + 1 == slot):
                cert = self.coordinator.propose_and_seal(
                    slot=slot,
                    term=self.term,
                    predecessor_manifest_digest=str(pending["predecessor_manifest_digest"]),
                    state_manifest_digest=next_manifest.manifest_digest,
                    proposer_id=self.host_id,
                    operation_id=str(pending["operation_id"]),
                    active_member_ids=active_member_ids,
                )
            else:
                raise RollbackDetectedError(f"Cannot reconcile pending slot {slot} against quorum state.")
        except StaleTermError as exc:
            raise StaleEpochError(str(exc)) from exc

        ckpt = SessionCheckpoint(
            session_id=self.session_id,
            processor_id=self.processor_id,
            host_id=self.host_id,
            ground_fingerprint=self.ground_fingerprint,
            revision=slot,
            term=self.term,
            predecessor_checkpoint_digest=str(pending["predecessor_checkpoint_digest"]),
            state_manifest=next_manifest,
            operation_id=str(pending["operation_id"]),
            input_digest=str(pending["input_digest"]),
            temporal_coord=t_coord,
            quorum_certificate=cert,
        )
        self._persist_checkpoint(ckpt)
        if pp.exists():
            pp.unlink()
        return ckpt

    def cas_transition(
        self,
        *,
        operation_id: str,
        expected_revision: int,
        expected_state_digest: str,
        input_payload: Any,
        transition_fn: Callable[[bytes], bytes | Mapping[str, Any]],
        active_member_ids: Sequence[str] | None = None,
        _fault_hook_after_stage: Callable[[], None] | None = None,
        _fault_hook_after_quorum: Callable[[], None] | None = None,
    ) -> tuple[SessionCheckpoint, bool]:
        """Execute a deterministic CAS state transition anchored by $2q > N$ quorum.

        Returns `(checkpoint, replayed_from_log)`.
        """
        self._load_disk_checkpoints()
        input_digest = domain_digest("vld.session-op-input.v1", parse_strict_json(canonical_json(input_payload)))

        # Idempotent replay check by operation_id
        if operation_id in self._ops_index:
            prev_in_digest, committed_rev = self._ops_index[operation_id]
            if prev_in_digest != input_digest:
                raise SessionConflictError(
                    f"Idempotency conflict on operation_id {operation_id!r}: input digest differs."
                )
            return self._checkpoints_by_rev[committed_rev], True

        if self._pending_path().exists():
            raise SessionConflictError("Unreconciled pending transition exists; call reconcile_pending() first.")

        try:
            self.coordinator.fence_term(self.term, active_member_ids=active_member_ids)
        except StaleTermError as exc:
            raise StaleEpochError(str(exc)) from exc

        cur = self.current_checkpoint
        if cur is None:
            raise SessionError("Cannot run cas_transition before initialize().")
        if expected_revision != cur.revision:
            raise SessionConflictError(
                f"CAS revision mismatch: expected {expected_revision}, actual {cur.revision}."
            )
        if expected_state_digest != cur.state_digest:
            raise SessionConflictError(
                f"CAS state_digest mismatch: expected {expected_state_digest}, actual {cur.state_digest}."
            )

        cur_bytes = self.store.read_bytes(cur.state_manifest)
        next_out = transition_fn(cur_bytes)
        next_bytes = canonical_bytes(dict(next_out)) if isinstance(next_out, Mapping) else bytes(next_out)
        next_manifest = self.store.put_bytes(next_bytes, media_type=cur.state_manifest.media_type)

        base_clock = cur.temporal_coord.clock.advance(self.processor_id, 1)
        next_coord = TemporalCoordinate(
            scheduler_id=self.scheduler_id,
            epoch=self.term,
            clock=base_clock,
        )
        next_slot = cur.revision + 1

        pending_doc = {
            "input_digest": input_digest,
            "operation_id": operation_id,
            "predecessor_checkpoint_digest": cur.checkpoint_digest,
            "predecessor_manifest_digest": cur.state_manifest.manifest_digest,
            "schema": "vld.session-pending.v1",
            "slot": next_slot,
            "state_manifest": next_manifest.to_dict(),
            "temporal_coord": next_coord.to_dict(),
            "term": self.term,
        }
        _atomic_write_json(self._pending_path(), pending_doc)

        if _fault_hook_after_stage is not None:
            _fault_hook_after_stage()

        try:
            cert = self.coordinator.propose_and_seal(
                slot=next_slot,
                term=self.term,
                predecessor_manifest_digest=cur.state_manifest.manifest_digest,
                state_manifest_digest=next_manifest.manifest_digest,
                proposer_id=self.host_id,
                operation_id=operation_id,
                active_member_ids=active_member_ids,
            )
        except StaleTermError as exc:
            raise StaleEpochError(str(exc)) from exc
        except QuorumError as exc:
            raise SessionError(str(exc)) from exc

        if _fault_hook_after_quorum is not None:
            _fault_hook_after_quorum()

        ckpt = SessionCheckpoint(
            session_id=self.session_id,
            processor_id=self.processor_id,
            host_id=self.host_id,
            ground_fingerprint=self.ground_fingerprint,
            revision=next_slot,
            term=self.term,
            predecessor_checkpoint_digest=cur.checkpoint_digest,
            state_manifest=next_manifest,
            operation_id=operation_id,
            input_digest=input_digest,
            temporal_coord=next_coord,
            quorum_certificate=cert,
        )
        self._persist_checkpoint(ckpt)
        if self._pending_path().exists():
            self._pending_path().unlink()
        return ckpt, False


class VirtualProcessor:
    """Portable Virtual Processor identity surviving complete host destruction and `os._exit` recovery."""

    def __init__(self, session: ComputationalSession) -> None:
        self.session = session

    @property
    def processor_id(self) -> str:
        return self.session.processor_id

    @property
    def host_id(self) -> str:
        return self.session.host_id

    @property
    def term(self) -> int:
        return self.session.term

    def step_json(
        self,
        operation_id: str,
        input_payload: Mapping[str, Any],
        step_fn: Callable[[dict[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        *,
        active_member_ids: Sequence[str] | None = None,
        _fault_hook_after_stage: Callable[[], None] | None = None,
        _fault_hook_after_quorum: Callable[[], None] | None = None,
    ) -> tuple[SessionCheckpoint, bool]:
        """Advance JSON-encoded processor state by one CAS + quorum-fenced transition."""
        if operation_id in self.session._ops_index:
            return self.session.cas_transition(
                operation_id=operation_id,
                expected_revision=0,
                expected_state_digest="0" * 64,
                input_payload=input_payload,
                transition_fn=lambda b: b,
                active_member_ids=active_member_ids,
            )
        try:
            self.session.coordinator.fence_term(self.session.term, active_member_ids=active_member_ids)
        except StaleTermError as exc:
            raise StaleEpochError(str(exc)) from exc
        cur = self.session.current_checkpoint
        if cur is None:
            raise SessionError("VirtualProcessor session is not initialized.")

        def _apply(raw_bytes: bytes) -> Mapping[str, Any]:
            state_obj = parse_strict_json(raw_bytes)
            if not isinstance(state_obj, dict):
                raise SessionError("VirtualProcessor state root must be a JSON object.")
            return step_fn(state_obj, input_payload)

        return self.session.cas_transition(
            operation_id=operation_id,
            expected_revision=cur.revision,
            expected_state_digest=cur.state_digest,
            input_payload=input_payload,
            transition_fn=_apply,
            active_member_ids=active_member_ids,
            _fault_hook_after_stage=_fault_hook_after_stage,
            _fault_hook_after_quorum=_fault_hook_after_quorum,
        )

    @classmethod
    def recover_on_new_host(
        cls,
        *,
        session_id: str,
        processor_id: str,
        new_host_id: str,
        ground_fingerprint: str,
        store: PagedObjectStore,
        coordinator: QuorumLineageCoordinator,
        session_dir: str | Path,
        active_member_ids: Sequence[str] | None = None,
    ) -> VirtualProcessor:
        r"""Rehome processor onto `new_host_id`, fence off prior host epoch ($t' = t_{\text{hw}} + 1$), and reconcile pending WAL."""
        hw = coordinator.high_water_certificate
        base_term = hw.proposal.term if hw is not None else 1
        fenced_term = base_term + 1
        coordinator.fence_term(fenced_term, active_member_ids=active_member_ids)
        sess = ComputationalSession(
            session_id=session_id,
            processor_id=processor_id,
            host_id=new_host_id,
            ground_fingerprint=ground_fingerprint,
            store=store,
            coordinator=coordinator,
            session_dir=session_dir,
            term=fenced_term,
        )
        sess.reconcile_pending(active_member_ids=active_member_ids)
        return cls(sess)


__all__ = [
    "ComputationalSession",
    "SessionCheckpoint",
    "SessionConflictError",
    "SessionError",
    "StaleEpochError",
    "VirtualProcessor",
]
