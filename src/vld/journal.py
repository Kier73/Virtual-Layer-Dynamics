r"""Durable hash-chained receipt journal with idempotency fencing and atomic persistence.

Mathematical contract:
Let events be $e_0, e_1, \dots, e_{k-1}$. Define the genesis anchor $h_{-1} = 0^{64}$.
Each event record binds its sequence index $i$, predecessor digest $h_{i-1}$, request identity $r_i$,
operation kind, Ground fingerprint, input payload, and resulting observation:

  $$h_i = H_{\text{vld.journal-event.v1}}(e_i \setminus \{\text{event\_digest}\}), \qquad e_i.\text{prev\_digest} = h_{i-1}$$

Idempotency & anti-equivocation invariant:
  - Submitting $(r, \text{payload})$ when $r = r_j$ already exists returns $e_j$ iff the request
    semantics (`ground_fingerprint`, `operation`, `payload`) match $e_j$ byte-for-byte.
  - Submitting $r = r_j$ with altered semantics raises `IdempotencyConflictError`.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

from vld.canonical import CanonicalizationError, canonical_json, domain_digest, parse_strict_json

GENESIS_PREV_DIGEST = "0" * 64
JOURNAL_DOMAIN = "vld.journal-event.v1"


class JournalIntegrityError(RuntimeError):
    """Raised when a persisted or supplied journal violates hash-chain or schema invariants."""


class IdempotencyConflictError(ValueError):
    """Raised when a duplicate request_id is submitted with mutated request parameters."""


@dataclass(frozen=True)
class JournalReceipt:
    """Single verified, hash-chained operation receipt."""

    seq: int
    prev_digest: str
    request_id: str
    ground_fingerprint: str
    operation: str
    payload: dict[str, Any]
    result: dict[str, Any]
    event_digest: str

    def body_without_digest(self) -> dict[str, Any]:
        return {
            "ground_fingerprint": self.ground_fingerprint,
            "operation": self.operation,
            "payload": self.payload,
            "prev_digest": self.prev_digest,
            "request_id": self.request_id,
            "result": self.result,
            "seq": self.seq,
        }

    def to_dict(self) -> dict[str, Any]:
        body = self.body_without_digest()
        body["event_digest"] = self.event_digest
        return body

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> JournalReceipt:
        required = {
            "seq",
            "prev_digest",
            "request_id",
            "ground_fingerprint",
            "operation",
            "payload",
            "result",
            "event_digest",
        }
        if set(raw.keys()) != required:
            raise JournalIntegrityError(f"Invalid receipt keyset: {sorted(raw.keys())}")
        if not isinstance(raw["seq"], int) or isinstance(raw["seq"], bool) or raw["seq"] < 0:
            raise JournalIntegrityError("seq must be a non-negative int.")
        for field in ("prev_digest", "request_id", "ground_fingerprint", "operation", "event_digest"):
            if not isinstance(raw[field], str) or not raw[field]:
                raise JournalIntegrityError(f"{field} must be a non-empty str.")
        if not isinstance(raw["payload"], dict) or not isinstance(raw["result"], dict):
            raise JournalIntegrityError("payload and result must be JSON objects.")

        receipt = cls(
            seq=raw["seq"],
            prev_digest=raw["prev_digest"],
            request_id=raw["request_id"],
            ground_fingerprint=raw["ground_fingerprint"],
            operation=raw["operation"],
            payload=raw["payload"],
            result=raw["result"],
            event_digest=raw["event_digest"],
        )
        expected = domain_digest(JOURNAL_DOMAIN, receipt.body_without_digest())
        if receipt.event_digest != expected:
            raise JournalIntegrityError(
                f"Digest mismatch at seq {receipt.seq}: expected {expected}, got {receipt.event_digest}"
            )
        return receipt


class HashChainedJournal:
    """Append-only, atomically persisted receipt journal with idempotency enforcement."""

    def __init__(self, path: Path | str | None = None, *, max_events: int = 100_000, max_bytes: int = 16 * 1024 * 1024) -> None:
        if max_events <= 0 or max_bytes <= 0:
            raise ValueError("max_events and max_bytes must be positive.")
        self._path = Path(path) if path is not None else None
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._events: list[JournalReceipt] = []
        self._by_request_id: dict[str, JournalReceipt] = {}

        if self._path is not None and self._path.exists():
            self._load_and_verify()

    @property
    def events(self) -> tuple[JournalReceipt, ...]:
        return tuple(self._events)

    @property
    def head_digest(self) -> str:
        return self._events[-1].event_digest if self._events else GENESIS_PREV_DIGEST

    def __len__(self) -> int:
        return len(self._events)

    def _load_and_verify(self) -> None:
        assert self._path is not None
        raw_bytes = self._path.read_bytes()
        if len(raw_bytes) > self._max_bytes:
            raise JournalIntegrityError(f"Journal file exceeds max_bytes ({self._max_bytes}).")
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JournalIntegrityError(f"Journal is not valid UTF-8: {exc}") from exc

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > self._max_events:
            raise JournalIntegrityError(f"Journal exceeds max_events ({self._max_events}).")

        events: list[JournalReceipt] = []
        by_req: dict[str, JournalReceipt] = {}
        expected_prev = GENESIS_PREV_DIGEST

        for idx, line in enumerate(lines):
            try:
                parsed = parse_strict_json(line)
            except CanonicalizationError as exc:
                raise JournalIntegrityError(f"Malformed JSON at line {idx + 1}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise JournalIntegrityError(f"Journal line {idx + 1} is not a JSON object.")

            receipt = JournalReceipt.from_dict(parsed)
            if receipt.seq != idx:
                raise JournalIntegrityError(f"Sequence gap at line {idx + 1}: expected {idx}, got {receipt.seq}.")
            if receipt.prev_digest != expected_prev:
                raise JournalIntegrityError(
                    f"Broken hash chain at seq {idx}: expected prev_digest {expected_prev}, got {receipt.prev_digest}."
                )
            if receipt.request_id in by_req:
                raise JournalIntegrityError(f"Duplicate request_id {receipt.request_id!r} persisted in journal.")

            events.append(receipt)
            by_req[receipt.request_id] = receipt
            expected_prev = receipt.event_digest

        self._events = events
        self._by_request_id = by_req

    def verify(self) -> str:
        """Re-verify every in-memory (and persisted, if backed by path) receipt and return head digest."""
        if self._path is not None and self._path.exists():
            self._load_and_verify()
            return self.head_digest

        expected_prev = GENESIS_PREV_DIGEST
        seen: set[str] = set()
        for idx, ev in enumerate(self._events):
            JournalReceipt.from_dict(ev.to_dict())
            if ev.seq != idx or ev.prev_digest != expected_prev or ev.request_id in seen:
                raise JournalIntegrityError(f"In-memory chain invariant failed at seq {idx}.")
            seen.add(ev.request_id)
            expected_prev = ev.event_digest
        return expected_prev

    def record(
        self,
        *,
        request_id: str,
        ground_fingerprint: str,
        operation: str,
        payload: dict[str, Any],
        result_fn: Any,
    ) -> tuple[JournalReceipt, bool]:
        """Record an operation or replay its existing receipt if `request_id` was already committed.

        Returns `(receipt, replayed)` where `replayed` is `True` on exact idempotent retry.
        """
        if not request_id.strip():
            raise ValueError("request_id must be non-empty.")
        if not ground_fingerprint.strip() or not operation.strip():
            raise ValueError("ground_fingerprint and operation must be non-empty.")

        norm_payload = parse_strict_json(canonical_json(payload))
        if not isinstance(norm_payload, dict):
            raise ValueError("payload must be a JSON object.")

        existing = self._by_request_id.get(request_id)
        if existing is not None:
            if (
                existing.ground_fingerprint == ground_fingerprint
                and existing.operation == operation
                and existing.payload == norm_payload
            ):
                return existing, True
            raise IdempotencyConflictError(
                f"request_id {request_id!r} already committed with different Ground, operation, or payload."
            )

        if len(self._events) >= self._max_events:
            raise JournalIntegrityError(f"Journal capacity ({self._max_events} events) exhausted.")

        raw_result = result_fn() if callable(result_fn) else result_fn
        norm_result = parse_strict_json(canonical_json(raw_result))
        if not isinstance(norm_result, dict):
            raise ValueError("result must serialize to a JSON object.")

        seq = len(self._events)
        prev = self.head_digest
        provisional = {
            "ground_fingerprint": ground_fingerprint,
            "operation": operation,
            "payload": norm_payload,
            "prev_digest": prev,
            "request_id": request_id,
            "result": norm_result,
            "seq": seq,
        }
        digest = domain_digest(JOURNAL_DOMAIN, provisional)
        receipt = JournalReceipt(
            seq=seq,
            prev_digest=prev,
            request_id=request_id,
            ground_fingerprint=ground_fingerprint,
            operation=operation,
            payload=norm_payload,
            result=norm_result,
            event_digest=digest,
        )

        candidate_events = [*self._events, receipt]
        if self._path is not None:
            self._atomic_write(candidate_events)

        self._events.append(receipt)
        self._by_request_id[request_id] = receipt
        return receipt, False

    def _atomic_write(self, events: list[JournalReceipt]) -> None:
        assert self._path is not None
        encoded = "".join(canonical_json(ev.to_dict()) + "\n" for ev in events).encode("utf-8")
        if len(encoded) > self._max_bytes:
            raise JournalIntegrityError(f"Serialized journal ({len(encoded)} B) exceeds max_bytes ({self._max_bytes}).")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self._path.name + ".", suffix=".ckpt.tmp", dir=str(self._path.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self._path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
