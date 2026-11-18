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

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterator

if os.name == "nt":
    import msvcrt
else:
    import fcntl

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
    """Append-only, process-safe receipt journal with O(1) incremental append and idempotency fencing."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        max_events: int = 100_000,
        max_bytes: int = 16 * 1024 * 1024,
        lock_timeout_s: float = 15.0,
    ) -> None:
        if max_events <= 0 or max_bytes <= 0:
            raise ValueError("max_events and max_bytes must be positive.")
        if lock_timeout_s <= 0.0:
            raise ValueError("lock_timeout_s must be positive.")
        self._path = Path(path) if path is not None else None
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._lock_timeout_s = lock_timeout_s
        self._thread_lock = threading.RLock()
        self._guard_depth = 0
        self._events: list[JournalReceipt] = []
        self._by_request_id: dict[str, JournalReceipt] = {}
        self._prefix_hasher = hashlib.sha256()
        self._synced_bytes = 0
        self._synced_sha256 = self._prefix_hasher.copy().hexdigest()
        self._ends_with_newline = True

        if self._path is not None and self._path.exists():
            with self._exclusive_guard():
                self._sync_from_disk_unlocked(full=True)

    @property
    def events(self) -> tuple[JournalReceipt, ...]:
        with self._thread_lock:
            return tuple(self._events)

    @property
    def head_digest(self) -> str:
        with self._thread_lock:
            return self._events[-1].event_digest if self._events else GENESIS_PREV_DIGEST

    def __len__(self) -> int:
        with self._thread_lock:
            return len(self._events)

    @contextmanager
    def exclusive_guard(self) -> Iterator[None]:
        """Public reentrant intra-process and cross-process exclusive file guard."""
        with self._exclusive_guard():
            yield

    @contextmanager
    def _exclusive_guard(self) -> Iterator[None]:
        """Acquire intra-process reentrant lock plus cross-process OS lock on `<path>.lock`."""
        with self._thread_lock:
            if self._path is None or self._guard_depth > 0:
                self._guard_depth += 1
                try:
                    yield
                finally:
                    self._guard_depth -= 1
                return

            self._path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self._path.with_name(self._path.name + ".lock")
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            acquired = False
            try:
                if os.name == "nt":
                    # Win32 LockFile locks virtual byte range [0, 1) directly on a 0-byte file;
                    # never write before locking (writing byte 0 without the lock races with a peer's mandatory byte lock!).
                    deadline = time.monotonic() + self._lock_timeout_s
                    delay = 0.001
                    while True:
                        os.lseek(fd, 0, os.SEEK_SET)
                        try:
                            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                            acquired = True
                            break
                        except OSError as exc:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f"Timed out acquiring journal lock {lock_path}") from exc
                            time.sleep(delay)
                            delay = min(0.01, delay * 1.5)
                else:
                    deadline = time.monotonic() + self._lock_timeout_s
                    delay = 0.001
                    while True:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            acquired = True
                            break
                        except BlockingIOError as exc:
                            if time.monotonic() >= deadline:
                                raise TimeoutError(f"Timed out acquiring journal lock {lock_path}") from exc
                            time.sleep(delay)
                            delay = min(0.01, delay * 1.5)
                self._guard_depth += 1
                try:
                    yield
                finally:
                    self._guard_depth -= 1
            finally:
                if acquired:
                    try:
                        if os.name == "nt":
                            os.lseek(fd, 0, os.SEEK_SET)
                            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError:
                        pass
                os.close(fd)

    def _parse_and_validate_lines(
        self,
        lines: list[str],
        *,
        start_seq: int,
        expected_prev: str,
        existing_by_req: dict[str, JournalReceipt],
        line_offset: int,
    ) -> tuple[list[JournalReceipt], dict[str, JournalReceipt]]:
        new_events: list[JournalReceipt] = []
        added_by_req: dict[str, JournalReceipt] = {}
        cur_prev = expected_prev

        for rel_idx, line in enumerate(lines):
            seq = start_seq + rel_idx
            line_no = line_offset + rel_idx + 1
            try:
                parsed = parse_strict_json(line)
            except CanonicalizationError as exc:
                raise JournalIntegrityError(f"Malformed JSON at line {line_no}: {exc}") from exc
            if not isinstance(parsed, dict):
                raise JournalIntegrityError(f"Journal line {line_no} is not a JSON object.")

            receipt = JournalReceipt.from_dict(parsed)
            if receipt.seq != seq:
                raise JournalIntegrityError(f"Sequence gap at line {line_no}: expected {seq}, got {receipt.seq}.")
            if receipt.prev_digest != cur_prev:
                raise JournalIntegrityError(
                    f"Broken hash chain at seq {seq}: expected prev_digest {cur_prev}, got {receipt.prev_digest}."
                )
            if receipt.request_id in existing_by_req or receipt.request_id in added_by_req:
                raise JournalIntegrityError(f"Duplicate request_id {receipt.request_id!r} persisted in journal.")

            new_events.append(receipt)
            added_by_req[receipt.request_id] = receipt
            cur_prev = receipt.event_digest

        return new_events, added_by_req

    def _sync_from_disk_unlocked(self, *, full: bool = False) -> None:
        assert self._path is not None
        if not self._path.exists():
            if self._synced_bytes > 0 or self._events:
                raise JournalIntegrityError("Persisted journal file was removed after events were committed.")
            return

        raw_bytes = self._path.read_bytes()
        if len(raw_bytes) > self._max_bytes:
            raise JournalIntegrityError(f"Journal file exceeds max_bytes ({self._max_bytes}).")

        # Fast incremental path: previously verified byte prefix is unchanged; verify only new suffix lines.
        if not full and self._synced_bytes > 0 and len(raw_bytes) >= self._synced_bytes:
            prefix = raw_bytes[: self._synced_bytes]
            if hashlib.sha256(prefix).hexdigest() == self._synced_sha256:
                if len(raw_bytes) == self._synced_bytes:
                    return
                suffix = raw_bytes[self._synced_bytes :]
                try:
                    suffix_text = suffix.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise JournalIntegrityError(f"Journal suffix is not valid UTF-8: {exc}") from exc

                new_lines = [line for line in suffix_text.splitlines() if line.strip()]
                if len(self._events) + len(new_lines) > self._max_events:
                    raise JournalIntegrityError(f"Journal exceeds max_events ({self._max_events}).")

                new_events, added_by_req = self._parse_and_validate_lines(
                    new_lines,
                    start_seq=len(self._events),
                    expected_prev=self.head_digest,
                    existing_by_req=self._by_request_id,
                    line_offset=len(self._events),
                )
                self._events.extend(new_events)
                self._by_request_id.update(added_by_req)
                self._prefix_hasher.update(suffix)
                self._synced_bytes = len(raw_bytes)
                self._synced_sha256 = self._prefix_hasher.copy().hexdigest()
                self._ends_with_newline = raw_bytes.endswith(b"\n")
                return

        # Full chain verification from genesis (initial load or prefix modification check).
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JournalIntegrityError(f"Journal is not valid UTF-8: {exc}") from exc

        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) > self._max_events:
            raise JournalIntegrityError(f"Journal exceeds max_events ({self._max_events}).")

        events, by_req = self._parse_and_validate_lines(
            lines,
            start_seq=0,
            expected_prev=GENESIS_PREV_DIGEST,
            existing_by_req={},
            line_offset=0,
        )

        # Anti-rollback check if this instance already witnessed committed events.
        if self._events:
            if len(events) < len(self._events) or any(
                events[i].event_digest != self._events[i].event_digest for i in range(len(self._events))
            ):
                raise JournalIntegrityError(
                    "Persisted journal diverged from or truncated previously committed lineage."
                )

        self._events = events
        self._by_request_id = by_req
        self._prefix_hasher = hashlib.sha256(raw_bytes)
        self._synced_bytes = len(raw_bytes)
        self._synced_sha256 = self._prefix_hasher.copy().hexdigest()
        self._ends_with_newline = (len(raw_bytes) == 0) or raw_bytes.endswith(b"\n")

    def verify(self) -> str:
        """Re-verify every in-memory (and persisted, if backed by path) receipt and return head digest."""
        with self._exclusive_guard():
            if self._path is not None and (self._path.exists() or self._synced_bytes > 0):
                self._sync_from_disk_unlocked(full=True)
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
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty str.")
        if (
            not isinstance(ground_fingerprint, str)
            or not ground_fingerprint.strip()
            or not isinstance(operation, str)
            or not operation.strip()
        ):
            raise ValueError("ground_fingerprint and operation must be non-empty strings.")

        norm_payload = parse_strict_json(canonical_json(payload))
        if not isinstance(norm_payload, dict):
            raise ValueError("payload must be a JSON object.")

        with self._exclusive_guard():
            if self._path is not None:
                self._sync_from_disk_unlocked(full=False)

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

            if request_id in self._by_request_id:
                raise IdempotencyConflictError(
                    f"Reentrant operation already committed request_id {request_id!r}."
                )
            if len(self._events) >= self._max_events:
                raise JournalIntegrityError(f"Journal capacity ({self._max_events} events) exhausted.")

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

            if self._path is not None:
                self._append_event_unlocked(receipt)

            self._events.append(receipt)
            self._by_request_id[request_id] = receipt
            return receipt, False

    def _append_event_unlocked(self, receipt: JournalReceipt) -> None:
        assert self._path is not None
        prefix_pad = b"" if self._ends_with_newline else b"\n"
        encoded_line = prefix_pad + (canonical_json(receipt.to_dict()) + "\n").encode("utf-8")
        if self._synced_bytes + len(encoded_line) > self._max_bytes:
            raise JournalIntegrityError(
                f"Serialized journal ({self._synced_bytes + len(encoded_line)} B) exceeds max_bytes ({self._max_bytes})."
            )

        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "ab") as handle:
            prior_size = handle.tell()
            try:
                handle.write(encoded_line)
                handle.flush()
                os.fsync(handle.fileno())
            except Exception:
                try:
                    handle.truncate(prior_size)
                    handle.flush()
                except OSError:
                    pass
                raise

        self._prefix_hasher.update(encoded_line)
        self._synced_bytes += len(encoded_line)
        self._synced_sha256 = self._prefix_hasher.copy().hexdigest()
        self._ends_with_newline = True
