"""Unit, persistence, idempotency, and tamper-resistance tests for vld.journal."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from vld.canonical import canonical_json, domain_digest
from vld.journal import (
    GENESIS_PREV_DIGEST,
    JOURNAL_DOMAIN,
    HashChainedJournal,
    IdempotencyConflictError,
    JournalIntegrityError,
)


def test_journal_persistence_reload_and_idempotency(tmp_path: Path) -> None:
    journal_file = tmp_path / "state" / "processor.jsonl"
    journal = HashChainedJournal(journal_file)
    assert len(journal) == 0
    assert journal.head_digest == GENESIS_PREV_DIGEST

    calls = 0

    def compute_first() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": 42}

    r0, replayed0 = journal.record(
        request_id="req-1",
        ground_fingerprint="fp-abc",
        operation="navigate",
        payload={"n": 10},
        result_fn=compute_first,
    )
    assert replayed0 is False
    assert calls == 1
    assert r0.seq == 0
    assert r0.prev_digest == GENESIS_PREV_DIGEST

    # Exact retry does not execute result_fn again and returns identical receipt
    r0_retry, replayed_retry = journal.record(
        request_id="req-1",
        ground_fingerprint="fp-abc",
        operation="navigate",
        payload={"n": 10},
        result_fn=compute_first,
    )
    assert replayed_retry is True
    assert calls == 1
    assert r0_retry == r0

    # Second distinct event chains onto r0.event_digest
    r1, replayed1 = journal.record(
        request_id="req-2",
        ground_fingerprint="fp-abc",
        operation="advance",
        payload={"delta": 5},
        result_fn={"value": 47},
    )
    assert replayed1 is False
    assert r1.seq == 1
    assert r1.prev_digest == r0.event_digest

    # No temporary checkpoint files left behind
    assert list(journal_file.parent.glob("*.ckpt.tmp")) == []

    # Fresh reload from disk reconstructs chain and enforces idempotency conflict
    reloaded = HashChainedJournal(journal_file)
    assert len(reloaded) == 2
    assert reloaded.head_digest == r1.event_digest

    with pytest.raises(IdempotencyConflictError):
        reloaded.record(
            request_id="req-1",
            ground_fingerprint="fp-abc",
            operation="navigate",
            payload={"n": 999},  # mutated payload under existing request_id!
            result_fn={"value": 0},
        )


def test_journal_detects_payload_tamper_and_rehashed_midchain_edit(tmp_path: Path) -> None:
    journal_file = tmp_path / "tamper.jsonl"
    journal = HashChainedJournal(journal_file)
    journal.record(
        request_id="r0",
        ground_fingerprint="fp",
        operation="op",
        payload={"x": 1},
        result_fn={"y": 2},
    )
    journal.record(
        request_id="r1",
        ground_fingerprint="fp",
        operation="op",
        payload={"x": 2},
        result_fn={"y": 4},
    )

    lines = [json.loads(line) for line in journal_file.read_text(encoding="utf-8").splitlines()]
    # Adversary mutates event 0 AND recomputes event 0's own digest, leaving event 1's prev_digest broken!
    lines[0]["result"] = {"y": 999}
    body0 = {k: v for k, v in lines[0].items() if k != "event_digest"}
    lines[0]["event_digest"] = domain_digest(JOURNAL_DOMAIN, body0)
    journal_file.write_text("\n".join(canonical_json(row) for row in lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalIntegrityError, match="Broken hash chain"):
        HashChainedJournal(journal_file)


def test_journal_capacity_and_byte_limits(tmp_path: Path) -> None:
    journal_file = tmp_path / "bounded.jsonl"
    journal = HashChainedJournal(journal_file, max_events=1, max_bytes=4096)
    journal.record(
        request_id="only-one",
        ground_fingerprint="fp",
        operation="op",
        payload={},
        result_fn={"ok": True},
    )
    with pytest.raises(JournalIntegrityError, match="capacity"):
        journal.record(
            request_id="second",
            ground_fingerprint="fp",
            operation="op",
            payload={},
            result_fn={"ok": True},
        )
