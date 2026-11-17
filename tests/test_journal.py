"""Unit, persistence, idempotency, and tamper-resistance tests for vld.journal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
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


def test_cross_instance_sync_and_live_tamper_refusal(tmp_path: Path) -> None:
    journal_file = tmp_path / "shared.jsonl"
    j1 = HashChainedJournal(journal_file)
    j2 = HashChainedJournal(journal_file)

    r1, rep1 = j1.record(
        request_id="req-a",
        ground_fingerprint="fp",
        operation="op",
        payload={"v": 1},
        result_fn={"out": 10},
    )
    assert not rep1 and r1.seq == 0

    # j2 must sync r1 from disk before recording req-b (never overwriting r1!)
    r2, rep2 = j2.record(
        request_id="req-b",
        ground_fingerprint="fp",
        operation="op",
        payload={"v": 2},
        result_fn={"out": 20},
    )
    assert not rep2 and r2.seq == 1 and r2.prev_digest == r1.event_digest

    # Cross-instance idempotent retry on j1 sees req-b committed by j2
    r2_retry, rep_retry = j1.record(
        request_id="req-b",
        ground_fingerprint="fp",
        operation="op",
        payload={"v": 2},
        result_fn=lambda: (_ for _ in ()).throw(AssertionError("Should not execute on replay")),
    )
    assert rep_retry and r2_retry == r2
    assert len(j1) == 2

    # Live prefix tampering on disk while j1 is open must be caught on next record()
    lines = journal_file.read_text(encoding="utf-8").splitlines()
    tampered0 = json.loads(lines[0])
    tampered0["result"] = {"out": 999}
    lines[0] = canonical_json(tampered0)
    journal_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(JournalIntegrityError):
        j1.record(
            request_id="req-c",
            ground_fingerprint="fp",
            operation="op",
            payload={"v": 3},
            result_fn={"out": 30},
        )


def test_multiprocess_concurrent_journal_writes(tmp_path: Path) -> None:
    journal_file = tmp_path / "concurrent.jsonl"
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    worker_code = (
        "import sys\n"
        "from vld.journal import HashChainedJournal\n"
        "path, worker_id = sys.argv[1], int(sys.argv[2])\n"
        "j = HashChainedJournal(path)\n"
        "for step in range(8):\n"
        "    j.record(\n"
        "        request_id=f'w{worker_id}-s{step}',\n"
        "        ground_fingerprint='fp-mp',\n"
        "        operation='step',\n"
        "        payload={'worker': worker_id, 'step': step},\n"
        "        result_fn={'ack': step},\n"
        "    )\n"
        "# Shared idempotent request across all processes\n"
        "j.record(\n"
        "    request_id='shared-anchor',\n"
        "    ground_fingerprint='fp-mp',\n"
        "    operation='anchor',\n"
        "    payload={'epoch': 1},\n"
        "    result_fn={'anchored': True},\n"
        ")\n"
    )
    env = dict(os.environ)
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_pp}" if existing_pp else src_dir

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", worker_code, str(journal_file), str(wid)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for wid in range(4)
    ]
    for p in procs:
        out, err = p.communicate(timeout=20)
        assert p.returncode == 0, f"Worker failed: stdout={out!r} stderr={err!r}"

    verified = HashChainedJournal(journal_file)
    # 4 workers * 8 unique events + 1 deduplicated shared-anchor = 33 total events
    assert len(verified) == 33
    assert verified.verify() == verified.head_digest
