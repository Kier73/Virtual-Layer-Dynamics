"""Phase 3 tests: Scheduler-scoped temporal coordinates (`G13`), decoupled provenance (`G21`),
$2q > N$ quorum fencing, rollback refusal, and `os._exit` Virtual Processor crash continuity (`G22`).
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from vld.authority import (
    CausalOrder,
    ClockBridgeReceipt,
    ComputationalSession,
    CrossSchedulerComparisonError,
    EncounterEnvelope,
    InterpretationEnvelope,
    QuorumAuthorityMember,
    QuorumEquivocationError,
    QuorumInsufficientError,
    QuorumLineageCoordinator,
    QuorumMembership,
    QuorumProposal,
    RollbackDetectedError,
    SessionConflictError,
    StaleEpochError,
    StaleTermError,
    TemporalCoordinate,
    TemporalError,
    VectorClock,
    VirtualProcessor,
    decode_temporal_f64,
    decode_temporal_int,
    encode_temporal_f64,
    encode_temporal_int,
)
from vld.canonical import domain_digest
from vld.ground import LensKind
from vld.memory import DiskPageBacking, PagedObjectStore
from vld.triad import EpistemicStatus


def test_lossless_temporal_envelopes_and_g13_cross_scheduler_guard() -> None:
    # 1. Arbitrary-precision integer & exact IEEE-754 binary64 (-0.0, subnormals) round-trip
    huge = -(10**80) + 73
    env_i = encode_temporal_int(huge)
    assert decode_temporal_int(env_i) == huge

    with pytest.raises(TemporalError):
        encode_temporal_int(True)  # type: ignore[arg-type]
    with pytest.raises(TemporalError):
        decode_temporal_int({**env_i, "value": "007"})

    for f_val in (-0.0, 0.0, float.fromhex("0x0.0000000000001p-1022"), 3.141592653589793):
        dec = decode_temporal_f64(encode_temporal_f64(f_val))
        assert dec == f_val and math.copysign(1.0, dec) == math.copysign(1.0, f_val)

    with pytest.raises(TemporalError):
        encode_temporal_f64(float("nan"))
    with pytest.raises(TemporalError):
        encode_temporal_f64(float("inf"))

    # 2. Same-scheduler causal partial order vs CrossSchedulerComparisonError (G13)
    c_a1 = TemporalCoordinate("sched-alpha", 1, VectorClock.from_mapping({"p0": 2, "p1": 1}), wall_hint_f64=-0.0)
    c_a2 = c_a1.advance("p1", 3)
    c_branch = TemporalCoordinate("sched-alpha", 1, VectorClock.from_mapping({"p0": 3, "p1": 0}))

    assert c_a1.compare(c_a2) is CausalOrder.BEFORE
    assert c_a2.compare(c_a1) is CausalOrder.AFTER
    assert c_a1.compare(c_a1) is CausalOrder.EQUAL
    assert c_a2.compare(c_branch) is CausalOrder.CONCURRENT
    assert TemporalCoordinate.from_dict(c_a1.to_dict()) == c_a1

    c_beta = TemporalCoordinate("sched-beta", 1, VectorClock.from_mapping({"q0": 5}))
    with pytest.raises(CrossSchedulerComparisonError):
        c_a1.compare(c_beta)
    with pytest.raises(CrossSchedulerComparisonError):
        c_a1.receive_merge(c_beta, "p0")

    bridge = ClockBridgeReceipt(
        source_scheduler_id="sched-alpha",
        source_epoch=1,
        source_anchor=c_a2.clock,
        target_scheduler_id="sched-beta",
        target_epoch=1,
        target_anchor=VectorClock.from_mapping({"q0": 4}),
    )
    assert c_a1.compare(c_beta, bridge=bridge) is CausalOrder.BEFORE
    assert c_beta.compare(c_a1, bridge=bridge) is CausalOrder.AFTER


def test_decoupled_encounter_vs_interpretation_provenance_g21() -> None:
    coord = TemporalCoordinate("sensor-sched", 0, VectorClock.from_mapping({"edge0": 1}))
    raw_packet = b"\x10\x20\x30\x40"
    enc = EncounterEnvelope.capture(
        encounter_id="enc-001",
        observer_id="probe.port443",
        temporal_coord=coord,
        raw_bytes=raw_packet,
        channel_meta={"transport": "udp", "ttl": 54},
    )
    orig_enc_dict = enc.to_dict()

    g1 = domain_digest("test.g", "ground-v1")
    g2 = domain_digest("test.g", "ground-v2-demoted")
    interp_1 = InterpretationEnvelope(
        encounter_digest=enc.encounter_digest,
        ground_fingerprint=g1,
        lens_kind=LensKind.RAW_Z256,
        epistemic_status=EpistemicStatus.CERTIFIED,
        interpreted_value=[16, 32, 48, 64],
        interpreter_id="interp-A",
    )
    interp_2 = InterpretationEnvelope(
        encounter_digest=enc.encounter_digest,
        ground_fingerprint=g2,
        lens_kind=LensKind.UINT_BE,
        epistemic_status=EpistemicStatus.HYPOTHESIS,
        interpreted_value=0x10203040,
        interpreter_id="interp-B",
    )
    assert interp_1.verify_against_encounter(enc)
    assert interp_2.verify_against_encounter(enc)
    assert interp_1.interpretation_digest != interp_2.interpretation_digest
    # Re-interpreting or demoting never alters the EncounterEnvelope!
    assert EncounterEnvelope.from_dict(orig_enc_dict) == enc


def _build_5node_cluster(root: Path) -> tuple[QuorumMembership, list[QuorumAuthorityMember], dict[str, bytes], QuorumLineageCoordinator]:
    mids = tuple(f"auth-{i}" for i in range(5))
    membership = QuorumMembership("lineage.vp01", mids)
    secrets = {mid: f"secret-key-for-{mid}-0123456789".encode("utf-8") for mid in mids}
    members = [
        QuorumAuthorityMember(mid, secrets[mid], journal_path=root / "members" / f"{mid}.jsonl")
        for mid in mids
    ]
    coord = QuorumLineageCoordinator(
        membership,
        members,
        secrets,
        ledger_path=root / "coordinator_ledger.jsonl",
    )
    return membership, members, secrets, coord


def test_quorum_strict_majority_equivocation_and_rollback_fencing(tmp_path: Path) -> None:
    membership, members, _, coord = _build_5node_cluster(tmp_path)
    assert membership.size == 5 and membership.quorum_size == 3  # 2*3 = 6 > 5

    m1_digest_1 = domain_digest("test.m1", "state-1")
    m1_digest_2 = domain_digest("test.m1", "state-2")
    m1_fork = domain_digest("test.m1", "forked-state")

    # Minority (2 of 5) must fail strict majority!
    with pytest.raises(QuorumInsufficientError):
        coord.propose_and_seal(
            slot=1,
            term=1,
            predecessor_manifest_digest="0" * 64,
            state_manifest_digest=m1_digest_1,
            proposer_id="host-a",
            operation_id="op-init",
            active_member_ids=("auth-0", "auth-1"),
        )

    # 3 of 5 succeeds! Note auth-0 and auth-1 already voted for m1_digest_1 idempotently!
    cert1 = coord.propose_and_seal(
        slot=1,
        term=1,
        predecessor_manifest_digest="0" * 64,
        state_manifest_digest=m1_digest_1,
        proposer_id="host-a",
        operation_id="op-init",
        active_member_ids=("auth-0", "auth-1", "auth-2"),
    )
    assert len(cert1.votes) == 3

    # Cloned credential equivocation on same (slot=1, term=1) must be refused by auth-2!
    forged_prop = QuorumProposal(
        lineage_id=membership.lineage_id,
        membership_digest=membership.membership_digest,
        slot=1,
        term=1,
        predecessor_manifest_digest="0" * 64,
        state_manifest_digest=m1_fork,
        proposer_id="cloned-host-a",
        operation_id="op-fork",
    )
    with pytest.raises(QuorumEquivocationError):
        members[2].vote(forged_prop)

    # Advance to slot 2 on overlapping quorum {auth-2, auth-3, auth-4}
    cert2 = coord.propose_and_seal(
        slot=2,
        term=2,
        predecessor_manifest_digest=m1_digest_1,
        state_manifest_digest=m1_digest_2,
        proposer_id="host-b",
        operation_id="op-step-2",
        active_member_ids=("auth-2", "auth-3", "auth-4"),
    )
    assert cert2.proposal.slot == 2

    # Stale term=1 vote on auth-3 must now be refused!
    stale_prop = QuorumProposal(
        lineage_id=membership.lineage_id,
        membership_digest=membership.membership_digest,
        slot=3,
        term=1,
        predecessor_manifest_digest=m1_digest_2,
        state_manifest_digest=m1_fork,
        proposer_id="zombie-host-a",
        operation_id="op-stale",
    )
    with pytest.raises(StaleTermError):
        members[3].vote(stale_prop)

    # Presenting valid older slot=1 checkpoint is rejected as a rollback!
    with pytest.raises(RollbackDetectedError):
        coord.verify_fresh_checkpoint(slot=1, state_manifest_digest=m1_digest_1)
    assert coord.verify_fresh_checkpoint(slot=2, state_manifest_digest=m1_digest_2) == cert2


@pytest.mark.parametrize("crash_phase", ["after_stage", "after_quorum"])
def test_virtual_processor_hard_kill_os_exit_and_fresh_host_continuity(tmp_path: Path, crash_phase: str) -> None:
    """Hard-kill (`os._exit`) mid-transition followed by fresh-host recovery executes exact-once and fences zombie."""
    cluster_dir = tmp_path / "cluster"
    store_dir = tmp_path / "m1_store"
    session_dir = tmp_path / "session"
    ground_fp = domain_digest("test.ground", "vp-counter-v1")

    # 1. Initialize session on host-1 and run step 1 cleanly
    _, _, _, coord = _build_5node_cluster(cluster_dir)
    store = PagedObjectStore(DiskPageBacking(store_dir, page_size=128))
    sess_host1 = ComputationalSession(
        session_id="sess-crash-01",
        processor_id="vp-alpha",
        host_id="host-1",
        ground_fingerprint=ground_fp,
        store=store,
        coordinator=coord,
        session_dir=session_dir,
        term=1,
    )
    sess_host1.initialize({"counter": 10, "history": ["init"]})
    vp1 = VirtualProcessor(sess_host1)
    vp1.step_json(
        "op-add-5",
        {"delta": 5},
        lambda st, inp: {"counter": st["counter"] + inp["delta"], "history": [*st["history"], "add-5"]},
    )
    assert sess_host1.read_state_json() == {"counter": 15, "history": ["init", "add-5"]}

    # 2. Spawn child process representing host-1 executing "op-mul-3" and suffering hard `os._exit(99)` mid-commit!
    worker_code = f"""
import os
from pathlib import Path
from vld.authority import ComputationalSession, QuorumAuthorityMember, QuorumLineageCoordinator, QuorumMembership, VirtualProcessor
from vld.memory import DiskPageBacking, PagedObjectStore

root = Path({str(tmp_path)!r})
mids = tuple(f"auth-{{i}}" for i in range(5))
membership = QuorumMembership("lineage.vp01", mids)
secrets = {{mid: f"secret-key-for-{{mid}}-0123456789".encode("utf-8") for mid in mids}}
members = [QuorumAuthorityMember(mid, secrets[mid], journal_path=root / "cluster" / "members" / f"{{mid}}.jsonl") for mid in mids]
coord = QuorumLineageCoordinator(membership, members, secrets, ledger_path=root / "cluster" / "coordinator_ledger.jsonl")
store = PagedObjectStore(DiskPageBacking(root / "m1_store", page_size=128))
sess = ComputationalSession(
    session_id="sess-crash-01",
    processor_id="vp-alpha",
    host_id="host-1",
    ground_fingerprint={ground_fp!r},
    store=store,
    coordinator=coord,
    session_dir=root / "session",
    term=1,
)
vp = VirtualProcessor(sess)
hook_stage = (lambda: os._exit(99)) if {crash_phase!r} == "after_stage" else None
hook_quorum = (lambda: os._exit(98)) if {crash_phase!r} == "after_quorum" else None
vp.step_json(
    "op-mul-3",
    {{"factor": 3}},
    lambda st, inp: {{"counter": st["counter"] * inp["factor"], "history": [*st["history"], "mul-3"]}},
    _fault_hook_after_stage=hook_stage,
    _fault_hook_after_quorum=hook_quorum,
)
"""
    proc = subprocess.run([sys.executable, "-c", worker_code], capture_output=True, text=True)
    expected_rc = 99 if crash_phase == "after_stage" else 98
    assert proc.returncode == expected_rc, proc.stderr

    # 3. Recover VirtualProcessor on fresh host-2: must fence term=2 and complete `op-mul-3` EXACTLY ONCE (15 * 3 = 45, NOT 15 and NOT 135!)
    _, _, _, coord_fresh = _build_5node_cluster(cluster_dir)
    store_fresh = PagedObjectStore(DiskPageBacking(store_dir, page_size=128))
    vp2 = VirtualProcessor.recover_on_new_host(
        session_id="sess-crash-01",
        processor_id="vp-alpha",
        new_host_id="host-2",
        ground_fingerprint=ground_fp,
        store=store_fresh,
        coordinator=coord_fresh,
        session_dir=session_dir,
    )
    assert vp2.term == 2
    assert vp2.session.read_state_json() == {"counter": 45, "history": ["init", "add-5", "mul-3"]}

    # Re-submitting "op-mul-3" on host-2 is idempotently recognized as already committed!
    ckpt_replay, replayed = vp2.step_json(
        "op-mul-3",
        {"factor": 3},
        lambda st, inp: {"counter": -999, "history": []},
    )
    assert replayed is True
    assert ckpt_replay.revision == 3
    assert vp2.session.read_state_json()["counter"] == 45

    # Conflicting payload on same operation_id is refused!
    with pytest.raises(SessionConflictError):
        vp2.step_json(
            "op-mul-3",
            {"factor": 999},
            lambda st, inp: st,
        )

    # 4. Zombie host-1 (still at stale term=1) attempting a write MUST be deterministically refused!
    with pytest.raises(StaleEpochError):
        vp1.step_json(
            "op-zombie-write",
            {"delta": 1},
            lambda st, inp: {"counter": 0, "history": ["zombie"]},
        )

    # 5. Corrupting/rolling back `latest.json` on disk to revision 1 is caught by verify_latest_pointer!
    shutil.copyfile(session_dir / "checkpoint_000001.json", session_dir / "latest.json")
    with pytest.raises(RollbackDetectedError):
        vp2.session.verify_latest_pointer()


def test_cross_term_split_brain_duplicate_targets_and_concurrent_multiprocess_recovery(tmp_path: Path) -> None:
    """Verify Paxos Phase 1b/2a cross-term locking, duplicate target rejection, Ground freshness, and 2-process race."""
    from vld.authority import SessionError
    import json

    # 1. Duplicate member IDs in active_member_ids must NEVER spoof strict majority!
    membership, members, secrets, coord_a = _build_5node_cluster(tmp_path / "split")
    coord_b = QuorumLineageCoordinator(
        membership,
        members,
        secrets,
        ledger_path=tmp_path / "split" / "coordinator_ledger_b.jsonl",
    )
    with pytest.raises(QuorumInsufficientError):
        coord_a.fence_term(2, active_member_ids=("auth-0", "auth-0", "auth-0"))

    # 2. Cross-term split-brain refusal: coord_a seals D_A at (slot=1, term=2) on {0,1,2};
    #    coord_b fences term=3 on {2,3,4} and proposes conflicting D_B != D_A -> MUST raise QuorumEquivocationError!
    d_a = domain_digest("test.m1", "manifest-A")
    d_b = domain_digest("test.m1", "manifest-B")
    coord_a.propose_and_seal(
        slot=1,
        term=2,
        predecessor_manifest_digest="0" * 64,
        state_manifest_digest=d_a,
        proposer_id="host-a",
        operation_id="op-a",
        active_member_ids=("auth-0", "auth-1", "auth-2"),
    )
    coord_b.fence_term(3, active_member_ids=("auth-2", "auth-3", "auth-4"), owner_id="host-b")
    with pytest.raises(QuorumEquivocationError, match="Cross-term equivocation"):
        coord_b.propose_and_seal(
            slot=1,
            term=3,
            predecessor_manifest_digest="0" * 64,
            state_manifest_digest=d_b,
            proposer_id="host-b",
            operation_id="op-b",
            active_member_ids=("auth-2", "auth-3", "auth-4"),
        )

    # 3. Repeated pre-quorum recoveries strictly advance term (2 -> 3) & mismatched ground_fingerprint is rejected!
    race_root = tmp_path / "race"
    _, _, _, r_coord = _build_5node_cluster(race_root / "cluster")
    r_store = PagedObjectStore(DiskPageBacking(race_root / "m1", page_size=128))
    g_fp = domain_digest("test.ground", "race-v1")
    sess_init = ComputationalSession(
        session_id="sess-race",
        processor_id="vp-race",
        host_id="host-init",
        ground_fingerprint=g_fp,
        store=r_store,
        coordinator=r_coord,
        session_dir=race_root / "session",
        term=1,
    )
    sess_init.initialize({"val": 1})

    with pytest.raises(SessionError, match="ground_fingerprint"):
        ComputationalSession(
            session_id="sess-race",
            processor_id="vp-race",
            host_id="host-wrong-ground",
            ground_fingerprint=domain_digest("test.ground", "invalidated-v2"),
            store=r_store,
            coordinator=r_coord,
            session_dir=race_root / "session",
            term=1,
        )

    # 4. Concurrent multi-process VirtualProcessor.recover_on_new_host race against shared 5-node cluster!
    race_script = """
import json, sys
from pathlib import Path
from vld.authority import QuorumAuthorityMember, QuorumLineageCoordinator, QuorumMembership, VirtualProcessor
from vld.memory import DiskPageBacking, PagedObjectStore

root = Path(sys.argv[1])
host_id = sys.argv[2]
g_fp = sys.argv[3]
mids = tuple(f"auth-{i}" for i in range(5))
membership = QuorumMembership("lineage.vp01", mids)
secrets = {mid: f"secret-key-for-{mid}-0123456789".encode("utf-8") for mid in mids}
members = [QuorumAuthorityMember(mid, secrets[mid], journal_path=root / "cluster" / "members" / f"{mid}.jsonl") for mid in mids]
coord = QuorumLineageCoordinator(membership, members, secrets, ledger_path=root / "cluster" / "coordinator_ledger.jsonl")
store = PagedObjectStore(DiskPageBacking(root / "m1", page_size=128))
vp = VirtualProcessor.recover_on_new_host(
    session_id="sess-race",
    processor_id="vp-race",
    new_host_id=host_id,
    ground_fingerprint=g_fp,
    store=store,
    coordinator=coord,
    session_dir=root / "session",
)
print(json.dumps({"host_id": host_id, "term": vp.term}))
"""
    p_a = subprocess.Popen(
        [sys.executable, "-c", race_script, str(race_root), "racer-A", g_fp],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p_b = subprocess.Popen(
        [sys.executable, "-c", race_script, str(race_root), "racer-B", g_fp],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out_a, err_a = p_a.communicate(timeout=15)
    out_b, err_b = p_b.communicate(timeout=15)
    assert p_a.returncode == 0, err_a
    assert p_b.returncode == 0, err_b
    res_a = json.loads(out_a.strip())
    res_b = json.loads(out_b.strip())
    # Exclusive epoch ownership guarantees the two racing processes never share the same fencing term!
    assert res_a["term"] != res_b["term"]
    assert min(res_a["term"], res_b["term"]) >= 2

