"""Phase 2 tests: M1 PagedObjectStore, multi-rung Structural Memory, inverse locators, counterexample demotion, and active inquiry."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from vld.inquiry import (
    ActiveGroundInquiry,
    ProbePlanningError,
    ProbeScenario,
    TransitionProbe,
    plan_probes_beam,
    plan_probes_greedy,
    plan_probes_minimax,
)
from vld.memory import (
    CertificateStatus,
    DiskPageBacking,
    HorizonExceededError,
    InMemoryPageBacking,
    LawCertificationRefused,
    PagedObjectStore,
    PagedStoreError,
    StructuralMemoryRegistry,
    certify_sequence_law,
)
from vld.signal import GroundTransitionSystem, TransitionObservation, TransitionOutcome


def test_m1_paged_store_range_locality_and_tamper_detection(tmp_path: Path) -> None:
    mem_backing = InMemoryPageBacking(page_size=64)
    store = PagedObjectStore(mem_backing)

    # 0-byte edge case
    empty_m = store.put_bytes(b"")
    assert empty_m.size_bytes == 0 and len(empty_m.chunks) == 0
    assert store.read_bytes(empty_m) == b""

    # 640 bytes = 10 pages of 64 bytes; plus unaligned 655 bytes = 11 pages
    payload = bytes((i * 37 + 11) % 256 for i in range(655))
    manifest = store.put_bytes(payload, media_type="application/octet-stream")
    assert len(manifest.chunks) == 11
    mem_backing.pages_read = 0

    # Zero-length range touches 0 pages
    assert store.read_range(manifest, 200, 0) == b""
    assert mem_backing.pages_read == 0

    # Range [130, 250) spans chunk 2 (128..192) and chunk 3 (192..256) -> exactly 2 page reads!
    sub = store.read_range(manifest, 130, 120)
    assert sub == payload[130:250]
    assert mem_backing.pages_read == 2

    # Disk backing persistence + bitrot/tamper refusal
    disk_backing = DiskPageBacking(tmp_path / "m1_pages", page_size=64)
    disk_store = PagedObjectStore(disk_backing)
    disk_manifest = disk_store.put_bytes(payload)
    assert disk_store.read_bytes(disk_manifest) == payload

    # Corrupt last page on disk (flipping a padding byte or payload byte)
    last_digest = disk_manifest.chunks[-1].page_digest
    page_file = tmp_path / "m1_pages" / last_digest[:2] / f"{last_digest}.page"
    corrupted = bytearray(page_file.read_bytes())
    corrupted[-1] ^= 0xFF
    page_file.write_bytes(bytes(corrupted))
    with pytest.raises(PagedStoreError, match="SHA-256"):
        disk_store.read_range(disk_manifest, 640, 15)


def test_five_generative_law_families_and_exact_inverse_lookup() -> None:
    # 1. `int` progression with wrap witness + non-coprime gcd inverse
    wrapped_int = [90, 95, 3, 8, 13]  # step 5 mod 97
    law_int = certify_sequence_law(wrapped_int)
    assert law_int.family == "int"
    target_val = law_int.evaluate(73)
    assert law_int.locate_value(target_val) == 73

    non_coprime = certify_sequence_law([2, 6, 10, 2, 6], declared_modulus=12)
    assert non_coprime.locate_value(10) == 2
    assert non_coprime.locate_value(7) is None  # Odd residue unreachable from 2 + 4t mod 12

    # 2. `lcg` full-period 2^32 recurrence + O(log M) 2-adic inverse lookup!
    a, c, mod, x0 = 1664525, 1013904223, 2**32, 73
    lcg_samples = [x0]
    for _ in range(6):
        lcg_samples.append((a * lcg_samples[-1] + c) % mod)
    law_lcg = certify_sequence_law(lcg_samples, declared_modulus=mod)
    assert law_lcg.family == "lcg"
    distant_t = 731_942_117
    distant_val = law_lcg.evaluate(distant_t)
    assert law_lcg.locate_value(distant_val) == distant_t

    # Prime modulus LCG BSGS inverse
    p_mod = 1009
    p_samples = [17]
    for _ in range(6):
        p_samples.append((31 * p_samples[-1] + 7) % p_mod)
    law_lcg_prime = certify_sequence_law(p_samples, declared_modulus=p_mod)
    val_400 = law_lcg_prime.evaluate(400)
    loc_400 = law_lcg_prime.locate_value(val_400)
    assert loc_400 is not None and law_lcg_prime.evaluate(loc_400) == val_400

    # 3. `gf2` LFSR: s_n = s_{n-7} ^ s_{n-6} over GF(2), order 7 -> needs >= 16 bits
    reg = [1, 0, 1, 1, 0, 0, 1]
    bits = list(reg)
    while len(bits) < 5005:
        nxt = bits[-7] ^ bits[-1]
        bits.append(nxt)
    law_gf2 = certify_sequence_law(bits[:20], prefer_gf2=True)
    assert law_gf2.family == "gf2" and law_gf2.order <= 7
    assert law_gf2.evaluate(5000) == bits[5000]
    assert law_gf2.locate_value(0) == 1
    assert law_gf2.locate_value(1) == 0

    # 4. `poly` Newton binomial lift: f(t) = 3*t^2 - 2*t + 5 over Z
    poly_samples = [3 * t * t - 2 * t + 5 for t in range(7)]
    law_poly = certify_sequence_law(poly_samples, allow_unbounded_poly=True)
    assert law_poly.family == "poly" and law_poly.order == 2
    big_t = 10**12
    assert law_poly.evaluate(big_t) == 3 * big_t * big_t - 2 * big_t + 5

    # 5. `fgraph` eventual periodic orbit: tail [10, 20], cycle [1, 4, 2, 1, 4, 2]
    fgraph_samples = [10, 20, 1, 4, 2, 1, 4, 2]
    law_fg = certify_sequence_law(fgraph_samples)
    assert law_fg.family == "fgraph"
    assert law_fg.evaluate(0) == 10
    assert law_fg.evaluate(2 + 3 * 999_999 + 1) == 4
    assert law_fg.locate_value(2) == 4

    # Underdetermined / unidentifiable traces must be refused
    with pytest.raises(LawCertificationRefused):
        certify_sequence_law([10, 15, 20, 25, 30])  # unwrapped progression without declared modulus
    with pytest.raises(LawCertificationRefused):
        certify_sequence_law([1, 0, 1, 1, 0], prefer_gf2=True)  # too short for held-out GF(2) dof


def test_structural_memory_counterexample_demotion_and_receipt_invalidation() -> None:
    registry = StructuralMemoryRegistry()
    samples = [(5 + 7 * t) % 64 for t in range(6)]
    spec_v1, cert_v1 = registry.certify_and_admit("mem.counter", samples, domain="Z/64Z", declared_modulus=64)
    assert spec_v1.revision == 1 and cert_v1.verdict == "GROUND"

    val_early, rec_early = registry.read("mem.counter", 5)
    val_late, rec_late = registry.read("mem.counter", 500)
    assert val_early == (5 + 7 * 5) % 64
    assert val_late == (5 + 7 * 500) % 64
    assert rec_early.expanded_intermediate_states == 0
    assert registry.verify_read_receipt(rec_early) == (True, "valid")
    assert registry.verify_read_receipt(rec_late) == (True, "valid")

    # Inject counterexample at coordinate 20!
    expected_at_20 = (5 + 7 * 20) % 64
    demotion = registry.inject_counterexample("mem.counter", 20, (expected_at_20 + 1) % 64)
    assert demotion["demoted_to"] == "PREFIX-GROUND"
    assert demotion["invalidated_revision"] == 1
    assert demotion["new_revision"] == 2

    # Receipt at t=500 >= 20 is now deterministically INVALIDATED!
    ok_late, reason_late = registry.verify_read_receipt(rec_late)
    assert not ok_late and "invalidated_by_counterexample_at_20" in reason_late

    # Receipt at t=5 < 20 remains valid on prefix unless active head revision is strictly required
    assert registry.verify_read_receipt(rec_early) == (True, "valid")
    ok_strict, _ = registry.verify_read_receipt(rec_early, require_active_revision=True)
    assert not ok_strict

    # Active revision 2 succeeds for t < 20 and raises HorizonExceededError at t >= 20
    v19, _ = registry.read("mem.counter", 19)
    assert v19 == (5 + 7 * 19) % 64
    with pytest.raises(HorizonExceededError):
        registry.read("mem.counter", 20)

    # Prior audit history in both registry and journal is preserved intact!
    hist = registry.revision_history("mem.counter")
    assert len(hist) == 2
    assert hist[0]["status"] == CertificateStatus.INVALIDATED_BY_COUNTEREXAMPLE.value
    assert hist[1]["status"] == CertificateStatus.ACTIVE.value
    assert len(registry.journal) >= 4
    registry.journal.verify()


def test_active_perception_planners_and_surprise_hypothesis_revision() -> None:
    # Partial Ground: s0, s1; actions a, b. (s0, a)->(s0, [1]) known; (s1, a) unknown; (s0, b) & (s1, b) known equal
    partial = GroundTransitionSystem(
        states=("s0", "s1"),
        actions=("a", "b"),
        initial_state="s0",
        channels=("y",),
        observations=(
            TransitionObservation("s0", "a", (TransitionOutcome("s0", (1,)),)),
            TransitionObservation("s0", "b", (TransitionOutcome("s1", (0,)),)),
            TransitionObservation("s1", "b", (TransitionOutcome("s0", (0,)),)),
        ),
    )
    probes = (
        TransitionProbe(
            state="s1",
            action="a",
            scenarios=(
                ProbeScenario(TransitionOutcome("s1", (2,)), weight=Fraction(3, 4)),
                ProbeScenario(TransitionOutcome("s0", (1,)), weight=Fraction(1, 4)),
            ),
            cost=Fraction(3, 2),
            label="probe_s1_a",
        ),
    )

    # Reject float costs/budgets
    with pytest.raises(ProbePlanningError):
        TransitionProbe("s1", "a", probes[0].scenarios, cost=1.5)  # type: ignore[arg-type]

    p_greedy = plan_probes_greedy(partial, probes, horizon=2, budget=Fraction(2))
    p_beam = plan_probes_beam(partial, probes, horizon=2, beam_width=2, depth=2, budget=Fraction(2))
    plan_mm = plan_probes_minimax(partial, probes, horizon=2, budget=Fraction(2), max_probes=2)

    assert p_greedy is not None and p_greedy.label == "probe_s1_a"
    assert len(p_beam) == 1 and p_beam[0].label == "probe_s1_a"
    assert plan_mm.first_probe is not None and plan_mm.first_probe.label == "probe_s1_a"
    assert plan_mm.root.worst_case_unresolved_pairs == 0
    assert plan_mm.root.worst_case_unresolved_cells == 0

    # ActiveGroundInquiry with a SURPRISE environment outcome (output=(5,) not in declared scenarios!)
    inquiry = ActiveGroundInquiry(partial, policy="minimax", horizon=2)
    summary = inquiry.run_inquiry_loop(
        probes,
        environment_oracle=lambda _s, _a: TransitionOutcome("s1", (5,)),
        budget=Fraction(5),
    )
    assert summary["complete"] is True
    assert summary["unresolved_pairs"] == 0
    assert summary["executed_probes"][0]["surprise"] is True
    assert len(inquiry.ledger.revisions) == 2
    assert len(inquiry.ledger.events) >= 4
