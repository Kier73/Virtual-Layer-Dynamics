from __future__ import annotations

from copy import deepcopy
from fractions import Fraction

import pytest

from vld.canonical import domain_digest
from vld.independent_trace_audit import verify_trace_navigation_bundle
from vld.independent_trace_inquiry_audit import (
    TraceInquiryAuditError,
    verify_trace_inquiry_bundle,
)
from vld.inquiry import ActiveTraceInquiry, TraceInquiryError, TraceInquiryRefuted


def _oracle(ch: int, coordinate: int, p: int = 101) -> int:
    return 1 if ch == 0 else pow(2, coordinate, p)


def _engage(session: ActiveTraceInquiry, fn=_oracle) -> list[dict]:
    events = []
    for i in range(128):
        probe = session.next_probe()
        if probe is None:
            break
        events.append(session.observe(
            probe, fn(probe.channel, probe.coordinate),
            evidence_ref=f"fixture:observation:{i}:{probe.channel}:{probe.coordinate}",
        ))
    return events


def test_joint_observation_recovers_law_when_each_channel_alone_is_unidentifiable():
    # A=diag(1,2), y0(t)=1, y1(t)=2^t (mod 101).
    # Neither individual geometric trace spans two independent order-2 rows.
    # Their shared law is y(t+2)=3*y(t+1)-2*y(t) mod 101.
    session = ActiveTraceInquiry([[1, 1], [1, 2]], prime=101, order=2, budget=20)
    selected = []
    for _ in range(2):
        probe = session.next_probe()
        assert probe is not None
        assert probe.phase == "identify"
        assert probe.rank_gain == 1
        selected.append(probe.channel)
        session.observe(probe, _oracle(probe.channel, probe.coordinate), evidence_ref=f"oracle:rank:{probe.channel}")
    assert selected == [0, 1]
    assert session.rank == 2
    assert session.coefficients == (3, 99)
    assert session.status == "VALIDATING"
    assert session.next_probe().phase == "validate"

    _engage(session)
    assert session.status == "HELDOUT_SUPPORTED"
    assert session.next_probe() is None
    assert len(session.bundle()["events"]) == 6
    verified = verify_trace_inquiry_bundle(session.bundle())
    assert verified["rank"] == 2
    assert verified["scope"] == "INTERNALLY_AUDITED_UNAUTHENTICATED_OBSERVATIONS"

    witness = {"transition": [[1, 0], [0, 2]], "observers": [[1, 0], [0, 1]],
               "initial_state": [1, 1]}
    far = 10**100 + 17
    paired = session.navigate(far, operator_witness=witness)
    assert paired["navigation"]["values"] == [1, pow(2, far, 101)]
    assert verify_trace_navigation_bundle(paired["navigation"])["status"] == (
        "VERIFIED_CONDITIONAL_ON_DECLARED_OPERATOR"
    )
    assert paired["inquiry"]["bundle_digest"] == session.bundle()["bundle_digest"]


def test_one_observer_cannot_acquire_identifying_rank_even_with_many_samples():
    only_constant = ActiveTraceInquiry([[1, 1]], prime=101, order=2, budget=12)
    events = _engage(only_constant, lambda _ch, _coord: 1)
    assert len(events) == 12
    assert only_constant.rank == 1
    assert only_constant.coefficients is None
    assert only_constant.status == "UNDERDETERMINED"
    assert all(ev["phase"] == "bridge" for ev in events[1:])
    assert verify_trace_inquiry_bundle(only_constant.bundle())["status"] == "UNDERDETERMINED"


def test_cost_aware_selection_is_deterministic_and_does_not_overspend():
    session = ActiveTraceInquiry(
        [[1, 1], [1, 2]], prime=101, order=2,
        observer_costs=[5, Fraction(1, 2)], budget=16,
    )
    first = session.next_probe()
    assert first is not None and first.channel == 1 and first.rank_gain == 1
    session.observe(first, _oracle(first.channel, first.coordinate), evidence_ref="probe:cheap")
    second = session.next_probe()
    assert second is not None and second.channel == 0 and second.rank_gain == 1
    session.observe(second, _oracle(second.channel, second.coordinate), evidence_ref="probe:expensive")
    assert session.coefficients == (3, 99)
    _engage(session)
    assert session.status == "HELDOUT_SUPPORTED"
    assert session.bundle()["remaining_budget"] == [1, 2]  # cost 15.5
    assert verify_trace_inquiry_bundle(session.bundle())["status"] == "HELDOUT_SUPPORTED"


def test_budget_exhaustion_stays_underdetermined_and_refuses_navigation():
    session = ActiveTraceInquiry([[1, 1], [1, 2]], prime=101, order=2, budget=1)
    _engage(session)
    assert session.status == "UNDERDETERMINED"
    assert session.next_probe() is None
    with pytest.raises(TraceInquiryError, match="held-out"):
        session.navigate(1000)
    assert verify_trace_inquiry_bundle(session.bundle())["status"] == "UNDERDETERMINED"


def test_surprise_counterexample_is_preserved_and_prevents_navigation():
    session = ActiveTraceInquiry([[1, 1], [1, 2]], prime=101, order=2)
    for _ in range(2):
        probe = session.next_probe()
        session.observe(probe, _oracle(probe.channel, probe.coordinate), evidence_ref="origin:training")
    assert session.status == "VALIDATING"
    probe = session.next_probe()
    wrong = (_oracle(probe.channel, probe.coordinate) + 1) % 101
    event = session.observe(probe, wrong, evidence_ref="origin:counterexample")
    assert event["refuted"] is True
    assert session.status == "REFUTED"
    assert session.next_probe() is None
    assert session.bundle()["events"][-1]["value"] == wrong
    with pytest.raises(TraceInquiryError):
        session.navigate(2**200)
    assert verify_trace_inquiry_bundle(session.bundle())["status"] == "REFUTED"


def test_stale_probes_bad_values_and_missing_evidence_are_refused():
    session = ActiveTraceInquiry([[1, 1], [1, 2]], prime=101, order=2)
    probe = session.next_probe()
    with pytest.raises(TraceInquiryError, match="attributed"):
        session.observe(probe, 1, evidence_ref="")
    with pytest.raises(TraceInquiryError, match="residue"):
        session.observe(probe, True, evidence_ref="fixture")
    session.observe(probe, 1, evidence_ref="fixture:first")
    with pytest.raises(TraceInquiryError, match="Stale"):
        session.observe(probe, 1, evidence_ref="fixture:replay")


def test_independent_audit_rejects_forged_adaptive_choice_even_with_fresh_hashes():
    session = ActiveTraceInquiry([[1, 1], [1, 2]], prime=101, order=2,
                                  observer_costs=[4, 1])
    _engage(session)
    correct = session.bundle()
    bad = deepcopy(correct)
    # Changing the chosen observer cannot be laundered by recomputing hashes.
    bad["events"][0]["channel"] = 0
    ev_body = {k: v for k, v in bad["events"][0].items() if k != "event_digest"}
    bad["events"][0]["event_digest"] = domain_digest("vld.trace-inquiry-event.v1", ev_body)
    body = {k: v for k, v in bad.items() if k != "bundle_digest"}
    bad["bundle_digest"] = domain_digest("vld.trace-inquiry-bundle.v1", body)
    with pytest.raises(TraceInquiryAuditError, match="selection"):
        verify_trace_inquiry_bundle(bad)

    for field, mutated in [("status", "PROVEN_UNIVERSAL"), ("final_coefficients", [1, 1]),
                           ("remaining_budget", [0, 1])]:
        bad = deepcopy(correct)
        bad[field] = mutated
        bad["bundle_digest"] = domain_digest(
            "vld.trace-inquiry-bundle.v1", {k: v for k, v in bad.items() if k != "bundle_digest"}
        )
        with pytest.raises(TraceInquiryAuditError):
            verify_trace_inquiry_bundle(bad)


@pytest.mark.parametrize("eigenvalues", [(1, 2), (2, 5), (1, 2, 4), (2, 3, 7)])
def test_active_identification_of_distinct_eigenspaces(eigenvalues):
    p = 101
    d = len(eigenvalues)
    traces = [[pow(root, t, p) for t in range(d)] for root in eigenvalues]
    session = ActiveTraceInquiry(traces, prime=p, order=d, budget=64)
    _engage(session, lambda ch, t: pow(eigenvalues[ch], t, p))
    assert session.status == "HELDOUT_SUPPORTED"
    assert session.rank == d
    assert verify_trace_inquiry_bundle(session.bundle())["status"] == "HELDOUT_SUPPORTED"
    far = 10**40 + 59
    projected = session.navigate(far)
    assert projected["navigation"]["values"] == [pow(root, far, p) for root in eigenvalues]
    assert verify_trace_navigation_bundle(projected["navigation"])["status"] == (
        "VERIFIED_OBSERVATIONAL_HYPOTHESIS"
    )
