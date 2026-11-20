"""TRACE-001 reproducible demonstration: two observer traces become a navigable Ground."""

from __future__ import annotations

import json
from time import perf_counter

from vld.independent_trace_audit import verify_trace_navigation_bundle
from vld.navigation import TraceLawUnderdetermined, discover_trace_law


def main() -> None:
    p = 65521
    state = [1, 0]
    traces: list[list[int]] = [[], []]
    operator = [[1, 1], [1, 0]]
    for _ in range(13):
        traces[0].append(state[0])
        traces[1].append(state[1])
        state = [(state[0] + state[1]) % p, state[0]]
    witness = {"transition": operator, "observers": [[1, 0], [0, 1]], "initial_state": [1, 0]}

    single_observer_underdetermined = False
    try:
        discover_trace_law([traces[0]], prime=p, training_length=3)
    except TraceLawUnderdetermined:
        single_observer_underdetermined = True

    law = discover_trace_law(traces, prime=p, training_length=3)
    target = 10**60 + 17
    start = perf_counter()
    receipt = law.navigate(target, operator_witness=witness)
    audit = verify_trace_navigation_bundle(receipt)
    duration_ms = (perf_counter() - start) * 1000

    assert single_observer_underdetermined
    assert audit["values"] == [17164, 51927]
    print(json.dumps({
        "experiment": "TRACE-001",
        "single_observer_underdetermined": single_observer_underdetermined,
        "joint_coefficients": list(law.coefficients),
        "train_frames": law.training_length,
        "heldout_frames_per_channel": len(traces[0]) - law.training_length,
        "target": target,
        "projected_values": audit["values"],
        "independent_audit_status": audit["status"],
        "elapsed_ms_combined": round(duration_ms, 3),
        "law_digest": law.law_digest,
        "receipt_digest": receipt["claim_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
