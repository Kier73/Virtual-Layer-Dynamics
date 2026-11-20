"""TRACE-002 — active observation rather than passive full-trace acquisition."""

from __future__ import annotations

import json
from vld.inquiry import ActiveTraceInquiry
from vld.independent_trace_inquiry_audit import verify_trace_inquiry_bundle
from vld.independent_trace_audit import verify_trace_navigation_bundle


def main() -> None:
    # The hidden operator in this demo is diagonal. Each observer independently
    # sees one geometric sequence; neither has rank 2 by itself.
    p = 101
    ground = ActiveTraceInquiry([[1, 1], [1, 2]], prime=p, order=2, budget=20)
    acquired = []
    while (probe := ground.next_probe()) is not None:
        value = 1 if probe.channel == 0 else pow(2, probe.coordinate, p)
        ground.observe(
            probe,
            value,
            evidence_ref=f"local-fixture:channel-{probe.channel}:t{probe.coordinate}",
        )
        acquired.append({
            "channel": probe.channel, "time": probe.coordinate,
            "phase": probe.phase, "guaranteed_rank_gain": probe.rank_gain,
        })

    inquiry_receipt = ground.bundle()
    inquiry_audit = verify_trace_inquiry_bundle(inquiry_receipt)
    t = 10**100 + 17
    witness = {
        "transition": [[1, 0], [0, 2]],
        "observers": [[1, 0], [0, 1]],
        "initial_state": [1, 1],
    }
    result = ground.navigate(t, operator_witness=witness)
    nav_audit = verify_trace_navigation_bundle(result["navigation"])
    print(json.dumps({
        "experiment": "TRACE-002",
        "observer_status": ground.status,
        "rank": ground.rank,
        "law_coefficients": list(ground.coefficients or ()),
        "acquisitions": acquired,
        "query_count": len(acquired),
        "inquiry_audit": inquiry_audit["status"],
        "inquiry_scope": inquiry_audit["scope"],
        "coordinate": t,
        "projected_values": result["navigation"]["values"],
        "operator_audit": nav_audit["status"],
        "transcript_digest": inquiry_receipt["bundle_digest"],
    }, indent=2))


if __name__ == "__main__":
    main()
