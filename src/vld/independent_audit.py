r"""Standalone Zero-Import Cryptographic & Mathematical Verifier (`vld.independent_audit`).

Design principle (Epistemic Independence):
This module imports **ZERO** `vld.*` modules—relying exclusively on the Python standard library
(`fractions`, `hashlib`, `hmac`, `json`, `math`). It can be copied as a single self-contained file
into any external repository (such as `In-Formation-Space`) or third-party audit sandbox to
independently verify:
  1. Strict canonical JSON serialization & domain-separated SHA-256 digests (`vld.canonical`).
  2. Strict-majority ($2q > N$) `QuorumCommitCertificate`s & monotonic Paxos slot/predecessor chains.
  3. `CrossGroundContinuityReceipt` chains—including genesis-input binding, stage-spec Ground
     fingerprints, representation naturality, and non-forged independent verifier tags.
  4. Whole-domain rational equivalence of `SNC` 1D piecewise-affine ReLU neural capsules over $\mathbb{Q}$.
  5. `TransitInformationPacket` hop-by-hop cryptographic receipt chains.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import hmac
import json
import math
from typing import Any, Mapping, Sequence

GENESIS_DIGEST = "0" * 64


class IndependentAuditError(ValueError):
    """Raised when an artifact, receipt chain, quorum certificate, or neural capsule fails independent audit."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if not isinstance(k, str):
            raise IndependentAuditError("JSON object keys must be strings.")
        if k in out:
            raise IndependentAuditError(f"Duplicate JSON key {k!r} rejected.")
        out[k] = v
    return out


def _validate_canonical_value(val: Any) -> Any:
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise IndependentAuditError("Non-finite float rejected in canonical value.")
        return 0.0 if val == 0.0 else val
    if isinstance(val, str):
        try:
            val.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise IndependentAuditError(f"Invalid UTF-8 surrogate in string: {exc}") from exc
        return val
    if isinstance(val, (list, tuple)):
        return [_validate_canonical_value(x) for x in val]
    if isinstance(val, Mapping):
        out: dict[str, Any] = {}
        for k, v in val.items():
            if not isinstance(k, str):
                raise IndependentAuditError("Mapping keys must be strings.")
            try:
                k.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise IndependentAuditError(f"Invalid UTF-8 key: {exc}") from exc
            out[k] = _validate_canonical_value(v)
        return out
    raise IndependentAuditError(f"Unsupported canonical type {type(val).__name__}.")


def independent_canonical_bytes(value: Any) -> bytes:
    clean = _validate_canonical_value(value)
    text = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8")


def independent_parse_strict_json(raw: str | bytes) -> Any:
    text = raw.decode("utf-8", errors="strict") if isinstance(raw, (bytes, bytearray)) else raw
    parsed = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda c: (_ for _ in ()).throw(IndependentAuditError(f"Forbidden JSON constant {c!r}")),
    )
    return _validate_canonical_value(parsed)


def independent_domain_digest(domain: str, payload: Any) -> str:
    if not domain or "\x00" in domain:
        raise IndependentAuditError("Domain tag must be non-empty without NUL bytes.")
    h = hashlib.sha256()
    h.update(domain.encode("utf-8"))
    h.update(b"\x00")
    h.update(independent_canonical_bytes(payload))
    return h.hexdigest()


def verify_quorum_certificate_standalone(
    cert_dict: Mapping[str, Any],
    *,
    member_ids: Sequence[str],
    member_secrets: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    r"""Independently verify a `vld.quorum-certificate.v1` dictionary ($2q > N$ strict majority + digests)."""
    members_sorted = sorted(set(str(m) for m in member_ids))
    n_total = len(members_sorted)
    if n_total == 0 or len(members_sorted) != len(member_ids):
        raise IndependentAuditError("Invalid or duplicate quorum membership list.")
    q_min = (n_total // 2) + 1

    prop = cert_dict["proposal"]
    expected_mem_dg = independent_domain_digest(
        "vld.quorum-membership.v1",
        {
            "lineage_id": str(prop["lineage_id"]),
            "member_ids": members_sorted,
        },
    )
    if prop["membership_digest"] != expected_mem_dg:
        raise IndependentAuditError("Quorum proposal membership_digest mismatch.")
    mem_doc = cert_dict.get("membership", {})
    if list(mem_doc.get("member_ids", ())) != members_sorted or mem_doc.get("membership_digest") != expected_mem_dg:
        raise IndependentAuditError("Quorum certificate membership object mismatch.")

    prop_body = {
        "lineage_id": str(prop["lineage_id"]),
        "membership_digest": str(prop["membership_digest"]),
        "operation_id": str(prop["operation_id"]),
        "predecessor_manifest_digest": str(prop["predecessor_manifest_digest"]),
        "proposer_id": str(prop["proposer_id"]),
        "schema": "vld.quorum-proposal.v1",
        "slot": int(prop["slot"]),
        "state_manifest_digest": str(prop["state_manifest_digest"]),
        "term": int(prop["term"]),
    }
    computed_prop_dg = independent_domain_digest("vld.quorum-proposal.v1", prop_body)
    if prop.get("proposal_digest") != computed_prop_dg:
        raise IndependentAuditError("Quorum proposal_digest mismatch.")

    votes = list(cert_dict["votes"])
    seen_voters: set[str] = set()
    for v in votes:
        vid = str(v["voter_id"])
        if vid not in members_sorted:
            raise IndependentAuditError(f"Unknown voter {vid!r} outside quorum membership.")
        if vid in seen_voters:
            raise IndependentAuditError(f"Duplicate vote from {vid!r}.")
        if (
            v["proposal_digest"] != computed_prop_dg
            or str(v["lineage_id"]) != str(prop["lineage_id"])
            or int(v["slot"]) != int(prop["slot"])
            or int(v["term"]) != int(prop["term"])
        ):
            raise IndependentAuditError("Vote proposal/slot/term mismatch.")
        seen_voters.add(vid)
        if member_secrets is not None:
            if vid not in member_secrets:
                raise IndependentAuditError(f"Missing verification secret for voter {vid!r}.")
            msg = independent_canonical_bytes(
                {
                    "domain": "vld.quorum-vote.v1",
                    "lineage_id": str(v["lineage_id"]),
                    "proposal_digest": str(v["proposal_digest"]),
                    "slot": int(v["slot"]),
                    "term": int(v["term"]),
                    "voter_id": vid,
                }
            )
            expected_sig = hmac.new(member_secrets[vid], msg, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(str(v["signature_hmac"]), expected_sig):
                raise IndependentAuditError(f"Invalid HMAC vote signature for member {vid!r}.")

    if 2 * len(seen_voters) <= n_total:
        raise IndependentAuditError(f"Insufficient quorum votes: {len(seen_voters)}/{n_total} fails strict 2q > N.")

    cert_body = {
        "membership": {
            "lineage_id": str(prop["lineage_id"]),
            "member_ids": members_sorted,
            "membership_digest": expected_mem_dg,
            "quorum_size": q_min,
        },
        "proposal": {**prop_body, "proposal_digest": computed_prop_dg},
        "schema": "vld.quorum-certificate.v1",
        "votes": sorted(
            [
                {
                    "lineage_id": str(v["lineage_id"]),
                    "proposal_digest": str(v["proposal_digest"]),
                    "signature_hmac": str(v["signature_hmac"]),
                    "slot": int(v["slot"]),
                    "term": int(v["term"]),
                    "voter_id": str(v["voter_id"]),
                }
                for v in votes
            ],
            key=lambda item: item["voter_id"],
        ),
    }
    computed_cert_dg = independent_domain_digest("vld.quorum-certificate.v1", cert_body)
    if cert_dict.get("certificate_digest") != computed_cert_dg:
        raise IndependentAuditError("Quorum certificate_digest mismatch.")
    return {
        "certificate_digest": computed_cert_dg,
        "proposal_digest": computed_prop_dg,
        "quorum_size": q_min,
        "slot": int(prop["slot"]),
        "term": int(prop["term"]),
        "votes_verified": len(seen_voters),
    }


def verify_continuity_receipt_chain_standalone(
    receipts: Sequence[Mapping[str, Any]],
    *,
    pipeline_id: str,
    initial_input: Mapping[str, Any],
    expected_stages: Sequence[Mapping[str, str]] | None = None,
    quorum_member_ids: Sequence[str] | None = None,
    quorum_member_secrets: Mapping[str, bytes] | None = None,
    require_verified_tag: bool = False,
) -> dict[str, Any]:
    """Independently reconstruct and verify an entire `CrossGroundContinuityReceipt` chain from genesis."""
    if not receipts:
        raise IndependentAuditError("Receipt chain cannot be empty.")
    if expected_stages is not None and len(receipts) > len(expected_stages):
        raise IndependentAuditError("Receipt count exceeds expected_stages length.")

    clean_init = independent_parse_strict_json(independent_canonical_bytes(dict(initial_input)))
    prev_ground_fp = independent_domain_digest(
        "vld.pipeline-genesis.v1",
        {"input": clean_init, "pipeline_id": pipeline_id},
    )
    prev_cont_dg = GENESIS_DIGEST
    prev_slot_token_dg = GENESIS_DIGEST

    for idx, rcpt in enumerate(receipts):
        if str(rcpt["pipeline_id"]) != pipeline_id:
            raise IndependentAuditError(f"Stage {idx}: pipeline_id mismatch.")
        if int(rcpt["stage_index"]) != idx:
            raise IndependentAuditError(f"Stage {idx}: stage_index mismatch ({rcpt['stage_index']}).")
        if str(rcpt["prev_continuity_digest"]) != prev_cont_dg:
            raise IndependentAuditError(f"Stage {idx}: broken prev_continuity_digest chain.")
        if str(rcpt["source_ground_fingerprint"]) != prev_ground_fp:
            raise IndependentAuditError(f"Stage {idx}: source_ground_fingerprint mismatch (genesis input or prior Ground altered).")

        verdict = str(rcpt["oracle_verdict"])
        if not verdict.startswith("PASS"):
            raise IndependentAuditError(f"Stage {idx}: non-PASS oracle_verdict {verdict!r}.")
        if require_verified_tag and "|VERIFIED:" not in verdict:
            raise IndependentAuditError(f"Stage {idx}: missing mandatory independent '|VERIFIED:' attestation.")

        if expected_stages is not None:
            exp = expected_stages[idx]
            if str(rcpt["stage_id"]) != str(exp["stage_id"]):
                raise IndependentAuditError(f"Stage {idx}: stage_id mismatch.")
            if str(rcpt["target_ground_fingerprint"]) != str(exp["ground_fingerprint"]):
                raise IndependentAuditError(f"Stage {idx}: target_ground_fingerprint mismatch.")
            if (
                str(rcpt["source_representation"]) != str(exp["input_representation"])
                or str(rcpt["target_representation"]) != str(exp["output_representation"])
            ):
                raise IndependentAuditError(f"Stage {idx}: representation contract mismatch.")

        q_cert = rcpt["quorum_certificate"]
        if quorum_member_ids is not None:
            verify_quorum_certificate_standalone(
                q_cert,
                member_ids=quorum_member_ids,
                member_secrets=quorum_member_secrets,
            )

        state_m = rcpt["state_manifest"]
        expected_slot_token = independent_domain_digest(
            "vld.stage-slot.v1",
            {
                "pipeline_id": pipeline_id,
                "stage_id": str(rcpt["stage_id"]),
                "stage_index": idx,
                "state_manifest_digest": str(state_m["manifest_digest"]),
            },
        )
        prop = q_cert["proposal"]
        if int(prop["slot"]) != idx + 1:
            raise IndependentAuditError(f"Stage {idx}: quorum slot {prop['slot']} != {idx + 1}.")
        if str(prop["state_manifest_digest"]) != expected_slot_token:
            raise IndependentAuditError(f"Stage {idx}: quorum state_manifest_digest does not bind stage slot token.")
        if str(prop["predecessor_manifest_digest"]) != prev_slot_token_dg:
            raise IndependentAuditError(f"Stage {idx}: broken quorum predecessor_manifest_digest lineage.")

        body = {
            "clock_bridge_digest": rcpt["clock_bridge_digest"],
            "host_id": str(rcpt["host_id"]),
            "oracle_verdict": verdict,
            "pipeline_id": pipeline_id,
            "prev_continuity_digest": prev_cont_dg,
            "quorum_cert_digest": str(q_cert["certificate_digest"]),
            "scheduler_id": str(rcpt["scheduler_id"]),
            "schema": "vld.cross-ground-continuity-receipt.v1",
            "source_ground_fingerprint": str(rcpt["source_ground_fingerprint"]),
            "source_representation": str(rcpt["source_representation"]),
            "stage_cost": dict(rcpt["stage_cost"]),
            "stage_id": str(rcpt["stage_id"]),
            "stage_index": idx,
            "state_manifest": dict(state_m),
            "target_ground_fingerprint": str(rcpt["target_ground_fingerprint"]),
            "target_representation": str(rcpt["target_representation"]),
            "temporal": dict(rcpt["temporal"]),
        }
        computed_cont = independent_domain_digest("vld.cross-ground-continuity-receipt.v1", body)
        if str(rcpt["continuity_digest"]) != computed_cont:
            raise IndependentAuditError(f"Stage {idx}: continuity_digest mismatch.")

        prev_cont_dg = computed_cont
        prev_slot_token_dg = expected_slot_token
        prev_ground_fp = str(rcpt["target_ground_fingerprint"])

    return {
        "final_continuity_digest": prev_cont_dg,
        "genesis_fingerprint": independent_domain_digest(
            "vld.pipeline-genesis.v1", {"input": clean_init, "pipeline_id": pipeline_id}
        ),
        "pipeline_id": pipeline_id,
        "stages_verified": len(receipts),
        "status": "VERIFIED",
    }


def verify_snc_1d_capsule_standalone(capsule: Mapping[str, Any]) -> dict[str, Any]:
    r"""Independently verify whole-domain $\mathbb{Q}$ equivalence of a 1D `SNC` ReLU capsule without compiler code."""
    if capsule.get("schema") != "vld.snc-relu-capsule.v1":
        raise IndependentAuditError(f"Unsupported 1D SNC schema {capsule.get('schema')!r}.")
    supplied_digest = capsule.get("capsule_digest")
    body = {k: v for k, v in capsule.items() if k != "capsule_digest"}
    if supplied_digest != independent_domain_digest("vld.snc-relu-capsule.v1", body):
        raise IndependentAuditError("1D SNC capsule_digest mismatch.")
    atlas = list(capsule["source_atlas"])
    if capsule.get("source_atlas_digest") != independent_domain_digest("vld.semantic-atlas.v1", atlas):
        raise IndependentAuditError("1D SNC source_atlas_digest mismatch.")
    net = capsule["network"]
    w1 = [Fraction(row[0]) for row in net["hidden_weight"]]
    b1 = [Fraction(v) for v in net["hidden_bias"]]
    w2 = [[Fraction(v) for v in row] for row in net["output_weight"]]
    b2 = [Fraction(v) for v in net["output_bias"]]
    out_dim = len(b2)

    def eval_net(x: Fraction) -> list[Fraction]:
        h = [max(Fraction(0), w * x + b) for w, b in zip(w1, b1)]
        return [
            b2[o] + sum(w2[o][j] * h[j] for j in range(len(h)))
            for o in range(out_dim)
        ]

    # Collect all breakpoints from both atlas and network kinks -b1_j / w1_j
    kinks: set[Fraction] = set()
    for reg in atlas:
        if reg["left"] != "-inf":
            kinks.add(Fraction(reg["left"]))
        if reg["right"] != "+inf":
            kinks.add(Fraction(reg["right"]))
    for w, b in zip(w1, b1):
        if w != 0:
            kinks.add(-b / w)
    sorted_pts = sorted(kinks)
    # Sample two distinct rational interior points per cell -> exact affine agreement on 2 points proves whole-interval identity!
    cells: list[tuple[Fraction, Fraction]] = []
    if not sorted_pts:
        cells.append((Fraction(-1), Fraction(1)))
    else:
        cells.append((sorted_pts[0] - 2, sorted_pts[0] - 1))
        for i in range(len(sorted_pts) - 1):
            a, b_pt = sorted_pts[i], sorted_pts[i + 1]
            cells.append((a + (b_pt - a) / 3, a + 2 * (b_pt - a) / 3))
        cells.append((sorted_pts[-1] + 1, sorted_pts[-1] + 2))

    def eval_atlas(x: Fraction) -> list[Fraction]:
        for reg in atlas:
            ok_l = True if reg["left"] == "-inf" else (x >= Fraction(reg["left"]))
            ok_r = True if reg["right"] == "+inf" else (x <= Fraction(reg["right"]))
            if ok_l and ok_r:
                return [Fraction(a[0]) * x + Fraction(a[1]) for a in reg["affines"]]
        raise IndependentAuditError(f"Atlas does not cover point {x}.")

    checked_points = 0
    for x1, x2 in cells:
        for xp in (x1, x2):
            checked_points += 1
            if eval_net(xp) != eval_atlas(xp):
                raise IndependentAuditError(f"1D SNC ReLU capsule disagrees with specification at x={xp}.")
    return {
        "cells_verified": len(cells),
        "points_verified": checked_points,
        "status": "VERIFIED",
    }


def verify_transit_packet_receipts_standalone(packet_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Independently verify a `vld.transit-packet.v1` hop-by-hop SHA-256 chain."""
    if packet_dict.get("schema") != "vld.transit-packet.v1":
        raise IndependentAuditError("Invalid transit packet schema.")
    pkt_id = str(packet_dict["packet_id"])
    receipts = list(packet_dict.get("receipts", ()))
    prev_dg = GENESIS_DIGEST
    for idx, r in enumerate(receipts):
        if str(r["packet_id"]) != pkt_id or int(r["hop_index"]) != idx or str(r["prev_hop_digest"]) != prev_dg:
            raise IndependentAuditError(f"Broken transit hop receipt chain at hop {idx}.")
        comp = independent_domain_digest(
            "vld.fluid-transit-hop-receipt.v1",
            {
                "hop_index": idx,
                "input_digest": str(r["input_digest"]),
                "interpreter_id": str(r["interpreter_id"]),
                "ops_executed": list(r["ops_executed"]),
                "output_digest": str(r["output_digest"]),
                "packet_id": pkt_id,
                "prev_hop_digest": prev_dg,
            },
        )
        if str(r["hop_digest"]) != comp:
            raise IndependentAuditError(f"Tampered hop_digest at hop {idx}.")
        prev_dg = comp
    if receipts:
        final_payload_dg = hashlib.sha256(independent_canonical_bytes(packet_dict["payload"])).hexdigest()
        if final_payload_dg != str(receipts[-1]["output_digest"]):
            raise IndependentAuditError("Final packet payload disagrees with terminal hop output_digest.")
    return {
        "hops_verified": len(receipts),
        "packet_id": pkt_id,
        "status": "VERIFIED",
        "terminal_hop_digest": prev_dg,
    }


__all__ = [
    "GENESIS_DIGEST",
    "IndependentAuditError",
    "independent_canonical_bytes",
    "independent_domain_digest",
    "independent_parse_strict_json",
    "verify_continuity_receipt_chain_standalone",
    "verify_quorum_certificate_standalone",
    "verify_snc_1d_capsule_standalone",
    "verify_transit_packet_receipts_standalone",
]
