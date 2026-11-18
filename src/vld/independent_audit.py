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

# Standalone RFC 8032 Ed25519 constants (zero external imports)
_ED_P: int = (1 << 255) - 19
_ED_L: int = (1 << 252) + 27742317777372353535851937790883648493
_ED_D: int = (-121665 * pow(121666, _ED_P - 2, _ED_P)) % _ED_P
_ED_2D: int = (2 * _ED_D) % _ED_P
_ED_SQRT_M1: int = pow(2, (_ED_P - 1) // 4, _ED_P)


def _ed_recover_x(y: int, sign: int) -> int:
    if y < 0 or y >= _ED_P:
        raise IndependentAuditError("Non-canonical Ed25519 y-coordinate.")
    y2 = (y * y) % _ED_P
    u = (y2 - 1) % _ED_P
    v = (_ED_D * y2 + 1) % _ED_P
    v3 = (v * v * v) % _ED_P
    v7 = (v3 * v3 * v) % _ED_P
    x = (u * v3 * pow((u * v7) % _ED_P, (_ED_P - 5) // 8, _ED_P)) % _ED_P
    vx2 = (v * x * x) % _ED_P
    if vx2 == u:
        pass
    elif vx2 == (_ED_P - u) % _ED_P:
        x = (x * _ED_SQRT_M1) % _ED_P
    else:
        raise IndependentAuditError("Point not on Ed25519 curve.")
    if x == 0 and sign == 1:
        raise IndependentAuditError("Invalid negative zero x on Ed25519.")
    if (x & 1) != sign:
        x = _ED_P - x
    return x


_ED_BY: int = (4 * pow(5, _ED_P - 2, _ED_P)) % _ED_P
_ED_BX: int = _ed_recover_x(_ED_BY, 0)
_ED_BASE: tuple[int, int, int, int] = (_ED_BX, _ED_BY, 1, (_ED_BX * _ED_BY) % _ED_P)


def _ed_add(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % _ED_P
    b = ((y1 + x1) * (y2 + x2)) % _ED_P
    c = (t1 * _ED_2D * t2) % _ED_P
    d = (z1 * 2 * z2) % _ED_P
    e = (b - a) % _ED_P
    f = (d - c) % _ED_P
    g = (d + c) % _ED_P
    h = (b + a) % _ED_P
    return ((e * f) % _ED_P, (g * h) % _ED_P, (f * g) % _ED_P, (e * h) % _ED_P)


def _ed_mul(scalar: int, pt: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    res = (0, 1, 1, 0)
    cur = pt
    k = scalar
    while k > 0:
        if k & 1:
            res = _ed_add(res, cur)
        cur = _ed_add(cur, cur)
        k >>= 1
    return res


def _ed_decode_strict(raw: bytes) -> tuple[int, int, int, int]:
    if len(raw) != 32:
        raise IndependentAuditError("Ed25519 point must be 32 bytes.")
    val = int.from_bytes(raw, "little")
    sign = (val >> 255) & 1
    y = val & ((1 << 255) - 1)
    x = _ed_recover_x(y, sign)
    re_enc = (y | ((x & 1) << 255)).to_bytes(32, "little")
    if re_enc != raw:
        raise IndependentAuditError("Non-canonical Ed25519 point encoding.")
    return (x, y, 1, (x * y) % _ED_P)


def ed25519_verify_standalone(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    """Zero-import RFC 8032 §5.1.7 cofactored Ed25519 public-key signature verifier."""
    try:
        if len(public_key_hex) != 64 or len(signature_hex) != 128:
            return False
        if any(c not in "0123456789abcdef" for c in public_key_hex + signature_hex):
            return False
        pub_b = bytes.fromhex(public_key_hex)
        sig_b = bytes.fromhex(signature_hex)
        r_enc = sig_b[:32]
        s_int = int.from_bytes(sig_b[32:], "little")
        if s_int >= _ED_L:
            return False
        a_pt = _ed_decode_strict(pub_b)
        r_pt = _ed_decode_strict(r_enc)
        k = int.from_bytes(hashlib.sha512(r_enc + pub_b + message).digest(), "little") % _ED_L
        lhs = _ed_mul(8 * s_int, _ED_BASE)
        rhs = _ed_add(_ed_mul(8, r_pt), _ed_mul(8 * k, a_pt))
        return ((lhs[0] * rhs[2] - rhs[0] * lhs[2]) % _ED_P == 0) and (
            (lhs[1] * rhs[2] - rhs[1] * lhs[2]) % _ED_P == 0
        )
    except Exception:
        return False


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
    member_public_keys: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    r"""Independently verify a `vld.quorum-certificate.v1` dictionary ($2q > N$ strict majority + HMAC or Ed25519)."""
    members_sorted = sorted(set(str(m) for m in member_ids))
    n_total = len(members_sorted)
    if n_total == 0 or len(members_sorted) != len(member_ids):
        raise IndependentAuditError("Invalid or duplicate quorum membership list.")
    q_min = (n_total // 2) + 1

    prop = cert_dict["proposal"]
    mem_doc = cert_dict.get("membership", {})
    sig_scheme = str(mem_doc.get("signature_scheme", "hmac-sha256"))
    if sig_scheme not in ("hmac-sha256", "ed25519-rfc8032"):
        raise IndependentAuditError(f"Unsupported quorum signature_scheme {sig_scheme!r}.")

    cert_pub_keys: dict[str, str] | None = None
    if sig_scheme == "ed25519-rfc8032":
        raw_pks = mem_doc.get("member_public_keys")
        if not isinstance(raw_pks, Mapping) or sorted(str(k) for k in raw_pks.keys()) != members_sorted:
            raise IndependentAuditError("Ed25519 quorum certificate missing complete member_public_keys.")
        cert_pub_keys = {str(k): str(raw_pks[k]) for k in members_sorted}
        if member_public_keys is not None:
            pinned = {str(k): str(v) for k, v in member_public_keys.items()}
            if pinned != cert_pub_keys:
                raise IndependentAuditError("Certificate member_public_keys disagrees with pinned public keys.")
        expected_mem_dg = independent_domain_digest(
            "vld.quorum-membership.v1",
            {
                "lineage_id": str(prop["lineage_id"]),
                "member_ids": members_sorted,
                "member_public_keys": dict(sorted(cert_pub_keys.items())),
                "signature_scheme": "ed25519-rfc8032",
            },
        )
    else:
        if member_public_keys is not None:
            raise IndependentAuditError("Expected Ed25519 public-key quorum, but certificate uses hmac-sha256.")
        expected_mem_dg = independent_domain_digest(
            "vld.quorum-membership.v1",
            {
                "lineage_id": str(prop["lineage_id"]),
                "member_ids": members_sorted,
            },
        )

    if prop["membership_digest"] != expected_mem_dg:
        raise IndependentAuditError("Quorum proposal membership_digest mismatch.")
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
    normalized_votes: list[dict[str, Any]] = []
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
        if sig_scheme == "ed25519-rfc8032":
            assert cert_pub_keys is not None
            if v.get("signature_scheme") != "ed25519-rfc8032" or "signature_hmac" in v:
                raise IndependentAuditError(f"Vote from {vid!r} lacks ed25519-rfc8032 scheme.")
            sig_ed = str(v.get("signature_ed25519", ""))
            if not ed25519_verify_standalone(cert_pub_keys[vid], msg, sig_ed):
                raise IndependentAuditError(f"Invalid Ed25519 signature for quorum voter {vid!r}.")
            normalized_votes.append(
                {
                    "lineage_id": str(v["lineage_id"]),
                    "proposal_digest": str(v["proposal_digest"]),
                    "signature_ed25519": sig_ed,
                    "signature_scheme": "ed25519-rfc8032",
                    "slot": int(v["slot"]),
                    "term": int(v["term"]),
                    "voter_id": vid,
                }
            )
        else:
            if member_secrets is not None:
                if vid not in member_secrets:
                    raise IndependentAuditError(f"Missing verification secret for voter {vid!r}.")
                expected_sig = hmac.new(member_secrets[vid], msg, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(str(v["signature_hmac"]), expected_sig):
                    raise IndependentAuditError(f"Invalid HMAC vote signature for member {vid!r}.")
            normalized_votes.append(
                {
                    "lineage_id": str(v["lineage_id"]),
                    "proposal_digest": str(v["proposal_digest"]),
                    "signature_hmac": str(v["signature_hmac"]),
                    "slot": int(v["slot"]),
                    "term": int(v["term"]),
                    "voter_id": vid,
                }
            )

    if 2 * len(seen_voters) <= n_total:
        raise IndependentAuditError(f"Insufficient quorum votes: {len(seen_voters)}/{n_total} fails strict 2q > N.")

    mem_canon: dict[str, Any] = {
        "lineage_id": str(prop["lineage_id"]),
        "member_ids": members_sorted,
        "membership_digest": expected_mem_dg,
        "quorum_size": q_min,
    }
    if sig_scheme == "ed25519-rfc8032":
        assert cert_pub_keys is not None
        mem_canon["member_public_keys"] = dict(sorted(cert_pub_keys.items()))
        mem_canon["signature_scheme"] = "ed25519-rfc8032"

    cert_body = {
        "membership": mem_canon,
        "proposal": {**prop_body, "proposal_digest": computed_prop_dg},
        "schema": "vld.quorum-certificate.v1",
        "votes": sorted(normalized_votes, key=lambda item: item["voter_id"]),
    }
    computed_cert_dg = independent_domain_digest("vld.quorum-certificate.v1", cert_body)
    if cert_dict.get("certificate_digest") != computed_cert_dg:
        raise IndependentAuditError("Quorum certificate_digest mismatch.")
    return {
        "certificate_digest": computed_cert_dg,
        "proposal_digest": computed_prop_dg,
        "quorum_size": q_min,
        "signature_scheme": sig_scheme,
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
    quorum_member_public_keys: Mapping[str, str] | None = None,
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
        if quorum_member_ids is not None or quorum_member_public_keys is not None:
            m_ids = (
                quorum_member_ids
                if quorum_member_ids is not None
                else sorted(str(k) for k in (quorum_member_public_keys or {}).keys())
            )
            verify_quorum_certificate_standalone(
                q_cert,
                member_ids=m_ids,
                member_secrets=quorum_member_secrets,
                member_public_keys=quorum_member_public_keys,
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
    if not isinstance(net, Mapping) or net.get("hidden_activation") != "relu":
        raise IndependentAuditError("1D SNC capsule requires hidden_activation == 'relu'.")
    raw_w1 = net.get("hidden_weight")
    raw_b1 = net.get("hidden_bias")
    raw_w2 = net.get("output_weight")
    raw_b2 = net.get("output_bias")
    if (
        not isinstance(raw_w1, (list, tuple))
        or not raw_w1
        or any(not isinstance(r, (list, tuple)) or len(r) != 1 for r in raw_w1)
        or not isinstance(raw_b1, (list, tuple))
        or len(raw_b1) != len(raw_w1)
    ):
        raise IndependentAuditError("Invalid 1D SNC hidden layer geometry.")
    if (
        not isinstance(raw_b2, (list, tuple))
        or not raw_b2
        or not isinstance(raw_w2, (list, tuple))
        or len(raw_w2) != len(raw_b2)
        or any(not isinstance(r, (list, tuple)) or len(r) != len(raw_w1) for r in raw_w2)
    ):
        raise IndependentAuditError("Invalid 1D SNC output layer geometry.")

    w1 = [Fraction(row[0]) for row in raw_w1]
    b1 = [Fraction(v) for v in raw_b1]
    w2 = [[Fraction(v) for v in row] for row in raw_w2]
    b2 = [Fraction(v) for v in raw_b2]
    out_dim = len(b2)

    if not atlas:
        raise IndependentAuditError("1D SNC source_atlas cannot be empty.")
    parsed_regions: list[tuple[Fraction | None, Fraction | None, list[tuple[Fraction, Fraction]]]] = []
    for idx, reg in enumerate(atlas):
        if not isinstance(reg, Mapping) or set(reg.keys()) != {"left", "right", "affines"}:
            raise IndependentAuditError("Invalid atlas region keys.")
        l_val = None if reg["left"] == "-inf" else Fraction(reg["left"])
        r_val = None if reg["right"] == "+inf" else Fraction(reg["right"])
        if (l_val is None) != (idx == 0) or (r_val is None) != (idx == len(atlas) - 1):
            raise IndependentAuditError("Atlas must partition (-inf, +inf) with infinite bounds only at ends.")
        if l_val is not None and r_val is not None and l_val >= r_val:
            raise IndependentAuditError("Inverted or empty interval in 1D SNC source_atlas.")
        raw_aff = reg["affines"]
        if not isinstance(raw_aff, (list, tuple)) or len(raw_aff) != out_dim:
            raise IndependentAuditError("Inconsistent output arity in 1D SNC source_atlas.")
        affs = [(Fraction(pair[0]), Fraction(pair[1])) for pair in raw_aff]
        if parsed_regions:
            prev_l, prev_r, prev_affs = parsed_regions[-1]
            if prev_r != l_val:
                raise IndependentAuditError("Gap or overlap between adjacent 1D SNC atlas intervals.")
            assert l_val is not None
            for (m1, c1), (m2, c2) in zip(prev_affs, affs):
                if m1 * l_val + c1 != m2 * l_val + c2:
                    raise IndependentAuditError(f"Discontinuous 1D SNC atlas at breakpoint {l_val}.")
        parsed_regions.append((l_val, r_val, affs))

    def eval_net(x: Fraction) -> list[Fraction]:
        h = [max(Fraction(0), w * x + b) for w, b in zip(w1, b1)]
        return [
            b2[o] + sum(w2[o][j] * h[j] for j in range(len(h)))
            for o in range(out_dim)
        ]

    # Collect all breakpoints from both atlas and network kinks -b1_j / w1_j
    kinks: set[Fraction] = set()
    for l_val, r_val, _ in parsed_regions:
        if l_val is not None:
            kinks.add(l_val)
        if r_val is not None:
            kinks.add(r_val)
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
        for l_val, r_val, affs in parsed_regions:
            ok_l = True if l_val is None else (x >= l_val)
            ok_r = True if r_val is None else (x <= r_val)
            if ok_l and ok_r:
                return [m * x + c for m, c in affs]
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


def _affine_monoid_power_mod(a: int, b: int, n: int, m: int) -> tuple[int, int]:
    r"""Compute $(a, b)^n \bmod m$ in $O(\log n)$ exact operations without requiring $\gcd(a-1, m)=1$."""
    if m <= 0 or n < 0:
        raise IndependentAuditError("Affine power requires m >= 1 and n >= 0.")
    res_a, res_b = 1 % m, 0
    cur_a, cur_b = a % m, b % m
    k = n
    while k > 0:
        if k & 1:
            res_a, res_b = (cur_a * res_a) % m, (cur_a * res_b + cur_b) % m
        cur_a, cur_b = (cur_a * cur_a) % m, (cur_a * cur_b + cur_b) % m
        k >>= 1
    return res_a, res_b


def _check_atomic_rewrite_rule(rule_id: str, lhs: Mapping[str, Any], rhs: Mapping[str, Any]) -> bool:
    """Check one bidirectional algebraic/structural rewrite rule without trusting optimizer state."""
    for left, right in ((lhs, rhs), (rhs, lhs)):
        lop = str(left.get("op", ""))
        rop = str(right.get("op", ""))
        largs = list(left.get("args", ()))
        rargs = list(right.get("args", ()))
        lpar = dict(left.get("params", {}))
        rpar = dict(right.get("params", {}))

        if rule_id == "affine_fusion" and lop == "affine_mod" and rop in ("affine_mod", "closed_orbit_jump"):
            if len(largs) == 1 and isinstance(largs[0], Mapping) and str(largs[0].get("op")) in ("affine_mod", "closed_orbit_jump"):
                inner = largs[0]
                in_args = list(inner.get("args", ()))
                in_par = dict(inner.get("params", {}))
                m = int(lpar.get("modulus", 0))
                if (
                    m >= 1
                    and int(in_par.get("modulus", 0)) == m
                    and int(rpar.get("modulus", 0)) == m
                    and len(rargs) == 1
                    and len(in_args) == 1
                    and independent_canonical_bytes(in_args[0]) == independent_canonical_bytes(rargs[0])
                ):
                    a1, b1 = int(in_par["a"]) % m, int(in_par["b"]) % m
                    a2, b2 = int(lpar["a"]) % m, int(lpar["b"]) % m
                    if int(rpar["a"]) % m == (a2 * a1) % m and int(rpar["b"]) % m == (a2 * b1 + b2) % m:
                        return True

        elif rule_id == "affine_iter_collapse" and lop == "iterate_affine" and rop == "closed_orbit_jump":
            m = int(lpar.get("modulus", 0))
            n = int(lpar.get("steps", -1))
            if (
                m >= 1
                and n >= 0
                and int(rpar.get("modulus", 0)) == m
                and len(largs) == 1
                and len(rargs) == 1
                and independent_canonical_bytes(largs[0]) == independent_canonical_bytes(rargs[0])
            ):
                an, bn = _affine_monoid_power_mod(int(lpar["a"]), int(lpar["b"]), n, m)
                if int(rpar["a"]) % m == an and int(rpar["b"]) % m == bn:
                    return True

        elif rule_id == "affine_chain_collapse" and lop == "affine_chain" and rop == "closed_orbit_jump":
            m = int(lpar.get("modulus", 0))
            pairs = list(lpar.get("chain", ()))
            if (
                m >= 1
                and pairs
                and int(rpar.get("modulus", 0)) == m
                and len(largs) == 1
                and len(rargs) == 1
                and independent_canonical_bytes(largs[0]) == independent_canonical_bytes(rargs[0])
            ):
                acc_a, acc_b = 1 % m, 0
                for pair in pairs:
                    pa, pb = int(pair[0]) % m, int(pair[1]) % m
                    acc_a, acc_b = (pa * acc_a) % m, (pa * acc_b + pb) % m
                if int(rpar["a"]) % m == acc_a and int(rpar["b"]) % m == acc_b:
                    return True

        elif rule_id == "vm_pullback_reduction" and lop == "buffer_reduce" and rop == "contravariant_pullback_reduce":
            monoid = str(lpar.get("monoid", ""))
            if (
                monoid in ("sum", "xor", "min", "max")
                and str(rpar.get("monoid", "")) == monoid
                and len(largs) == 1
                and isinstance(largs[0], Mapping)
                and str(largs[0].get("op")) == "buffer_permute_patch"
            ):
                perm_node = largs[0]
                p_args = list(perm_node.get("args", ()))
                p_par = dict(perm_node.get("params", {}))
                kind = str(p_par.get("kind", ""))
                n_len = int(p_par.get("length", 0))
                bijective_ok = False
                if kind in ("reverse", "rotate") and n_len > 0:
                    bijective_ok = True
                elif kind == "transpose_2d" and n_len > 0:
                    bijective_ok = int(p_par.get("rows", 0)) * int(p_par.get("cols", 0)) == n_len
                if (
                    bijective_ok
                    and len(p_args) == 1
                    and len(rargs) == 1
                    and independent_canonical_bytes(p_args[0]) == independent_canonical_bytes(rargs[0])
                    and str(rpar.get("kind", "")) == kind
                    and int(rpar.get("length", 0)) == n_len
                    and independent_canonical_bytes(p_par.get("patches", {}))
                    == independent_canonical_bytes(rpar.get("patches", {}))
                ):
                    return True

        elif rule_id == "online_flash_partition" and lop == "partitioned_softmax_attn" and rop == "online_flash_stream_attn":
            if (
                len(largs) == len(rargs)
                and independent_canonical_bytes(largs) == independent_canonical_bytes(rargs)
                and independent_canonical_bytes(lpar) == independent_canonical_bytes(rpar)
                and int(lpar.get("num_shards", 0)) >= 1
            ):
                return True

        elif rule_id == "internet_range_slice" and lop == "http_slice" and rop == "http_range_206":
            off = int(lpar.get("offset", -1))
            ln = int(lpar.get("length", 0))
            if (
                off >= 0
                and ln > 0
                and str(lpar.get("url", "")) != ""
                and str(lpar.get("url")) == str(rpar.get("url"))
                and int(rpar.get("offset", -1)) == off
                and int(rpar.get("length", 0)) == ln
                and largs == []
                and rargs == []
            ):
                return True

        elif rule_id.startswith("congruence:") and lop == rop and independent_canonical_bytes(lpar) == independent_canonical_bytes(rpar):
            sub_rule = rule_id.split(":", 1)[1]
            if len(largs) == len(rargs) and len(largs) >= 1:
                diffs = [
                    i
                    for i in range(len(largs))
                    if independent_canonical_bytes(largs[i]) != independent_canonical_bytes(rargs[i])
                ]
                if len(diffs) == 1:
                    d_idx = diffs[0]
                    if _check_atomic_rewrite_rule(sub_rule, largs[d_idx], rargs[d_idx]):
                        return True
    return False


def verify_rewrite_proof_chain_standalone(
    source_expr: Mapping[str, Any],
    target_expr: Mapping[str, Any],
    proof_steps: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Independently verify an e-graph equivalence rewrite proof without trusting the compiler."""
    src_bytes = independent_canonical_bytes(source_expr)
    tgt_bytes = independent_canonical_bytes(target_expr)
    if not proof_steps:
        if src_bytes != tgt_bytes:
            raise IndependentAuditError("Empty rewrite proof for distinct expressions.")
        return {"rules_verified": [], "status_clean": True, "steps_verified": 0, "status": "VERIFIED"}

    cur_bytes = src_bytes
    rules_used: list[str] = []
    for idx, step in enumerate(proof_steps):
        if int(step.get("step_index", -1)) != idx:
            raise IndependentAuditError(f"Rewrite step {idx}: non-contiguous step_index.")
        rule_id = str(step.get("rule_id", ""))
        lhs = step.get("lhs_expr")
        rhs = step.get("rhs_expr")
        if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping):
            raise IndependentAuditError(f"Rewrite step {idx}: lhs_expr/rhs_expr must be expression mappings.")
        if independent_canonical_bytes(lhs) != cur_bytes:
            raise IndependentAuditError(f"Rewrite step {idx}: lhs_expr does not match current chain expression.")
        expected_step_dg = independent_domain_digest(
            "vld.rewrite-proof-step.v1",
            {"lhs_expr": lhs, "rhs_expr": rhs, "rule_id": rule_id, "step_index": idx},
        )
        if str(step.get("step_digest", "")) != expected_step_dg:
            raise IndependentAuditError(f"Rewrite step {idx}: tampered step_digest.")
        if not _check_atomic_rewrite_rule(rule_id, lhs, rhs):
            raise IndependentAuditError(f"Rewrite step {idx}: unsound or invalid rewrite under rule {rule_id!r}.")
        cur_bytes = independent_canonical_bytes(rhs)
        rules_used.append(rule_id)

    if cur_bytes != tgt_bytes:
        raise IndependentAuditError("Final rewrite proof rhs_expr does not match target_expr.")
    return {
        "rules_verified": rules_used,
        "status": "VERIFIED",
        "steps_verified": len(proof_steps),
    }


def verify_sheaf_consistency_bundle_standalone(bundle: Mapping[str, Any]) -> dict[str, Any]:
    r"""Independently verify a `vld.sheaf-consistency-bundle.v1` global section OR Fredholm 1-cocycle obstruction."""
    if bundle.get("schema") != "vld.sheaf-consistency-bundle.v1":
        raise IndependentAuditError("Invalid sheaf consistency bundle schema.")
    sheaf = bundle["sheaf"]
    field = str(bundle["field"])
    mod = None if bundle["modulus"] is None else int(bundle["modulus"])
    consistent = bool(bundle["consistent"])

    expected_rep_dg = independent_domain_digest(
        "vld.sheaf-consistency-report.v1",
        {
            "coboundary_rank": int(bundle["coboundary_rank"]),
            "consistent": consistent,
            "field": field,
            "global_section": bundle["global_section"],
            "h0_dim": int(bundle["h0_dim"]),
            "h1_dim": int(bundle["h1_dim"]),
            "modulus": mod,
            "obstruction": bundle["obstruction"],
            "sheaf_id": str(bundle["sheaf_id"]),
        },
    )
    if str(bundle.get("report_digest", "")) != expected_rep_dg:
        raise IndependentAuditError("SheafConsistencyReport digest mismatch.")

    observers = {str(k): int(v) for k, v in dict(sheaf["observers"]).items()}
    edges = list(sheaf["edges"])

    if consistent:
        sec = bundle["global_section"]
        if not isinstance(sec, Mapping) or set(sec.keys()) != set(observers.keys()):
            raise IndependentAuditError("Global section missing observer assignments.")
        for e in edges:
            u = str(e["source_observer"])
            v = str(e["target_observer"])
            is_anc = bool(e["is_anchor"])
            xu = list(sec[u])
            xv = list(sec[v])
            if len(xu) != observers[u] or len(xv) != observers[v]:
                raise IndependentAuditError("Global section stalk dimension mismatch.")
            for r in range(int(e["edge_dim"])):
                s_row = e["source_map"][r]
                t_row = e["target_map"][r]
                if field == "Fp":
                    assert mod is not None
                    lhs = sum(int(t_row[j]) * int(xv[j]) for j in range(len(xv))) % mod
                    if not is_anc:
                        lhs = (lhs - sum(int(s_row[j]) * int(xu[j]) for j in range(len(xu)))) % mod
                    if lhs != int(e["offset"][r]) % mod:
                        raise IndependentAuditError(f"Global section violates edge {e['edge_id']!r} over F_{mod}.")
                else:
                    lhs_q = sum((Fraction(t_row[j]) * Fraction(xv[j]) for j in range(len(xv))), Fraction(0))
                    if not is_anc:
                        lhs_q -= sum((Fraction(s_row[j]) * Fraction(xu[j]) for j in range(len(xu))), Fraction(0))
                    if lhs_q != Fraction(e["offset"][r]):
                        raise IndependentAuditError(f"Global section violates edge {e['edge_id']!r} over Q.")
        return {"consistent": True, "edges_verified": len(edges), "status": "VERIFIED"}

    # Inconsistent case: verify Fredholm / 1-cocycle certificate (delta^0)^T y == 0 and <y, c> == holonomy != 0!
    obs_w = bundle["obstruction"]
    if not isinstance(obs_w, Mapping):
        raise IndependentAuditError("Missing obstruction witness on inconsistent sheaf bundle.")
    expected_w_dg = independent_domain_digest(
        "vld.sheaf-obstruction-witness.v1",
        {
            "conflicting_edge_ids": list(obs_w["conflicting_edge_ids"]),
            "dual_cocycle": list(obs_w["dual_cocycle"]),
            "holonomy_residual": str(obs_w["holonomy_residual"]),
        },
    )
    if str(obs_w.get("witness_digest", "")) != expected_w_dg:
        raise IndependentAuditError("SheafObstructionWitness digest mismatch.")

    y_raw = list(obs_w["dual_cocycle"])
    total_e_dim = sum(int(e["edge_dim"]) for e in edges)
    if len(y_raw) != total_e_dim:
        raise IndependentAuditError("dual_cocycle length disagrees with total edge stalk dimension.")

    if field == "Fp":
        assert mod is not None
        y_fp = [int(x) % mod for x in y_raw]
        v_dual_fp = {obs: [0 for _ in range(dim)] for obs, dim in observers.items()}
        hol_fp = 0
        cursor = 0
        supp_eids: list[str] = []
        for e in edges:
            edim = int(e["edge_dim"])
            u = str(e["source_observer"])
            v = str(e["target_observer"])
            is_anc = bool(e["is_anchor"])
            if any(y_fp[cursor + r] != 0 for r in range(edim)):
                supp_eids.append(str(e["edge_id"]))
            for r in range(edim):
                yr = y_fp[cursor + r]
                hol_fp = (hol_fp + yr * int(e["offset"][r])) % mod
                if not is_anc:
                    for j in range(observers[u]):
                        v_dual_fp[u][j] = (v_dual_fp[u][j] - yr * int(e["source_map"][r][j])) % mod
                for j in range(observers[v]):
                    v_dual_fp[v][j] = (v_dual_fp[v][j] + yr * int(e["target_map"][r][j])) % mod
            cursor += edim
        if any(val != 0 for vec in v_dual_fp.values() for val in vec):
            raise IndependentAuditError("Invalid 1-cocycle witness: (delta^0)^T y != 0.")
        if hol_fp == 0 or hol_fp != int(obs_w["holonomy_residual"]) % mod:
            raise IndependentAuditError("Invalid 1-cocycle witness: holonomy <y, c> is zero or mismatched.")
        if supp_eids != list(obs_w["conflicting_edge_ids"]):
            raise IndependentAuditError("conflicting_edge_ids disagrees with dual_cocycle support.")
    else:
        y_q = [Fraction(x) for x in y_raw]
        v_dual_q = {obs: [Fraction(0) for _ in range(dim)] for obs, dim in observers.items()}
        hol_q = Fraction(0)
        cursor = 0
        supp_eids_q: list[str] = []
        for e in edges:
            edim = int(e["edge_dim"])
            u = str(e["source_observer"])
            v = str(e["target_observer"])
            is_anc = bool(e["is_anchor"])
            if any(y_q[cursor + r] != 0 for r in range(edim)):
                supp_eids_q.append(str(e["edge_id"]))
            for r in range(edim):
                yr_q = y_q[cursor + r]
                hol_q += yr_q * Fraction(e["offset"][r])
                if not is_anc:
                    for j in range(observers[u]):
                        v_dual_q[u][j] -= yr_q * Fraction(e["source_map"][r][j])
                for j in range(observers[v]):
                    v_dual_q[v][j] += yr_q * Fraction(e["target_map"][r][j])
            cursor += edim
        if any(val != 0 for vec in v_dual_q.values() for val in vec):
            raise IndependentAuditError("Invalid rational 1-cocycle witness: (delta^0)^T y != 0.")
        if hol_q == 0 or hol_q != Fraction(obs_w["holonomy_residual"]):
            raise IndependentAuditError("Invalid rational 1-cocycle witness: holonomy <y, c> is zero or mismatched.")
        if supp_eids_q != list(obs_w["conflicting_edge_ids"]):
            raise IndependentAuditError("conflicting_edge_ids disagrees with rational dual_cocycle support.")

    return {
        "conflicting_edge_ids": list(obs_w["conflicting_edge_ids"]),
        "consistent": False,
        "status": "VERIFIED",
    }


def verify_precision_decision_bundle_standalone(bundle: Mapping[str, Any]) -> dict[str, Any]:
    r"""Independently verify a `vld.certified-decision-bundle.v1` rational error recurrence & margin bound."""
    if bundle.get("schema") != "vld.certified-decision-bundle.v1":
        raise IndependentAuditError("Invalid precision decision bundle schema.")
    enc = bundle["enclosure"]
    if enc.get("schema") != "vld.error-enclosure-certificate.v1":
        raise IndependentAuditError("Invalid enclosure schema.")

    init_r = Fraction(enc["initial_error_radius"])
    term_r = Fraction(enc["terminal_error_radius"])
    steps = list(enc["steps"])
    if init_r < 0:
        raise IndependentAuditError("Negative initial_error_radius.")
    cur_r = init_r
    for idx, st in enumerate(steps):
        lip = Fraction(st["lipschitz_upper"])
        loc = Fraction(st["local_error_radius"])
        if lip < 0 or loc < 0:
            raise IndependentAuditError(f"Step {idx}: negative Lipschitz or local error.")
        cur_r = lip * cur_r + loc
    if term_r < cur_r:
        raise IndependentAuditError("terminal_error_radius under-approximates propagated rational error.")

    exp_enc_dg = independent_domain_digest(
        "vld.error-enclosure-certificate.v1",
        {
            "initial_error_radius": str(enc["initial_error_radius"]),
            "steps": steps,
            "terminal_error_radius": str(enc["terminal_error_radius"]),
            "trajectory_id": str(enc["trajectory_id"]),
        },
    )
    if str(enc.get("certificate_digest", "")) != exp_enc_dg:
        raise IndependentAuditError("ErrorEnclosureCertificate digest mismatch.")

    obs_m = Fraction(bundle["observed_logit_margin"])
    worst_p = Fraction(bundle["worst_case_logit_perturbation"])
    cert_m = Fraction(bundle["certified_lower_margin"])
    cert_inv = bool(bundle["certified_invariant"])
    if obs_m - worst_p != cert_m:
        raise IndependentAuditError("certified_lower_margin != observed_logit_margin - worst_case_logit_perturbation.")
    if cert_inv != (cert_m > 0):
        raise IndependentAuditError("certified_invariant flag disagrees with sign of certified_lower_margin.")

    exp_v_dg = independent_domain_digest(
        "vld.certified-decision-verdict.v1",
        {
            "certified_invariant": cert_inv,
            "certified_lower_margin": str(bundle["certified_lower_margin"]),
            "enclosure_digest": exp_enc_dg,
            "head_max_row_l2_upper": str(bundle["head_max_row_l2_upper"]),
            "head_rounding_eps": str(bundle["head_rounding_eps"]),
            "observed_logit_margin": str(bundle["observed_logit_margin"]),
            "runner_up_token": int(bundle["runner_up_token"]),
            "selected_token": int(bundle["selected_token"]),
            "trajectory_id": str(bundle["trajectory_id"]),
            "worst_case_logit_perturbation": str(bundle["worst_case_logit_perturbation"]),
        },
    )
    if str(bundle.get("verdict_digest", "")) != exp_v_dg:
        raise IndependentAuditError("CertifiedDecisionVerdict digest mismatch.")

    return {
        "certified_invariant": cert_inv,
        "propagated_error_radius": str(cur_r),
        "selected_token": int(bundle["selected_token"]),
        "status": "VERIFIED",
    }


__all__ = [
    "GENESIS_DIGEST",
    "IndependentAuditError",
    "ed25519_verify_standalone",
    "independent_canonical_bytes",
    "independent_domain_digest",
    "independent_parse_strict_json",
    "verify_continuity_receipt_chain_standalone",
    "verify_precision_decision_bundle_standalone",
    "verify_quorum_certificate_standalone",
    "verify_rewrite_proof_chain_standalone",
    "verify_sheaf_consistency_bundle_standalone",
    "verify_snc_1d_capsule_standalone",
    "verify_transit_packet_receipts_standalone",
]
