from __future__ import annotations

from fractions import Fraction
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping
import pytest

from vld.authority import (
    Ed25519KeyPair,
    QuorumAuthorityMember,
    QuorumEquivocationError,
    QuorumLineageCoordinator,
    QuorumMembership,
    build_ed25519_quorum_identities,
    ed25519_verify,
    verify_ed25519_hex,
)
from vld.compiler import (
    AuthorityPolicy,
    EffectLedger,
    EffectSemantics,
    IntentExpr,
    IntentionSpec,
    IntentionToRealizationCompiler,
    ObservationContract,
    RewriteProofStep,
    RewriteVerificationError,
    verify_rewrite_proof_chain,
)
from vld.continuous import ContinuousStageSpec, ContinuousVirtualSystem
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.independent_audit import (
    IndependentAuditError,
    ed25519_verify_standalone,
    independent_canonical_bytes,
    verify_continuity_receipt_chain_standalone,
)
from vld.journal import HashChainedJournal
from vld.memory import DiskPageBacking, PagedObjectStore
from vld.substrates.path_grounds import CountingHTTPOriginServer


def test_ed25519_rfc8032_vectors_malleability_and_keystore(tmp_path: Path) -> None:
    # RFC 8032 §7.1 Test Vector 1 (empty) & Test Vector 2 (0x72) & Test Vector 3 (0xaf82)
    vectors = [
        (
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
            "",
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
        ),
        (
            "4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
            "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
            "72",
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
        ),
        (
            "c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
            "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025",
            "af82",
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a",
        ),
    ]
    for idx, (seed_hex, pub_hex, msg_hex, sig_hex) in enumerate(vectors):
        kp = Ed25519KeyPair.from_seed(f"tv-{idx}", bytes.fromhex(seed_hex))
        msg = bytes.fromhex(msg_hex)
        assert kp.public_key_hex == pub_hex
        assert seed_hex not in repr(kp)
        assert kp.sign_hex(msg) == sig_hex
        assert verify_ed25519_hex(pub_hex, msg, sig_hex)
        assert ed25519_verify_standalone(pub_hex, msg, sig_hex)
        # Bit-flip message or signature -> must fail
        assert not verify_ed25519_hex(pub_hex, msg + b"\x00", sig_hex)
        bad_sig = bytearray.fromhex(sig_hex)
        bad_sig[10] ^= 0x01
        assert not ed25519_verify(kp.public_key_bytes, msg, bytes(bad_sig))
        assert not ed25519_verify_standalone(pub_hex, msg, bytes(bad_sig).hex())

        # Malleability attack: S' = S + L >= L must be strictly rejected!
        order_l = (1 << 252) + 27742317777372353535851937790883648493
        s_orig = int.from_bytes(bytes.fromhex(sig_hex)[32:], "little")
        malleable_sig = bytes.fromhex(sig_hex)[:32] + (s_orig + order_l).to_bytes(32, "little")
        assert not ed25519_verify(kp.public_key_bytes, msg, malleable_sig)
        assert not ed25519_verify_standalone(pub_hex, msg, malleable_sig.hex())

    # Private keystore 0600 round-trip
    gen_kp = Ed25519KeyPair.generate("host.alpha.signer")
    ks_path = gen_kp.save_private_keystore(tmp_path / "keys" / "alpha.ed25519")
    reloaded = Ed25519KeyPair.load_private_keystore(ks_path)
    assert reloaded.public_key_hex == gen_kp.public_key_hex
    assert verify_ed25519_hex(reloaded.public_key_hex, b"probe", reloaded.sign_hex(b"probe"))

    if shutil.which("node"):
        spki_hex = gen_kp.to_spki_der().hex()
        sig_h = gen_kp.sign_hex(b"cross-runtime-vld")
        script = (
            "const crypto = require('node:crypto');"
            f"const pub = crypto.createPublicKey({{key: Buffer.from('{spki_hex}', 'hex'), format: 'der', type: 'spki'}});"
            f"const ok = crypto.verify(null, Buffer.from('{b'cross-runtime-vld'.hex()}', 'hex'), pub, Buffer.from('{sig_h}', 'hex'));"
            "console.log(JSON.stringify({ok}));"
        )
        res = json.loads(subprocess.check_output(["node", "-e", script], text=True))
        assert res["ok"] is True


def test_ed25519_public_key_quorum_and_crash_recovery_without_disclosed_secrets(tmp_path: Path) -> None:
    member_ids = ("q0", "q1", "q2")
    keypairs, public_keys = build_ed25519_quorum_identities(member_ids, deterministic_domain_seed=b"test-pk-quorum")
    membership = QuorumMembership("lineage.pk.01", member_ids, member_public_keys=public_keys)
    assert membership.signature_scheme == "ed25519-rfc8032"

    m_journals = {m: tmp_path / f"{m}.journal.jsonl" for m in member_ids}
    members_h1 = [QuorumAuthorityMember.from_ed25519_keypair(m, keypairs[m], journal_path=m_journals[m]) for m in member_ids]
    # Coordinator receives ZERO secrets!
    coord_h1 = QuorumLineageCoordinator(membership, members_h1, ledger_path=tmp_path / "coord1.jsonl")

    store = PagedObjectStore(DiskPageBacking(tmp_path / "m1_pages", page_size=4096))
    shared_journal = HashChainedJournal(tmp_path / "continuous.jsonl")

    executions = {"s0": 0, "s1": 0}
    g0 = VirtualGroundSpec("stage.0", "1.0", "arith", "json", ("add",), ("exact",), ("audit",))
    g1 = VirtualGroundSpec("stage.1", "1.0", "arith", "json", ("mul",), ("exact",), ("audit",))

    def s0_exec(inp: Mapping[str, Any]) -> tuple[dict[str, int], NavigationCostVector, str]:
        executions["s0"] += 1
        return {"v": int(inp["x"]) + 10}, NavigationCostVector(0.1, 1.0, 16.0, 1.0), "PASS:s0"

    def s1_fail_once(_inp: Mapping[str, Any]) -> tuple[dict[str, int], NavigationCostVector, str]:
        executions["s1"] += 1
        raise RuntimeError("simulated_host1_loss_after_s0_quorum_commit")

    def s1_recover(inp: Mapping[str, Any]) -> tuple[dict[str, int], NavigationCostVector, str]:
        executions["s1"] += 1
        return {"v": int(inp["v"]) * 3}, NavigationCostVector(0.1, 1.0, 16.0, 1.0), "PASS:s1"

    stages_h1 = [
        ContinuousStageSpec("s0", g0, "in_schema", "mid_schema", "sched.a", s0_exec, lambda i, o: (o["v"] == i["x"] + 10, "s0_ok")),
        ContinuousStageSpec("s1", g1, "mid_schema", "out_schema", "sched.a", s1_fail_once, lambda i, o: (True, "s1_ok")),
    ]
    sys_h1 = ContinuousVirtualSystem(
        "pk.crash.pipe",
        store=store,
        quorum=coord_h1,
        active_quorum_members=("q0", "q1"),
        journal=shared_journal,
        require_independent_verifier=True,
    )
    with pytest.raises(RuntimeError, match="simulated_host1_loss"):
        sys_h1.execute_stages(stages_h1, {"x": 7}, host_id="host.alpha")

    assert executions == {"s0": 1, "s1": 1}

    # Replacement Host 2 rotates to intersecting quorum ("q1", "q2") with its own fresh coordinator ledger!
    members_h2 = [QuorumAuthorityMember.from_ed25519_keypair(m, keypairs[m], journal_path=m_journals[m]) for m in member_ids]
    coord_h2 = QuorumLineageCoordinator(membership, members_h2, ledger_path=tmp_path / "coord2.jsonl")
    stages_h2 = [
        ContinuousStageSpec("s0", g0, "in_schema", "mid_schema", "sched.b", s0_exec, lambda i, o: (o["v"] == i["x"] + 10, "s0_ok")),
        ContinuousStageSpec("s1", g1, "mid_schema", "out_schema", "sched.b", s1_recover, lambda i, o: (o["v"] == i["v"] * 3, "s1_ok")),
    ]
    sys_h2 = ContinuousVirtualSystem(
        "pk.crash.pipe",
        store=store,
        quorum=coord_h2,
        active_quorum_members=("q1", "q2"),
        journal=shared_journal,
        require_independent_verifier=True,
    )
    res_h2 = sys_h2.execute_stages(stages_h2, {"x": 7}, host_id="host.beta")
    assert res_h2.final_state == {"v": 51}
    assert res_h2.stages_recovered_from_quorum == 1
    assert executions["s0"] == 1  # Stage 0 was NEVER re-executed!

    # Published receipt bundle has ZERO HMAC or private seeds!
    bundle_receipts = [r.to_dict() for r in res_h2.receipts]
    serialized = independent_canonical_bytes(bundle_receipts).decode("utf-8")
    assert "signature_hmac" not in serialized
    assert "signature_ed25519" in serialized

    audit = verify_continuity_receipt_chain_standalone(
        bundle_receipts,
        pipeline_id="pk.crash.pipe",
        initial_input={"x": 7},
        quorum_member_ids=member_ids,
        quorum_member_public_keys=public_keys,
        require_verified_tag=True,
    )
    assert audit["status"] == "VERIFIED"
    assert audit["stages_verified"] == 2

    # Swapping a pinned public key must fail standalone verification!
    rogue_kp = Ed25519KeyPair.generate("rogue")
    tampered_pks = {**public_keys, "q1": rogue_kp.public_key_hex}
    with pytest.raises(IndependentAuditError, match="pinned public keys"):
        verify_continuity_receipt_chain_standalone(
            bundle_receipts,
            pipeline_id="pk.crash.pipe",
            initial_input={"x": 7},
            quorum_member_ids=member_ids,
            quorum_member_public_keys=tampered_pks,
            require_verified_tag=True,
        )

    # Equivocation check under Ed25519
    with pytest.raises(QuorumEquivocationError):
        coord_h2.propose_and_seal(
            slot=2,
            term=2,
            predecessor_manifest_digest=res_h2.receipts[0].quorum_certificate.proposal.state_manifest_digest,
            state_manifest_digest="f" * 64,
            proposer_id="host.beta",
            operation_id="forged_slot2",
            active_member_ids=("q1", "q2"),
        )


def test_intention_compiler_three_realizers_and_adversarial_rewrite_rejection() -> None:
    compiler = IntentionToRealizationCompiler()

    # Non-trivial modulus m=256 with a=5 (gcd(a-1, m) = gcd(4, 256) = 4 > 1!) at horizon 10^12!
    x_var = IntentExpr.var("x", "int_vector")
    expr = IntentExpr.iterate_affine(x_var, a=5, b=13, steps=10**12, modulus=256)
    intention = IntentionSpec(
        intention_id="intent.orbit.1e12",
        expression=expr,
        observation=ObservationContract("int_vector", numerical_tolerance=Fraction(0)),
        authority=AuthorityPolicy("lineage.orbit", ("q0", "q1", "q2"), require_public_key_quorum=True),
    )
    plan = compiler.compile(intention)
    backends = {c.backend_kind for c in plan.candidates}
    assert {"direct_algebraic_navigation", "fluid_in_transit_stream", "hosted_continuous_execution"} <= backends

    report = compiler.verify_interchangeable_realizations(plan, {"x": [3, 19, 104, 251]})
    assert report["status"] == "INTERCHANGEABLE_VERIFIED"
    assert len(report["backends_verified"]) == 3
    # Verify exact algebraic value against small cyclic order mod 256 (5^64 == 1 mod 256, 10^12 is divisible by 64*256=16384 -> identity!)
    assert report["admitted_observation"] == [3, 19, 104, 251]

    # Test non-multiple-of-period horizon 10^12 + 7 vs explicit 7-step check!
    expr_7 = IntentExpr.iterate_affine(x_var, a=5, b=13, steps=10**12 + 7, modulus=256)
    plan_7 = compiler.compile(
        IntentionSpec(
            "intent.orbit.1e12_plus_7",
            expr_7,
            ObservationContract("int_vector"),
            AuthorityPolicy("lineage.orbit7"),
        )
    )
    rep_7 = compiler.verify_interchangeable_realizations(plan_7, {"x": [11, 42]})
    manual = [11, 42]
    for _ in range(7):
        manual = [(5 * v + 13) % 256 for v in manual]
    assert rep_7["admitted_observation"] == manual

    # Adversarial forged rewrite proof MUST be rejected by independent checker!
    bad_target = IntentExpr.closed_orbit_jump(x_var, a=1, b=99, modulus=256)
    forged_step = RewriteProofStep(
        step_index=0,
        rule_id="affine_iter_collapse",
        lhs_expr=expr.to_dict(),
        rhs_expr=bad_target.to_dict(),
    )
    with pytest.raises(RewriteVerificationError, match="unsound or invalid rewrite"):
        verify_rewrite_proof_chain(expr, bad_target, [forged_step])


def test_intention_compiler_vm_pullback_flash_stream_and_http_range_206() -> None:
    compiler = IntentionToRealizationCompiler()

    # 1. Contravariant VM pullback across 2D transpose + sparse CoW patches
    buf_var = IntentExpr.var("buf", "bytes")
    perm = IntentExpr.buffer_permute_patch(
        buf_var, kind="transpose_2d", length=64, rows=8, cols=8, patches={3: 200, 41: 17}
    )
    red_expr = IntentExpr.buffer_reduce(perm, monoid="xor")
    vm_plan = compiler.compile(
        IntentionSpec("intent.vm.xor", red_expr, ObservationContract("int_scalar"), AuthorityPolicy("lin.vm"))
    )
    vm_rep = compiler.verify_interchangeable_realizations(vm_plan, {"buf": bytes((i * 7 + 3) % 256 for i in range(64))})
    assert "direct_algebraic_navigation" in vm_rep["backends_verified"]
    assert vm_rep["status"] == "INTERCHANGEABLE_VERIFIED"

    # 2. Partitioned Softmax Attention -> Online Flash-Stream Monoid
    q_var = IntentExpr.var("Q", "tensor2d")
    sh_var = IntentExpr.var("shards", "kv_shards")
    attn_expr = IntentExpr.partitioned_softmax_attn(q_var, sh_var, num_shards=3, causal=False)
    attn_plan = compiler.compile(
        IntentionSpec(
            "intent.flash.attn",
            attn_expr,
            ObservationContract("tensor2d", numerical_tolerance=Fraction(1, 10**7)),
            AuthorityPolicy("lin.attn"),
        )
    )
    queries = [[0.4, -0.2, 0.7], [-0.5, 0.9, 0.1]]
    shards = [
        {"keys": [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.1]], "values": [[1.0, 2.0], [3.0, -1.0]]},
        {"keys": [[0.7, -0.3, 0.2]], "values": [[-2.0, 4.0]]},
        {"keys": [[-0.2, -0.8, 0.6], [0.3, 0.3, -0.5]], "values": [[0.5, 1.5], [2.5, -0.5]]},
    ]
    attn_rep = compiler.verify_interchangeable_realizations(attn_plan, {"Q": queries, "shards": shards})
    assert attn_rep["max_numerical_discrepancy"] <= 1e-7

    # 3. Qualified Internet HTTP 206 Range operator vs Full-Fetch HTTP Slice
    asset_payload = bytes((i * 13 + 5) % 256 for i in range(4096))
    with CountingHTTPOriginServer({"blob.bin": asset_payload}) as origin:
        # Wrap with Range-supporting HTTPPathIntermediaryServer
        from vld.substrates.path_grounds import HTTPPathIntermediaryServer

        with HTTPPathIntermediaryServer(origin.base_url, mode="cache") as edge_cache:
            target_url = f"{edge_cache.base_url}/blob.bin"
            http_expr = IntentExpr.http_slice(target_url, offset=512, length=32)
            http_plan = compiler.compile(
                IntentionSpec("intent.http.slice", http_expr, ObservationContract("byte_list"), AuthorityPolicy("lin.http"))
            )
            sel = http_plan.get_candidate()
            assert sel.target_expression.op == "http_range_206"
            assert sel.backend_kind == "qualified_internet_path"
            http_rep = compiler.verify_interchangeable_realizations(http_plan, {})
            assert http_rep["admitted_observation"] == list(asset_payload[512:544])
            # Origin was hit only ONCE despite multiple candidate executions because edge_cache retained state!
            assert origin.origin_hits == 1


def test_effect_ledger_idempotency_and_reverse_compensation(tmp_path: Path) -> None:
    j_path = tmp_path / "effects.jsonl"
    ledger = EffectLedger(HashChainedJournal(j_path))

    external_writes: list[str] = []

    def _side_effect() -> dict[str, int]:
        external_writes.append("wrote_invoice_42")
        return {"invoice_id": 42}

    r1 = ledger.execute_effect(
        intention_id="pay.42",
        effect_key="emit_invoice",
        semantics=EffectSemantics.IDEMPOTENT,
        action_fn=_side_effect,
    )
    assert not r1.replayed_from_ledger
    assert external_writes == ["wrote_invoice_42"]

    # Simulate crash & reload from disk journal: retry MUST NOT duplicate external write!
    ledger_reloaded = EffectLedger(HashChainedJournal(j_path))
    r2 = ledger_reloaded.execute_effect(
        intention_id="pay.42",
        effect_key="emit_invoice",
        semantics=EffectSemantics.IDEMPOTENT,
        action_fn=_side_effect,
    )
    assert r2.replayed_from_ledger
    assert r2.result == {"invoice_id": 42}
    assert external_writes == ["wrote_invoice_42"]  # Strictly 1 invocation!

    # Compensatable workflow failing on step 3 must roll back steps 2 then 1 in reverse order!
    events: list[str] = []
    with pytest.raises(RuntimeError, match="step3_downstream_timeout"):
        ledger_reloaded.execute_compensatable_workflow(
            intention_id="deploy.99",
            steps=[
                ("alloc_slot", lambda: (events.append("do_1") or "s1"), lambda v: events.append(f"undo_{v}")),
                ("stage_blob", lambda: (events.append("do_2") or "s2"), lambda v: events.append(f"undo_{v}")),
                ("commit_dns", lambda: (_ for _ in ()).throw(RuntimeError("step3_downstream_timeout")), lambda v: events.append(f"undo_{v}")),
            ],
        )
    assert events == ["do_1", "do_2", "undo_s2", "undo_s1"]
