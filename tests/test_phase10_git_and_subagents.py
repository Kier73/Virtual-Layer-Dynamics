from __future__ import annotations

from fractions import Fraction
import json
import pytest

from vld.agents import (
    ContinuousLayerBuilderAgent,
    DirectedSubagentDevelopmentLoop,
    ReceiverHypervisor,
    SubagentEngineeringTask,
    SubagentProposalOption,
    build_compact_hosted_subagent_fixture,
    create_subagent_grant,
)
from vld.cli import main as cli_main
from vld.substrates import (
    CountingHTTPOriginServer,
    GitGroundError,
    GitPackGraphGround,
    GitQualificationRefused,
    GitSmartHTTPFixtureServer,
    HTTPPathIntermediaryServer,
    apply_git_delta,
    build_git_pack_v2,
    git_object_oid,
    parse_and_verify_git_pack,
)


def _encode_delta_varint(n: int) -> bytes:
    out = bytearray()
    v = n
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            b |= 0x80
        out.append(b)
        if not v:
            break
    return bytes(out)


def test_git_smart_http_v2_pack_delta_and_bypass_negative_control() -> None:
    # 1. Genuine Protocol v2 server performing `want \ Ancestors(have)` graph-difference computation
    with GitSmartHTTPFixtureServer(mode="delta") as srv:
        ground = GitPackGraphGround(srv.base_url)
        refs = ground.observe_refs()
        assert "refs/heads/main" in refs and "refs/tags/v0.1" in refs
        cert = ground.qualify_graph_delta_operator(
            base_oid=refs["refs/tags/v0.1"],
            target_oid=refs["refs/heads/main"],
        )
        assert cert.full_closure_objects == 7
        assert cert.delta_objects == 3
        assert cert.delta_pack_bytes < cert.full_closure_pack_bytes
        assert cert.byte_reduction_ratio < 0.6

    # 2. Mandatory Negative Control: server ignoring `have` predicates MUST be refused!
    with GitSmartHTTPFixtureServer(mode="ignore_have") as bad_srv:
        bad_ground = GitPackGraphGround(bad_srv.base_url)
        refs_b = bad_ground.observe_refs()
        with pytest.raises(GitQualificationRefused, match="failed graph-difference qualification"):
            bad_ground.qualify_graph_delta_operator(
                base_oid=refs_b["refs/tags/v0.1"],
                target_oid=refs_b["refs/heads/main"],
            )

    # 3. Direct REF_DELTA copy + literal insert verification & trailer tamper detection
    base_blob = b"ABCDEFGHIJKLMNOP" * 8  # 128 bytes
    base_oid = git_object_oid("blob", base_blob)
    # Target = base_blob[0:64] + b"_PATCHED_" (73 bytes)
    delta_stream = (
        _encode_delta_varint(len(base_blob))
        + _encode_delta_varint(73)
        + bytes([0x80 | 0x10, 64])  # COPY offset=0, size=64
        + bytes([9]) + b"_PATCHED_"  # INSERT 9 literal bytes
    )
    assert apply_git_delta(base_blob, delta_stream) == base_blob[:64] + b"_PATCHED_"

    thin_pack = build_git_pack_v2([("ref_delta", base_oid, delta_stream)])
    unpacked = parse_and_verify_git_pack(thin_pack, known_base_objects={base_oid: ("blob", base_blob)})
    exp_target_oid = git_object_oid("blob", base_blob[:64] + b"_PATCHED_")
    assert unpacked[exp_target_oid] == ("blob", base_blob[:64] + b"_PATCHED_")

    # Corrupt 1 bit in PACK trailer -> must fail checksum check!
    tampered_pack = thin_pack[:-1] + bytes([thin_pack[-1] ^ 0xFF])
    with pytest.raises(GitGroundError, match="checksum mismatch"):
        parse_and_verify_git_pack(tampered_pack, known_base_objects={base_oid: ("blob", base_blob)})


def test_directed_hosted_model_subagents_improve_from_counterexamples_and_upgrade_vld() -> None:
    hyp = ReceiverHypervisor("subagent.dev.hypervisor")
    hyp.register_agent(create_subagent_grant("subagent.qwen.d16", allowed_memory_prefix="subagent."))
    hyp.register_agent(create_subagent_grant("subagent.smol.d24", allowed_memory_prefix="subagent."))

    # Two heterogeneous VLD-hosted Transformer subagents (d=16, L=2 vs d=24, L=3)
    sa_alpha = build_compact_hosted_subagent_fixture("subagent.qwen.d16", hyp, hidden_size=16, num_layers=2, seed=11)
    sa_beta = build_compact_hosted_subagent_fixture("subagent.smol.d24", hyp, hidden_size=24, num_layers=3, seed=29)
    builder = ContinuousLayerBuilderAgent("subagent.qwen.d16", hyp)

    # Verify M1 KV prefix cache reuse & orbit-checkpointed note generation
    note = sa_alpha.generate_engineering_note("Inspect recurrence law:", max_new_tokens=5)
    assert len(note["generated_token_ids"]) == 5 and len(note["checkpoint_digest"]) == 64

    with CountingHTTPOriginServer({"weights.bin": b"VLD-WEIGHT-SLICE" * 64}) as origin:
        with (
            HTTPPathIntermediaryServer(origin.base_url, mode="passthrough") as passthrough_proxy,
            HTTPPathIntermediaryServer(origin.base_url, mode="cache") as genuine_cache,
        ):
            # Force initial uncalibrated LLM tie/preference toward short decoy tokenizations at Gen 0
            # by choosing verbalizers where Gen 0 (`U_cal == 0`) picks the decoy, then verify Gen 1 flips to 100% certified!
            probe_task_probe = SubagentEngineeringTask(
                "task.orbit.discovery",
                "structural_memory",
                "Discover certified progression law on modular orbit x_t = (7*t + 5) mod 29.",
                (
                    SubagentProposalOption(
                        "00_decoy_unwrapped_3pt",
                        "propose_3pt_decoy",
                        "A",
                        Fraction(0),
                        Fraction(0),
                        Fraction(1, 10),
                        Fraction(0),
                        "pattern.df0.unwrapped",
                        lambda: hyp.propose_structural_law(
                            "subagent.qwen.d16",
                            memory_id="subagent.decoy",
                            samples=[7, 42, 19],  # 3-sample unwrapped/zero-df -> Hypervisor refuses!
                            domain="Z_mod",
                            declared_modulus=None,
                        ),
                    ),
                    SubagentProposalOption(
                        "01_certified_8pt_shortcut",
                        "install_orbit_shortcut",
                        "A",
                        Fraction(5),
                        Fraction(1),
                        Fraction(99, 100),
                        Fraction(1),
                        "pattern.df5.certified",
                        lambda: {
                            "admitted": True,
                            "upgrade_digest": builder.probe_certify_and_install_orbit_shortcut(
                                "upg.subagent.orbit",
                                memory_id="subagent.orbit_ok",
                                probe_fn=lambda t: (12 + 9 * t) % 29,
                                num_probes=8,
                                domain="Z_mod",
                                declared_modulus=29,
                                target_horizon=10**12,
                            ).artifact_digest,
                        },
                    ),
                ),
            )

            path_task = SubagentEngineeringTask(
                "task.internet.cache_qualify",
                "internet_path",
                "Qualify in-path HTTP conditional/Range host with upstream origin hit control.",
                (
                    SubagentProposalOption(
                        "00_decoy_passthrough_proxy",
                        "qualify_passthrough",
                        "B",
                        Fraction(0),
                        Fraction(0),
                        Fraction(0),
                        Fraction(0),
                        "pattern.nc0.passthrough",
                        lambda: {
                            "admitted": True,
                            "upgrade_digest": builder.qualify_and_install_http_path_cache(
                                "upg.bad.proxy",
                                locus_id="locus.passthrough",
                                intermediary_url=passthrough_proxy.base_url,
                                asset_key="weights.bin",
                                origin_hit_reader=lambda: origin.origin_hits,
                            ).artifact_digest,
                        },
                    ),
                    SubagentProposalOption(
                        "01_genuine_edge_cache",
                        "qualify_edge_cache",
                        "B",
                        Fraction(2),
                        Fraction(1),
                        Fraction(9, 10),
                        Fraction(1),
                        "pattern.nc1.edge_cache",
                        lambda: {
                            "admitted": True,
                            "upgrade_digest": builder.qualify_and_install_http_path_cache(
                                "upg.good.cache",
                                locus_id="locus.edge_cache",
                                intermediary_url=genuine_cache.base_url,
                                asset_key="weights.bin",
                                origin_hit_reader=lambda: origin.origin_hits,
                            ).artifact_digest,
                        },
                    ),
                ),
            )

            loop = DirectedSubagentDevelopmentLoop("vld.coevolution.01", [sa_alpha, sa_beta])
            ev_report = loop.run_development_rounds(
                [
                    [probe_task_probe],             # Round 0: uncalibrated hits decoy -> Hypervisor refuses (0.0)!
                    [probe_task_probe, path_task],  # Round 1: directed & calibrated -> 100% verified upgrades (1.0)!
                ]
            )

            assert ev_report.initial_acceptance_rate == 0.0
            assert ev_report.final_acceptance_rate == 1.0
            assert ev_report.standalone_audit["status"] == "VERIFIED"
            # Cross-model CKA between d=16 and d=24 subagents was computed on every task!
            assert all(o.cross_model_cka is not None for o in ev_report.round_outcomes)
            # Both verified upgrades are now installed in ReceiverHypervisor state!
            installed = hyp.state.get("installed_upgrades", {})
            assert "upg.subagent.orbit" in installed
            assert "upg.good.cache" in installed
            assert "upg.bad.proxy" not in installed

            # Re-scoring an already-seen prompt hits M1 PagedKVCacheGround prefix cache!
            _ = sa_alpha.score_task_options(path_task)
            assert sa_alpha.kv_prefix_reuses >= 1


def test_cli_compile_subagent_and_verify_v070(capsys: pytest.CaptureFixture[str]) -> None:
    rc_info = cli_main(["info"])
    assert rc_info == 0
    info_doc = json.loads(capsys.readouterr().out)
    assert info_doc["version"] == "0.7.0"
    assert any("vld.compiler" in layer for layer in info_doc["layers"])

    rc_comp = cli_main(["compile", "--horizon", "1000000007", "--seed", "42"])
    assert rc_comp == 0
    comp_doc = json.loads(capsys.readouterr().out)
    assert comp_doc["status"] == "INTERCHANGEABLE_VERIFIED"
    assert len(comp_doc["backends_verified"]) == 3

    rc_sa = cli_main(["subagent", "--rounds", "2"])
    assert rc_sa == 0
    sa_doc = json.loads(capsys.readouterr().out)
    assert sa_doc["standalone_audit"]["status"] == "VERIFIED"
    assert len(sa_doc["quorum_public_keys"]) == 3
