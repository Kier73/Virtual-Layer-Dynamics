from __future__ import annotations

from fractions import Fraction
import math
import pytest

from vld.independent_audit import (
    IndependentAuditError,
    verify_precision_decision_bundle_standalone,
    verify_sheaf_consistency_bundle_standalone,
)
from vld.models import (
    CenteredGramRelationalSketch,
    CertifiedPrecisionEscalator,
    ErrorEnclosureCertificate,
    LayerErrorStep,
    PrecisionCertificateError,
    certify_logit_margin_invariance,
    compare_relational_sketches,
)
from vld.navigation import CellularSheaf


def test_cellular_sheaf_global_section_and_minimal_cocycle_localization() -> None:
    # 1. Consistent cyclic 3-observer 2D stalk sheaf over exact Q with rotated frames
    sheaf_ok = CellularSheaf("sheaf.q.consistent", field="Q")
    for obs in ("obs.A", "obs.B", "obs.C"):
        sheaf_ok.add_observer(obs, 2)
    # True underlying states: A=(3, 5), B=(8, -2), C=(4, 7)
    sheaf_ok.add_anchor_observation("anc.A", observer_id="obs.A", projection_map=[[1, 0], [0, 1]], reading=[3, 5])
    # Edge AB: x_B - [[2, 0], [0, -1]] x_A = (8 - 6, -2 - (-5)) = (2, 3)
    sheaf_ok.add_restriction_edge(
        "edge.AB",
        source_observer="obs.A",
        target_observer="obs.B",
        source_map=[[2, 0], [0, -1]],
        target_map=[[1, 0], [0, 1]],
        offset=[2, 3],
    )
    # Edge BC: x_C - [[1, 2], [0, 1]] x_B = (4 - 4, 7 - (-2)) = (0, 9)
    sheaf_ok.add_restriction_edge(
        "edge.BC",
        source_observer="obs.B",
        target_observer="obs.C",
        source_map=[[1, 2], [0, 1]],
        target_map=[[1, 0], [0, 1]],
        offset=[0, 9],
    )
    # Closing cycle CA: x_A - x_C = (-1, -2)
    sheaf_ok.add_restriction_edge(
        "edge.CA",
        source_observer="obs.C",
        target_observer="obs.A",
        source_map=[[1, 0], [0, 1]],
        target_map=[[1, 0], [0, 1]],
        offset=[-1, -2],
    )
    rep_ok = sheaf_ok.solve_global_section()
    assert rep_ok.consistent is True
    assert rep_ok.h0_dim == 0
    assert rep_ok.global_section == {
        "obs.A": ("3", "5"),
        "obs.B": ("8", "-2"),
        "obs.C": ("4", "7"),
    }
    sa_ok = verify_sheaf_consistency_bundle_standalone(rep_ok.to_bundle_dict())
    assert sa_ok["status"] == "VERIFIED" and sa_ok["consistent"] is True

    # 2. Multi-cycle F_97 sheaf: Cycle 1 (v0->v1->v2->v0) is consistent; Cycle 2 (v3->v4->v5->v3) has +17 holonomy drift!
    sheaf_bad = CellularSheaf("sheaf.f97.two_cycles", field="Fp", modulus=97)
    for idx in range(6):
        sheaf_bad.add_observer(f"v{idx}", 1)
    # Consistent cycle 1: offsets +10, +20, -30 (sum = 0 mod 97)
    sheaf_bad.add_restriction_edge("c1.e01", source_observer="v0", target_observer="v1", source_map=[[1]], target_map=[[1]], offset=[10])
    sheaf_bad.add_restriction_edge("c1.e12", source_observer="v1", target_observer="v2", source_map=[[1]], target_map=[[1]], offset=[20])
    sheaf_bad.add_restriction_edge("c1.e20", source_observer="v2", target_observer="v0", source_map=[[1]], target_map=[[1]], offset=[67])
    # Faulty cycle 2: offsets +5, +8, +4 (sum = 17 != 0 mod 97)
    sheaf_bad.add_restriction_edge("c2.e34", source_observer="v3", target_observer="v4", source_map=[[1]], target_map=[[1]], offset=[5])
    sheaf_bad.add_restriction_edge("c2.e45", source_observer="v4", target_observer="v5", source_map=[[1]], target_map=[[1]], offset=[8])
    sheaf_bad.add_restriction_edge("c2.e53", source_observer="v5", target_observer="v3", source_map=[[1]], target_map=[[1]], offset=[4])

    rep_bad = sheaf_bad.solve_global_section()
    assert rep_bad.consistent is False
    assert rep_bad.obstruction is not None
    # Exact localization: ONLY Cycle 2's edges appear in conflicting_edge_ids!
    assert rep_bad.obstruction.conflicting_edge_ids == ("c2.e34", "c2.e45", "c2.e53")

    bundle_bad = rep_bad.to_bundle_dict()
    sa_bad = verify_sheaf_consistency_bundle_standalone(bundle_bad)
    assert sa_bad["status"] == "VERIFIED" and sa_bad["consistent"] is False

    # Forging the dual cocycle must be caught by standalone Fredholm check!
    tampered = dict(bundle_bad)
    bad_obs = dict(bundle_bad["obstruction"])
    bad_obs["dual_cocycle"] = ["1", "0", "0", "1", "1", "1"]
    from vld.independent_audit import independent_domain_digest

    bad_obs["witness_digest"] = independent_domain_digest(
        "vld.sheaf-obstruction-witness.v1",
        {
            "conflicting_edge_ids": bad_obs["conflicting_edge_ids"],
            "dual_cocycle": bad_obs["dual_cocycle"],
            "holonomy_residual": bad_obs["holonomy_residual"],
        },
    )
    tampered["obstruction"] = bad_obs
    tampered["report_digest"] = independent_domain_digest(
        "vld.sheaf-consistency-report.v1",
        {
            "coboundary_rank": tampered["coboundary_rank"],
            "consistent": tampered["consistent"],
            "field": tampered["field"],
            "global_section": tampered["global_section"],
            "h0_dim": tampered["h0_dim"],
            "h1_dim": tampered["h1_dim"],
            "modulus": tampered["modulus"],
            "obstruction": bad_obs,
            "sheaf_id": tampered["sheaf_id"],
        },
    )
    with pytest.raises(IndependentAuditError, match="\\(delta\\^0\\)\\^T y != 0"):
        verify_sheaf_consistency_bundle_standalone(tampered)


def test_proof_carrying_precision_certificate_and_selective_escalation() -> None:
    steps = [
        LayerErrorStep("layer.0.fp4", Fraction("99/100"), Fraction("1/1000"), sinkhorn_col_l1_norm=Fraction("99/100")),
        LayerErrorStep("layer.1.fp8", Fraction("102/100"), Fraction("1/2000")),
    ]
    enc = ErrorEnclosureCertificate.propagate("traj.01", steps, initial_error_radius=Fraction("1/5000"))
    # Under-approximated terminal radius must be rejected
    with pytest.raises(PrecisionCertificateError, match="under-approximates"):
        ErrorEnclosureCertificate("traj.bad", Fraction("1/5000"), tuple(steps), Fraction("1/100000"))

    head_w = [
        [1.0, 0.0, 0.0],
        [0.98, 0.02, 0.0],  # nearly collinear with token 0!
        [-1.0, 0.0, 0.0],
    ]
    # Pairwise Cauchy-Schwarz tightening: ||w0 - w1||_2 ~ 0.0283 << 2*||W||_{2,inf} = 2.0!
    # With eps ~ 0.00172, global bound is ~ 0.00344, while pairwise bound on token 1 is ~ 0.000051!
    approx_collinear = [2.002, 2.000, -2.0]  # gap = 0.002 < global 0.00344, but >> pairwise 0.000051!
    v_tight = certify_logit_margin_invariance(approx_collinear, enc, lm_head_weight=head_w, head_rounding_eps=Fraction(1, 10**6))
    assert v_tight.certified_invariant is True
    assert v_tight.selected_token == 0
    sa_v = verify_precision_decision_bundle_standalone(v_tight.to_bundle_dict())
    assert sa_v["status"] == "VERIFIED" and sa_v["certified_invariant"] is True

    # Selective precision escalator: fast path skips high-precision; near-tie invokes high-precision fallback!
    fallback_calls = 0

    def _high_prec() -> list[float]:
        nonlocal fallback_calls
        fallback_calls += 1
        # Exact arithmetic flips ambiguous near-tie from token 0 to token 2!
        return [1.0001, 0.5, 1.0009]

    escalator = CertifiedPrecisionEscalator(head_max_row_l2=Fraction(1))
    tok_fast, mode_fast, _ = escalator.step_with_certificate(
        low_precision_fn=lambda: ([5.0, 1.0, 0.2], enc),
        high_precision_fallback_fn=_high_prec,
    )
    assert tok_fast == 0 and mode_fast == "CERTIFIED_LOW_PRECISION"
    assert fallback_calls == 0

    tok_esc, mode_esc, verdict_esc = escalator.step_with_certificate(
        low_precision_fn=lambda: ([1.0005, 0.5, 1.0002], enc),
        high_precision_fallback_fn=_high_prec,
    )
    assert mode_esc == "ESCALATED_HIGH_PRECISION"
    assert verdict_esc.certified_invariant is False
    assert tok_esc == 2
    assert fallback_calls == 1
    assert escalator.certified_fast_steps == 1 and escalator.escalated_steps == 1


def test_cross_model_relational_consensus_orthogonal_invariance_and_cone_control() -> None:
    # 1. Exact orthogonal invariance across d_A = 4 vs d_B = 12 (3D rotation embedded in 12D)!
    base_traj = [
        [1.0, 0.2, -0.5, 0.1],
        [0.8, 0.5, -0.2, -0.3],
        [-0.4, 0.9, 0.3, 0.2],
        [-0.7, -0.3, 0.8, -0.1],
        [0.2, -0.8, 0.6, 0.5],
        [0.9, -0.1, -0.4, 0.6],
        [-0.1, 0.7, -0.6, -0.4],
        [0.5, 0.4, 0.5, -0.7],
    ]
    theta = 0.73
    c, s = math.cos(theta), math.sin(theta)
    rotated_12d = [
        [c * r[0] - s * r[1], s * r[0] + c * r[1], c * r[2] - s * r[3], s * r[2] + c * r[3]] + [0.0] * 8
        for r in base_traj
    ]
    sk_a = CenteredGramRelationalSketch.from_hidden_states("model.d4", base_traj)
    sk_b = CenteredGramRelationalSketch.from_hidden_states("model.d12", rotated_12d)
    rep_iso = compare_relational_sketches(sk_a, sk_b, num_permutations=24)
    assert abs(rep_iso.linear_cka - 1.0) < 1e-11
    assert rep_iso.relational_divergence < 1e-5
    assert rep_iso.exceeds_permutation_null is True

    # 2. Narrow positive-cone anisotropy negative control: unrelated trajectories sharing a dominant mean vector!
    cone_m1 = [[20.0 + math.sin(t * 1.7 + j), 0.3 * math.cos(t * 2.3 - j), -0.2 * ((t * j) % 5)] for t in range(8) for j in [1]]
    cone_m2 = [[20.0 + math.cos(t * 3.1), -0.4 * ((t * 3 + 1) % 7), 0.35 * math.sin(t * 5.3), 0.1 * t] for t in range(8)]
    # Fix cone_m1 shape cleanly to 8 tokens x 5 dims
    cone_a = [[25.0, math.sin(t * 1.1), math.cos(t * 2.3), (-1.0) ** t * 0.4, (t % 3) * 0.2] for t in range(8)]
    cone_b = [[25.0, math.cos(t * 4.7), math.sin(t * 3.9), ((t * 2) % 5) * 0.15, -0.3 * (t % 2), 0.1] for t in range(8)]
    del cone_m1, cone_m2
    sk_cone_a = CenteredGramRelationalSketch.from_hidden_states("cone.a", cone_a)
    sk_cone_b = CenteredGramRelationalSketch.from_hidden_states("cone.b", cone_b)
    rep_cone = compare_relational_sketches(sk_cone_a, sk_cone_b, num_permutations=24)
    assert rep_cone.exceeds_permutation_null is False
    assert rep_cone.linear_cka < 0.55

    # 3. Sparse Achlioptas JL sketch across d=256 and d=384 preserves high CKA on shared latent factors!
    latents = [[math.sin(0.6 * t + 0.9 * k) for k in range(6)] for t in range(10)]
    h_256 = [[sum(latents[t][k] * math.cos((k + 1) * (j + 1) * 0.17) for k in range(6)) for j in range(256)] for t in range(10)]
    h_384 = [[sum(latents[t][k] * math.sin((k + 2) * (j + 3) * 0.13) for k in range(6)) for j in range(384)] for t in range(10)]
    jl_256 = CenteredGramRelationalSketch.from_hidden_states("jl.256", h_256, sketch_dim=192, seed_tag="s1")
    jl_384 = CenteredGramRelationalSketch.from_hidden_states("jl.384", h_384, sketch_dim=192, seed_tag="s2")
    rep_jl = compare_relational_sketches(jl_256, jl_384, num_permutations=24)
    assert rep_jl.linear_cka > 0.75
    assert rep_jl.exceeds_permutation_null is True

    # 4. Cross-statement pooled trajectories across variable BPE token lengths (eliminating RoPE position drift!)
    stmt_factors = [[math.cos(0.8 * m + 0.5 * k) for k in range(5)] for m in range(8)]
    # Model A splits statement m into (3 + m % 4) tokens in d=32; Model B splits into (5 + (2*m) % 5) tokens in d=48!
    trajs_a = [
        [
            [sum(stmt_factors[m][k] * math.sin((k + 1) * (j + 1) * 0.21) for k in range(5)) + 2.0 * tok_i * (1 if j == 0 else 0) for j in range(32)]
            for tok_i in range(3 + (m % 4))
        ]
        for m in range(8)
    ]
    trajs_b = [
        [
            [sum(stmt_factors[m][k] * math.sin((k + 1) * (j + 1) * 0.21) for k in range(5)) + 2.0 * tok_i * (1 if j == 1 else 0) for j in range(32)] + [0.0] * 16
            for tok_i in range(5 + ((2 * m) % 5))
        ]
        for m in range(8)
    ]
    pool_a = CenteredGramRelationalSketch.from_pooled_trajectories("pool.a", trajs_a)
    pool_b = CenteredGramRelationalSketch.from_pooled_trajectories("pool.b", trajs_b)
    rep_pool = compare_relational_sketches(pool_a, pool_b, num_permutations=24)
    assert rep_pool.linear_cka > 0.85
    assert rep_pool.exceeds_permutation_null is True
