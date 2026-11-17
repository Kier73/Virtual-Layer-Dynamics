r"""Phase 2 — Active Perception, Distinguishing-Probe Planners & Hypothesis Revision (`vld.inquiry`)."""

from __future__ import annotations

from vld.inquiry.probes import (
    GroundUncertaintyMetrics,
    MinimaxProbePlan,
    PolicyBranch,
    PolicyNode,
    ProbePlanningError,
    ProbeScenario,
    TransitionProbe,
    apply_counterfactual_probe,
    evaluate_ground_uncertainty,
    plan_probes_beam,
    plan_probes_greedy,
    plan_probes_minimax,
)
from vld.inquiry.revision import (
    ActiveGroundInquiry,
    HypothesisDefinition,
    HypothesisLedgerEvent,
    HypothesisRevisionLedger,
    HypothesisSetRevision,
)

__all__ = [
    "ActiveGroundInquiry",
    "GroundUncertaintyMetrics",
    "HypothesisDefinition",
    "HypothesisLedgerEvent",
    "HypothesisRevisionLedger",
    "HypothesisSetRevision",
    "MinimaxProbePlan",
    "PolicyBranch",
    "PolicyNode",
    "ProbePlanningError",
    "ProbeScenario",
    "TransitionProbe",
    "apply_counterfactual_probe",
    "evaluate_ground_uncertainty",
    "plan_probes_beam",
    "plan_probes_greedy",
    "plan_probes_minimax",
]
