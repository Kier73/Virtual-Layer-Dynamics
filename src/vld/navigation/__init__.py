r"""Phase 4 — Hierarchical Navigable Computation DAGs & Verification Pre-Gates (`vld.navigation`)."""

from __future__ import annotations

from vld.navigation.dag import (
    CommittedNodeResult,
    ComputationDAGError,
    ComputationNode,
    NavigableComputationDAG,
    SubgraphThunk,
)
from vld.navigation.gates import (
    BisimulationReport,
    IntervalBound,
    IntervalGateReport,
    analyze_dag_intervals,
    check_strong_bisimulation,
)

__all__ = [
    "BisimulationReport",
    "CommittedNodeResult",
    "ComputationDAGError",
    "ComputationNode",
    "IntervalBound",
    "IntervalGateReport",
    "NavigableComputationDAG",
    "SubgraphThunk",
    "analyze_dag_intervals",
    "check_strong_bisimulation",
]
