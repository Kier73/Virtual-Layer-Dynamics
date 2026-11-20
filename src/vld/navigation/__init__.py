r"""Phase 4 — Hierarchical Navigable Computation DAGs, Verification Pre-Gates & Observer-Orbit Composition (`vld.navigation`)."""

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
from vld.navigation.observer_orbit import (
    AttributedObserverReading,
    CertifiedSimilarityDecomposition,
    ComposedComponentSpec,
    ComposedGroundNavigator,
    ComposedGroundProjection,
    ContradictoryObservationError,
    CrossObserverCongruenceGate,
    FiniteFieldOperatorChain,
    ObserverOrbitError,
    discover_similarity_blocks,
)
from vld.navigation.trace_law import (
    DiscoveredTraceLaw,
    TraceLawError,
    TraceLawRefuted,
    TraceLawUnderdetermined,
    discover_trace_law,
)
from vld.navigation.sheaf import (
    CellularSheaf,
    SheafConsistencyError,
    SheafConsistencyReport,
    SheafEdgeSpec,
    SheafObstructionWitness,
)

__all__ = [
    "DiscoveredTraceLaw",
    "TraceLawError",
    "TraceLawRefuted",
    "TraceLawUnderdetermined",
    "discover_trace_law",
    "AttributedObserverReading",
    "BisimulationReport",
    "CellularSheaf",
    "CertifiedSimilarityDecomposition",
    "CommittedNodeResult",
    "ComposedComponentSpec",
    "ComposedGroundNavigator",
    "ComposedGroundProjection",
    "ComputationDAGError",
    "ComputationNode",
    "ContradictoryObservationError",
    "CrossObserverCongruenceGate",
    "FiniteFieldOperatorChain",
    "IntervalBound",
    "IntervalGateReport",
    "NavigableComputationDAG",
    "ObserverOrbitError",
    "SheafConsistencyError",
    "SheafConsistencyReport",
    "SheafEdgeSpec",
    "SheafObstructionWitness",
    "SubgraphThunk",
    "analyze_dag_intervals",
    "check_strong_bisimulation",
    "discover_similarity_blocks",
]
