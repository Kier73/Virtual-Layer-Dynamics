r"""Phase 7 — Agentic Hypervisors, ProofGround, Capability-Bounded Model Agents & Continuous Layer Builders (`vld.agents`)."""

from __future__ import annotations

from vld.agents.builder import (
    ContinuousLayerBuilderAgent,
    InstalledLayerUpgrade,
    LayerBuildCycleReport,
    create_default_builder_grant,
)
from vld.agents.controller import (
    AgentExplorationReport,
    InLayerModelAgent,
    create_default_exploration_grant,
)
from vld.agents.hypervisor import (
    AgentCapability,
    CASConflictError,
    CapabilityGrant,
    HypervisorRefusalError,
    ReceiverHypervisor,
)
from vld.agents.proof import (
    ProofEnvironment,
    ProofGround,
    ProofGroundAdapter,
    ProofGroundError,
    ProofHyperTransition,
    ProofObligation,
    VerifiedKernelReceipt,
)
from vld.agents.subagents import (
    DirectedSubagentDevelopmentLoop,
    HostedModelSubagent,
    ScoredProposalRecord,
    SubagentCalibrationMemory,
    SubagentEngineeringTask,
    SubagentError,
    SubagentEvolutionReport,
    SubagentProposalOption,
    SubagentRoundOutcome,
    build_compact_hosted_subagent_fixture,
    create_subagent_grant,
)

__all__ = [
    "AgentCapability",
    "AgentExplorationReport",
    "CASConflictError",
    "CapabilityGrant",
    "ContinuousLayerBuilderAgent",
    "DirectedSubagentDevelopmentLoop",
    "HostedModelSubagent",
    "HypervisorRefusalError",
    "InLayerModelAgent",
    "InstalledLayerUpgrade",
    "LayerBuildCycleReport",
    "ProofEnvironment",
    "ProofGround",
    "ProofGroundAdapter",
    "ProofGroundError",
    "ProofHyperTransition",
    "ProofObligation",
    "ReceiverHypervisor",
    "ScoredProposalRecord",
    "SubagentCalibrationMemory",
    "SubagentEngineeringTask",
    "SubagentError",
    "SubagentEvolutionReport",
    "SubagentProposalOption",
    "SubagentRoundOutcome",
    "VerifiedKernelReceipt",
    "build_compact_hosted_subagent_fixture",
    "create_default_builder_grant",
    "create_default_exploration_grant",
    "create_subagent_grant",
]
