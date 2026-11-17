r"""Phase 7 — Agentic Hypervisors, ProofGround & Capability-Bounded Model Agents (`vld.agents`)."""

from __future__ import annotations

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

__all__ = [
    "AgentCapability",
    "AgentExplorationReport",
    "CASConflictError",
    "CapabilityGrant",
    "HypervisorRefusalError",
    "InLayerModelAgent",
    "ProofEnvironment",
    "ProofGround",
    "ProofGroundAdapter",
    "ProofGroundError",
    "ProofHyperTransition",
    "ProofObligation",
    "ReceiverHypervisor",
    "VerifiedKernelReceipt",
    "create_default_exploration_grant",
]
