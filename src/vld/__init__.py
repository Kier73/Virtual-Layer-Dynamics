r"""Virtual Layer Dynamics (`vld`) — Consolidated Computational Ground Architecture.

Phase 0 foundational exports:
  - Canonical serialization & domain-separated SHA-256 (`vld.canonical`)
  - Kleene $\mathcal{K}_3$ guards, epistemic tags, Myhill-Nerode quotient & information triad (`vld.triad`)
  - VirtualGroundSpec, reversible ByteLens, dual-derivation AffineOrbitLaw & Pareto costs (`vld.ground`)
  - Atomic hash-chained receipt journal with idempotency fencing (`vld.journal`)
"""

from __future__ import annotations

from vld.canonical import (
    CanonicalizationError,
    canonical_bytes,
    canonical_json,
    domain_digest,
    parse_strict_json,
)
from vld.ground import (
    AffineOrbitLaw,
    ByteLens,
    DualDerivationMismatchError,
    LensKind,
    LensProjectionError,
    LiftedByteProjection,
    NavigationCostVector,
    OrbitJumpReceipt,
    ProgressionTraceProposal,
    UnderdeterminedGroundError,
    VirtualGroundSpec,
    certify_modular_progression,
    expected_shortcut_cost,
    propose_modular_progression,
)
from vld.journal import (
    GENESIS_PREV_DIGEST,
    HashChainedJournal,
    IdempotencyConflictError,
    JournalIntegrityError,
    JournalReceipt,
)
from vld.triad import (
    ActionInformationSummary,
    EpistemicStatus,
    GroundAnalysisReport,
    IncompleteGroundError,
    IndeterminateGuardTrap,
    TransitionClass,
    TruthDisposition,
    analyze_finite_ground,
    evaluate_interval_guard,
    require_determinate_branch,
)

__version__ = "0.1.0"

__all__ = [
    "ActionInformationSummary",
    "AffineOrbitLaw",
    "ByteLens",
    "CanonicalizationError",
    "DualDerivationMismatchError",
    "EpistemicStatus",
    "GENESIS_PREV_DIGEST",
    "GroundAnalysisReport",
    "HashChainedJournal",
    "IdempotencyConflictError",
    "IncompleteGroundError",
    "IndeterminateGuardTrap",
    "JournalIntegrityError",
    "JournalReceipt",
    "LensKind",
    "LensProjectionError",
    "LiftedByteProjection",
    "NavigationCostVector",
    "OrbitJumpReceipt",
    "ProgressionTraceProposal",
    "TransitionClass",
    "TruthDisposition",
    "UnderdeterminedGroundError",
    "VirtualGroundSpec",
    "__version__",
    "analyze_finite_ground",
    "canonical_bytes",
    "canonical_json",
    "certify_modular_progression",
    "domain_digest",
    "evaluate_interval_guard",
    "expected_shortcut_cost",
    "parse_strict_json",
    "propose_modular_progression",
    "require_determinate_branch",
]
