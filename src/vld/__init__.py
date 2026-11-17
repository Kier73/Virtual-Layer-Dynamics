r"""Virtual Layer Dynamics (`vld`) — Consolidated Computational Ground Architecture.

Active consolidated package layers (`v0.2.0`):
  - `vld.canonical`: Deterministic JSON & domain-separated SHA-256 addressing
  - `vld.triad`: Kleene $\mathcal{K}_3$ guards, epistemic tags, Myhill-Nerode quotient & erasure bits
  - `vld.ground`: `VirtualGroundSpec`, reversible `ByteLens`, dual-derivation `AffineOrbitLaw` & Pareto costs
  - `vld.journal`: Multi-process atomic hash-chained receipt journal with idempotency fencing
  - `vld.signal`: Multi-domain backends (`Signal8Backend`, `ModularRingBackend`, `ExactRationalBackend`),
    static instruction footprints, reversible `SignalStateMachine`, and base-8 transition lowering
  - `vld.memory`: M1 `PagedObjectStore` ($O(L/P)$ range reads), 5-rung `CertifiedLaw` ladder,
    exact 2-adic/$\gcd$ inverse locators, and counterexample-demoting `StructuralMemoryRegistry`
  - `vld.inquiry`: Active perception (`greedy`, `beam`, `minimax` probe planners) & `HypothesisRevisionLedger`
  - `vld.accel`: Sufficient-statistic `StructuralLinearOperator`, `LinearRecurrenceGround`,
    dual `stdlib`/`numpy` `NumericEngine`, and optional LLVM-compiled Rust CDLL kernels (`vld.accel.native`)
  - `vld.fabric`: Multi-framing `VantageEnvelope` codec, `StreamPipeVantage`, and RFC 9110 `Range`-enabled
    `VLDHttpVantageServer` / `VLDHttpVantageClient`
  - `vld.models`: Safetensors Range observer (`HFR`), `ByteLevelBPETokenizerGround`, `Qwen3` dense & sparse MoE
    realizers, crash-resumable `AutoregressiveModelRunner`, constructive piecewise-affine & FSM neural compilers,
    Net2Wider morphisms, and `GenerativeWeightGround`
  - `vld.agents`: `ProofGround` AND/OR hypergraph, `ReceiverHypervisor`, and `InLayerModelAgent`
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

__version__ = "0.2.0"

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
