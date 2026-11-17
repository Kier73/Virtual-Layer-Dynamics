r"""Virtual Layer Dynamics (`vld`) — Consolidated Computational Ground Architecture.

Active consolidated package layers (`v0.3.0` — Phases 0 through 7 Complete):
  - `vld.canonical`: Deterministic JSON & domain-separated SHA-256 addressing
  - `vld.triad`: Kleene $\mathcal{K}_3$ guards, epistemic tags, Myhill-Nerode quotient & erasure bits
  - `vld.ground`: `VirtualGroundSpec`, reversible `ByteLens`, dual-derivation `AffineOrbitLaw` & Pareto costs
  - `vld.journal`: Multi-process atomic hash-chained receipt journal with idempotency fencing
  - `vld.signal`: Multi-domain backends (`Signal8Backend`, `ModularRingBackend`, `ExactRationalBackend`),
    static instruction footprints, reversible `SignalStateMachine`, and base-8 transition lowering
  - `vld.memory`: M1 `PagedObjectStore` ($O(L/P)$ range reads), 5-rung `CertifiedLaw` ladder,
    exact 2-adic/$\gcd$ inverse locators, and counterexample-demoting `StructuralMemoryRegistry`
  - `vld.inquiry`: Active perception (`greedy`, `beam`, `minimax` probe planners) & `HypothesisRevisionLedger`
  - `vld.authority`: Lossless temporal coordinates, scheduler-scoped vector clocks (`G13`), decoupled
    `EncounterEnvelope` provenance (`G21`), $2q > N$ quorum fencing, M3 CAS sessions & `VirtualProcessor` continuity
  - `vld.navigation`: Unbounded lazy `NavigableComputationDAG`, Kleene $\mathcal{K}_3$ dynamic guards,
    interval abstract-interpretation pre-gates, and partition-refinement strong bisimulation
  - `vld.accel`: Sufficient-statistic `StructuralLinearOperator`, `LinearRecurrenceGround`,
    contiguous `ndarray`-resident `NumericEngine`, and optional LLVM-compiled Rust CDLL kernels
  - `vld.fabric`: Multi-framing `VantageEnvelope` codec, `ReplayWindowGuard`, crash-resumable `ReliableChannel`,
    zero-dependency X.509 v3 mTLS peer mesh, HTTP/HTTPS 206 `Range` vantages, and `G11` terminal-to-terminal forwarding
  - `vld.substrates`: CoW + Structural-Memory `VirtualPageSpace`, `DeterministicVMKernel`, ELF64 & binary WASM
    process Ground lifters (`Locate`/`Restore`/`Continue`/`Transform`), SIMT barriers, $O(1)$-delta tensor
    checkpoints, active Path-Native Grounds with negative controls, and portable computational `ROOM`s
  - `vld.models`: Safetensors Range observer (`HFR`), `ByteLevelBPETokenizerGround`, `Qwen3` dense & sparse MoE
    realizers, crash-resumable `AutoregressiveModelRunner`, constructive 1D/multivariate ReLU & FSM compilers,
    Net2Wider/rescaling morphisms, and `GenerativeWeightGround`
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

__version__ = "0.3.0"

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
