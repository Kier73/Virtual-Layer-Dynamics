r"""Virtual Layer Dynamics (`vld`) — Consolidated Computational Ground Architecture.

Active consolidated package layers (`v0.7.0` — Phases 0 through 10 Complete & Hardened):
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
    interval & strong-bisimulation pre-gates, `ComposedGroundNavigator`, $\mathbb{F}_p$ commutant similarity blocks,
    and `CrossObserverCongruenceGate`
  - `vld.accel`: Sufficient-statistic `StructuralLinearOperator`, `LinearRecurrenceGround`, `F8_E4M3`/`UE8M0`/`FP4`
    codecs, Birkhoff Sinkhorn `mHC` kernels, YaRN/interleaved RoPE, and NumPy + Rust CDLL acceleration
  - `vld.fabric`: Multi-framing `VantageEnvelope` codec, `ReplayWindowGuard`, crash-resumable `ReliableChannel`,
    zero-dependency X.509 v3 mTLS mesh, `ComputationalFabricRouter`, and `PartitionedTensorCutCoordinator`
  - `vld.substrates`: `HeadlessOS` (`x86_64` POSIX ABI, CoW `fork`, `pipe`, transactional `execve`, M1 OS snapshots),
    ELF64 & WASM lifters, SIMT barriers, $O(1)$-delta tensor logs, IPv4 router-hop & Internet Path Grounds, and `ROOM`s
  - `vld.models`: `HostedModelGlance`, `LazyFrontierWeightGround`, `DeepSeek-V4.1` (`mHC` + `FP8`/`FP4` + Sink-MLA)
    & `GLM-5.3` (`MLA` Matrix Absorption + hybrid MoE) realizers, `Qwen3`, and constructive SNC/PNF ReLU compilers
  - `vld.agents`: `ProofGround` AND/OR hypergraph, `ReceiverHypervisor`, `InLayerModelAgent`, and
    `ContinuousLayerBuilderAgent`
  - `vld.continuous`: Quorum-fenced, crash-resumable `ContinuousVirtualSystem` across changing Grounds,
    representations, hosts, and physical paths (`GGE-001..007`)
  - `vld.arbitration`: Multi-source Virtual-Layer model arbitration (`DeepSeek`, `GLM`, `Qwen`, `Llama`/`SmolLM`,
    HTTP & procedural), recursive `InLayerMetaArbitratorModel`, `ContinuousArbitrationService`, and `IdeaLedger`
  - `vld.compiler`: `IntentionToRealizationCompiler`, congruence-closed `ProofProducingEGraph` with standalone
    rewrite proof verification, interchangeable multi-backend Pareto realization, and `EffectLedger`
  - `vld.independent_audit`: Standalone zero-import cryptographic & mathematical verifier for Ed25519/HMAC receipts,
    rewrite proofs, cellular sheaf $H^1$ obstructions, precision certificates, and neural capsules
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

__version__ = "0.7.0"

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
