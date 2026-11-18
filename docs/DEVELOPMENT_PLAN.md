# Virtual Layer Dynamics — Master Consolidation & Development Plan

This roadmap defines the architecture, module hierarchy, mathematical contracts, and falsifiable completion gates for consolidating all predecessor capabilities (`E-CPU`, `Virtual-Ground-Interpreter`, `E-CPU-Virtual-Processor`, `In-Formation-Space`) into **Virtual Layer Dynamics (`vld`)**, culminating in substrate-adaptive distributed execution and Virtual-Layer-native Large Language Models and hypervisors.

---

## Target Consolidated Package Architecture (`src/vld/`)

```text
src/vld/
├── canonical.py        # [Phase 0 - Done] Deterministic JSON & domain-separated SHA-256
├── triad.py            # [Phase 0 - Done] Kleene K_3 guards, Myhill-Nerode quotient, erasure bits
├── ground.py           # [Phase 0 - Done] VirtualGroundSpec, ByteLens, AffineOrbitLaw, Pareto costs
├── journal.py          # [Phase 0 - Done] Atomic hash-chained receipt journal & idempotency fencing
├── cli.py              # [Phase 0-7 - Done] Unified `vld` verify, navigate, benchmark & serve CLI
├── signal/             # [Phase 1 - Done] Phasor/ring/rational backends, typed IR, static footprints
├── memory/             # [Phase 2 - Done] Structural Memory registry, 5-rung certifier, M1 paged store
├── inquiry/            # [Phase 2 - Done] Active perception, greedy/beam/minimax probes, revision ledger
├── authority/          # [Phase 3 - Done] Scheduler-scoped clocks (G13), 2q>N quorum M3 sessions & VP continuity
├── navigation/         # [Phase 4 - Done] Unbounded lazy NavigableComputationDAG, interval & bisimulation gates
├── accel/              # [Accel - Done] StructuralLinearOperator, ndarray-resident NumPy + Rust CDLL kernels
├── fabric/             # [Phase 4 - Done] VantageEnvelope, ReplayWindowGuard, ReliableChannel, mTLS & G11 forwarding
├── substrates/         # [Phase 5 - Done] CoW + Structural VM pages, ELF64/WASM lifters, O(1) delta tensor & Path Grounds
├── models/             # [Phase 6-7 - Done] Safetensors HFR, BPE, Qwen3 dense/MoE, orbit recovery, SNC/PNF
└── agents/             # [Phase 7 - Done] ProofGround, ReceiverHypervisor CAS/capabilities & ModelAgents
```

Every layer obeys a strict acyclic dependency order:
`canonical` $\to$ `triad` $\to$ `ground` / `journal` $\to$ `signal` / `memory` / `inquiry` $\to$ `authority` $\to$ `navigation` / `fabric` $\to$ `substrates` $\to$ `models` $\to$ `agents`.

---

## Phased Engineering Roadmap & Falsifiable Gates

### Phase 0 — Mathematical Kernel, Byte Lenses, Triad Analyzer & Durable Journal *(Completed)*
- **Scope:** Establish repository hygiene, `.gitignore` boundary for local research trees, LF normalization, zero-warning Python 3.11–3.14 packaging, deterministic canonicalization, Kleene $\mathcal{K}_3$ interval guards, Moore/Myhill-Nerode predictive quotient + transition erasure analyzer, reversible `ByteLens`, dual-derivation `AffineOrbitLaw` with strict wrap-identifiability, Pareto cost vectors, and atomic `HashChainedJournal`.
- **Predecessor Bugs Resolved:** Python 3.14 docstring escape warnings (`-W error::SyntaxWarning`), unwrapped modular progression mis-certification (`HYPOTHESIS` vs `CERTIFIED`), and external site-packages namespace collisions.
- **Verification Gate:** 100% pass on `pytest` + `python -m vld verify` + clean wheel/sdist build across Windows and Linux.

---

### Phase 1 — Signal-State Machine, Multi-Domain Backends & Compositional Footprints (`vld.signal`) *(Completed)*
- **Consolidates:** `vgi.values`, `vgi.ground`, `vgi.ir`, `vgi.compiler`, `vgi.machine`, `vgi.transitions` (`G01`–`G03`).
- **Key Deliverables:**
  1. Extract formal `ComputationalBackend` protocol decoupling machine control flow from domain algebra.
  2. Implement `Signal8Backend` ($\mathbb{Z}/8\mathbb{Z}$ unit phasors, observer reframing $R_{u \to v}$, I/Q pilot calibration) **plus** a second distinct backend (`ModularRingBackend` $\mathbb{Z}/2^w\mathbb{Z}$ and `ExactRationalBackend` $\mathbb{Q}$) with differing algebraic laws and explicit capability declarations.
  3. Attach explicit read/write memory footprints, preconditions, postconditions, and reversibility metadata to every IR instruction; reject invalid compositions statically before execution.
  4. Consolidate multi-digit base-8 `GroundTransitionSystem` lowering with differential trace equivalence checks against direct transition semantics.
- **Falsifiable Gate:** Exhaustive 4,772-case Signal8 invariant suite passes alongside differential execution across at least two non-isomorphic backends; indeterminate guards trap atomically with zero state mutation; static footprint checker rejects write-after-trap and undeclared cell accesses.

---

### Phase 2 — Structural Memory, Active Inquiry & Observation-to-Ground Pipeline (`vld.memory`, `vld.inquiry`) *(Completed)*
- **Consolidates:** `mailcpu.structural_memory`, `mailcpu.law_certifier`, `mailcpu.composed_ground`, `mailcpu.computational_objects` (M1), `vgi.partial_attention`, `vgi.partial_policy`, `ifs.bridges.hypothesis_revision` (`G04`, `G07`, `G12`, `G25`).
- **Key Deliverables:**
  1. **M1 Content-Addressed Paged Byte Store:** Fixed-size SHA-256 verified pages, streaming manifests, and bounded range reads without materializing entire multi-gigabyte objects in RAM.
  2. **Structural Memory & Law Certifier:** Revision-pinned generative state families (`int` progression, modular affine/LCG, $\operatorname{GF}(2)$ linear feedback, finite-orbit cycles, and piecewise-polynomial lifts) with strict identifiability witnesses and inverse lookup (`locate_value`) where bijective/2-adic structure permits.
  3. **Active Perception & Hypothesis Revision:** Import partial/noisy observations into versioned candidate Grounds; run `greedy`, `beam`, and `minimax` distinguishing-probe planners to resolve `UNKNOWN`/`AMBIGUOUS` cells; invalidate dependent certificates cleanly when a held-out counterexample arrives.
- **Falsifiable Gate:** Counterexample injection demotes a candidate Ground version and invalidates downstream read receipts without erasing prior audit history; direct $O(1)$ / $O(\log n)$ Structural Memory reads expand zero intermediate states and match full replay on bounded horizons.

---

### Phase 3 — Receiver Authority, Quorum Sessions & Portable Provenance (`vld.authority`) *(Completed)*
- **Consolidates:** `mailcpu.virtual_layer`, `mailcpu.classifier` (Hypervisor), `mailcpu.computational_session` (M3), `mailcpu.recovery_lineage`, `vp.core`, `ifs.formats.*`, `ifs.bridges.temporal_provenance` (`G13`, `G21`, `G22`).
- **Key Deliverables:**
  1. **Typed Temporal & Encounter Envelopes:** Vector clocks, scheduler-scoped coordinates (preventing cross-scheduler silent misinterpretation, resolving `G13`), and decoupled observation-vs-interpretation envelopes.
  2. **Receiver-Owned Hypervisor & Capability Kernel:** Every state mutation or operator invocation requires live capability verification, Ground/context freshness, and compare-and-swap (`CAS(digest(s), s')`) commit at the state owner.
  3. **Quorum-Fenced Sessions (M3) & Processor Continuity:** Strict-majority ($2q > N$) high-water lineage sealing M1 state manifests; portable Virtual Processor identity surviving complete process destruction and host replacement while rejecting rollback and cloned-credential equivocation.
- **Falsifiable Gate:** Hard-kill (`os._exit`) mid-transition followed by recovery on a fresh host instance preserves exact semantic execution once, refuses stale epoch/revision writes, and rejects cross-scheduler coordinate confusion.

---

### Phase 4 — Navigable Computation DAGs & Authenticated Distributed Fabric (`vld.navigation`, `vld.fabric`) *(Completed)*
- **Consolidates:** `mailcpu.computation_dag`, `mailcpu.navigation*`, `mailcpu.reliability`, `mailcpu.mtls_transport`, `mailcpu.computational_fabric*`, `mailcpu.partitioned_dataflow` (`G08`, `G11`, `G14`, `G15`, `G20`).
- **Key Deliverables:**
  1. **Hierarchical Navigable Computation DAG:** Content-addressed dependency frontiers removing the legacy 256-node flat ceiling via lazy subgraph expansion; static/dynamic readiness, checkpoint/restore, and strong-bisimulation / interval-abstract-interpretation pre-gates.
  2. **Authenticated Fabric & Direct Terminal-to-Terminal Dataflow:** Mutual-TLS peer discovery, crash-resumable `ReliableChannel` windows (separating transport sequence numbers from semantic operation IDs), range-resumable M1 remote page streaming (keeping bulk bytes out of 4 KiB control frames), and direct predecessor-host-to-consumer-host tensor/page forwarding while retaining commit authority at the origin coordinator (`G11`).
- **Falsifiable Gate:** Multi-process mTLS cluster transfers multi-megabyte M1 objects across simulated link drops and process restarts with zero control-plane bloat and verified terminal-to-terminal frontier reduction.

---

### Phase 5 — Virtual OS, Hardware/Bytecode Lifting & Substrate/Path-Native Realization (`vld.substrates`) *(Completed)*
- **Consolidates:** `mailcpu.interpreter`, `mailcpu.kernel`, `mailcpu.vm` (CoW + structural demand paging), `mailcpu.elf`, `mailcpu.egpu`, `mailcpu.tensor_stream`, `mailcpu.tensor_checkpoint`, `mailcpu.path_ground*`, `mailcpu.internet_ground*`, `experiments.portable_rooms` (`G09`, `G10`, `G23`, `G24`, `G26`–`G29`).
- **Key Deliverables:**
  1. **Software & OS Lifting:** Deterministic VM kernel, Copy-on-Write + Structural-Memory-backed virtual pages, transactional ELF/WASM lifting into navigable process Grounds (`Locate`, `Restore`, `Continue`, `Transform`).
  2. **SIMT / E-GPU & O(1)-Incremental Tensor Recovery:** Workgroup barrier semantics, tiled matrix/convolution operators, and append-only delta tensor checkpoints (eliminating the $O(K^2)$ cumulative prefix-copy overhead noted in `PR183`) integrated with receiver operation identities.
  3. **Active Host Formation & Path-Native Grounds:** Qualify external ports, HTTP/Git caches, kernel packet-filtering queues, and authenticated network intermediaries as bounded observer-hosts (`observe -> qualify -> construct -> admit -> navigate -> verify`), with mandatory negative controls distinguishing genuine in-path transformation from endpoint compute.
- **Falsifiable Gate:** Portable computational room (`ROOM`) produces identical verified outputs across native VM, WASM realizer, and hard-killed/rehomed host; incremental tensor matmul resumes in $O(1)$ per-tile I/O after `os._exit` and rejects forged tile digests.

---

### Phase 6 — Hosted & Partitioned Large Language Models Inside the Virtual Layer (`vld.models`) *(Completed)*
- **Consolidates:** `mailcpu.model_weight_ground` (`MWS-001/002`), `mailcpu.hf_model_ground` / `hf_layer_realization` / `model_orbit_handoff` (`HFG-001..003`), `vp.moe` (`VP-002`), `ifs.experiments.external_neural_structure_reading` (`HFR`), `ifs.experiments.tensor_cut_realization` (`TCR`), `ifs.experiments.aks*` (`G16`–`G19`).
- **Key Deliverables:**
  1. **Source-Pinned Model Artifact & Tokenizer Grounds:** Bounded Safetensors header/range reader (`HFR`), deterministic source-pinned BPE tokenizer Ground with round-trip receipts, and hierarchical lazy `ModelParameterSpace` supporting 0.6B dense through 235B+ sparse MoE architectures without full checkpoint download.
  2. **Verified Transformer & Sparse MoE Numerical Realizers:** Hypervisor-governed operators for RMSNorm, QKV + per-head norm, RoPE, causal GQA, deterministic top-$k$ MoE routing, and active-expert-only SwiGLU execution, checked against independent reference oracles under explicit BF16/FP32 error contracts.
  3. **Paged KV-State Ground & Multi-Host Model-Orbit Continuity:** Content-addressed M1 KV-cache pages, sufficient-state prefix reuse where algebraically valid (and explicit refusal where softmax forbids affine collapse), and quorum-fenced layer-to-layer / token-to-token handoffs across distinct hosts.
- **Falsifiable Gate:** Execute end-to-end multi-token autoregressive generation on a pinned real open-weight model (`Qwen3-0.6B` dense and sparse MoE layer slices) through `vld`, kill the active host mid-decode, resume from M1/M3 KV + orbit checkpoint on a second host, and match independent PyTorch reference tokens/logits within declared tolerance.

---

### Phase 7 — Procedural Model Generation, Synthetic Weight Geometry & Agentic Hypervisors (`vld.models.procedural`, `vld.agents`) *(Completed)*
- **Consolidates:** `ifs.experiments.semantic_neural_compilation` (`SNC-001`), `ifs.experiments.procedural_neural_formation` (`PNF-001..003`), `mailcpu.generative_weight_ground` (`MWS-002`), `mailcpu.weight_space` / `stateful_transition`, `vgi.proof` (`ProofGround`), `ifs.bridges.model_native_operations` (`USM-001..003`) (`G05`, `G06`, `G18`).
- **Key Deliverables:**
  1. **Constructive Semantic-to-Neural Compilation:** Compile symbolic specifications, finite-state machines, and multivariate piecewise-affine/rational atlases directly into verified neural subnetworks (hinge/polyhedral compilation, finite-transition synthesis) with zero training data and whole-domain activation-cell certificates.
  2. **Procedural Weight & Geometry Morphisms:** Operate directly on model geometry inside the Virtual Layer—verified width expansion (dyadic Net2Wider), depth identity insertion, compensated permutation/rescaling symmetries, low-rank + Structural-Memory law composition, and synthetic task/trajectory generation—preserving tracked lineage between parent and child model Grounds.
  3. **Models as Interactable Virtual-Layer Agents & Advisory Hypervisors:** Host procedural and pretrained models as first-class citizens inside `vld` sessions that observe Ground telemetry, propose candidate state laws, design active disambiguating probes, schedule MoE/fabric partitions, and search `ProofGround` obligations—while every proposed action passes through deterministic receiver Hypervisor capability gates, law certifiers, and independent verifiers.
- **Falsifiable Gate:**
  - A procedurally synthesized/morphed model capsule transfers across two independent processes, passes whole-domain or bounded-probe equivalence verification without importing producer compiler internals, and executes natively inside `vld`.
  - An in-layer model agent autonomously explores an unknown finite/affine environment via budgeted probes, proposes a Ground law, obtains deterministic certification, and navigates a $10^{12}$-step target query—while adversarial/ungrounded proposals from the same agent interface are deterministically refused by the Hypervisor.

---

### Phase 8 — Frontier Model Scale (`DeepSeek-V4.1` & `GLM-5.3`), `HeadlessOS`, Federated Fabric Routing & Unified Continuous Operation (`vld.continuous`) *(Completed)*
- **Consolidates:** `HeadlessOS` POSIX kernel (`G27`, `G28`), `ComputationalFabricRouter` & `PartitionedTensorCutCoordinator` (`TCR-001/002`, `G19`), `ComposedGroundNavigator`, $\mathbb{F}_p$ commutant similarity blocks & `CrossObserverCongruenceGate` (`GCI-001..005`), IPv4 router-hop & bounded Internet Path Grounds (`PATH-GROUND-004`, `INTERNET-004`, `INTERNET-005`), `ContinuousLayerBuilderAgent`, Frontier `DeepSeek-V4.1` & `GLM-5.3` Grounds, and `ContinuousVirtualSystem` (`GGE-001..007`).
- **Key Deliverables:**
  1. **Frontier Open-Weight Model Glancing & Live Range Execution:** `HostedModelGlance` and `LazyFrontierWeightGround` lift 500 GB – 1.5 TB+ checkpoints (`deepseek-ai/DeepSeek-V4.1-Flash` at `510.29 GB` and `zai-org/GLM-5.3-BF16` at `1.507 TB`) from metadata headers alone, paired with 3-way `Rust == NumPy == Stdlib` kernels for `F8_E4M3` + `UE8M0` block dequantization, `MXFP4` (`E2M1`) expert unpacking, Birkhoff-Sinkhorn `mHC` ($m=4$) mixing, Sink-MLA with conjugate inverse tail RoPE, and `GLM-5.3` `MLA` Matrix Absorption (`56.89x` KV compression).
  2. **Booted `HeadlessOS`, Federated Fabric Routing & Internet/Router Path Grounds:** 20-syscall Linux `x86_64` POSIX ABI (`fork` CoW, `pipe` IPC, transactional `execve`, W^X `mmap`/`mprotect`, petabyte `StructuralMemory` virtual files, M1 CoW OS snapshot/restore), multi-metric `ComputationalFabricRouter` + 4-worker mTLS SwiGLU tensor-cut execution with epoch-fenced crash failover and `math.fsum` scalar verification, and `RFC 1071` IPv4 router-hop / RFC hypertext / evidence-gated observation routers.
  3. **Unified Self-Improving `ContinuousVirtualSystem`:** Operates all substrate, OS, path, orbit, fabric, and frontier-model Grounds as a single quorum-fenced, write-ahead crash-resumable computational pipeline with `ClockBridgeReceipt` causal continuity and autonomous `ContinuousLayerBuilderAgent` Pareto upgrades (`vld continuous`).
- **Falsifiable Gate:**
  - Live HTTPS `206` Range glancing + layer execution verified against both real `510.29 GB` `deepseek-ai/DeepSeek-V4.1-Flash` (executing full Layer 0 touching only `12/384` `FP4` experts, `0.0775%` of checkpoint bytes) and real `1.507 TB` `zai-org/GLM-5.3-BF16` (verifying `0.0` float32 max error on Layer 0 64-head `MLA` Matrix Absorption touching `1/282` shards, `0.104%` of checkpoint bytes).
  - Multi-stage `ContinuousVirtualSystem` survives real child subprocess `os._exit(73)` destruction and mid-pipeline host replacement across intersecting $2q > N$ quorums with zero stage replay, fences zombie predecessor terms, and achieves $\mathbf{C}_{\text{Pass 2}} \prec \mathbf{C}_{\text{Pass 1}}$ after autonomous `ContinuousLayerBuilderAgent` upgrades.

---

### Phase 9 — Peer-Audit Hardening, Fluid In-Transit Computation, Universal Model Grounds & Recursive Model Arbitration (`v0.6.0`) *(Completed)*
- **Key Deliverables:**
  1. **Peer-Audit Verification & Security Hardening:** Fixed `.github/workflows/ci.yml` unquoted YAML colon defect (+ automated workflow syntax test), enforced atomic `0600` private-key creation (`os.open(..., 0o600)`) and `0700` directory permissions in `generate_ephemeral_mtls_pki`, enforced 7-invariant inductive genesis-input & stage-spec recovery binding + mandatory independent `verify_fn` + separated empirical `MeasuredStageTelemetry` in `ContinuousVirtualSystem`, and built standalone zero-import `vld.independent_audit`.
  2. **Fluid In-Transit Computation Substrate (`vld.substrates.fluid_transit`):** Lifted CPU loops (`TropicalMinPlusMatrix`, `FiniteTransducerSummary`), GPU multi-shard attention (`OnlineFlashStreamState`), Virtual Memory movements (`VirtualMemoryMovementView` with contravariant pullback commutation), and path-encoded emulated machines (`InTransitPathMachine` + `optimize_transit_schedule`) across stream and `HeadlessOS` POSIX pipe hops.
  3. **Universal Multi-Family Model Realizers & Self-Hosted Recursive Model Arbitration (`vld.models`, `vld.arbitration`):** Generalized `TransformerArchitectureSpec` and `TransformerLayerRealizer` to `llama` (`SmolLM2`), `qwen2` (`DeepSeek-R1-Distill`), and `glm4` (`GLM-Z1` 4-norm sandwich + fused `gate_up_proj` + interleaved `0.5` partial RoPE) alongside `qwen3`, `qwen3_moe`, `DeepSeek-V4.1`, and `GLM-5.3`; added `AgentCapability.ARBITRATE_MODEL`, `InLayerMetaArbitratorModel` (Depth-1 $\to$ Depth-2 recursive in-layer arbitration), `ContinuousArbitrationService` (`vld arbitrate`), and `IdeaLedger` (`docs/IDEA_LEDGER.md`).
