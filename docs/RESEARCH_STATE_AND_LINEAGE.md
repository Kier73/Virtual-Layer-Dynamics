# Research State & Predecessor Lineage

`Virtual-Layer-Dynamics` consolidates four substantial research codebases totaling **4,836 source/archive files, 1,586 documents, 1,541 Python modules, and 194 workflow specifications** into a single cohesive, rigorously tested production architecture.

This document records the exact technical state of each predecessor system at consolidation intake (2026-11-17), distinguishing **empirically verified local capabilities** from **partial integrations** and **open research frontiers**.

---

## 1. Predecessor Inventory & Responsibilities

| Predecessor Project | Scale (Snapshot) | Core Architectural Contributions | Upstream Reference |
|---|---|---|---|
| **E-CPU (`mailcpu`)** | 676 files · 457 Python modules · 143 docs | Computational Grounds, Structural Memory & law certification, MailCPU/HeadlessOS deterministic VM & ELF loader, receiver-local Hypervisor & `VirtualLayerRuntime`, content-addressed M1 byte pages, M2 artifact hosting, M3 quorum-fenced sessions, `NavigableComputationDAG`, mTLS/socket fabric, SIMT/E-GPU verified-prefix tensor checkpoints (`PR154`/`PR183`), `PATH-GROUND-001..006` / `INTERNET-001..003`, and `MWS-001..002` / `HFG-001..003` neural weight & Qwen3 layer Grounds. | `Kier73/E-CPU` |
| **Virtual Ground Interpreter (`vgi`)** | 2,669 files · 411 Python modules · 1,137 docs (active `v0.2.0` + 31 frozen builds) | Signal-native `Z/8Z` phasor machine (`SignalValue(z, u, eps)`), unit-character observer reframing $R_{u \to v}$, three-valued uncertainty guards (`TRUE`, `FALSE`, `INDETERMINATE` trap), multi-digit base-8 `GroundTransitionSystem` compiler (`PR32`), Moore/relational quotients, active perception & partial-probe planners (`greedy`, `beam`, `minimax`, `exhaustive`), exact rational/orbit/polynomial lifts, and `ProofGround` AND/OR proof-obligation graphs. | `Kier73/Virtual-Ground-Interpreter` |
| **E-CPU Virtual Processor (`vp`)** | 14 files · 6 Python modules · 3 docs | `VP-001` durable processor identity via hash-chained JSONL receipts across ephemeral runners, mandatory dual-derivation affine orbit jumps (`lcg_navigate == affine_orbit_jump`), and `VP-002` sparse MoE architecture Ground (`Qwen3-30B-A3B`, `Qwen3-235B-A22B`) with deterministic top-8 router verification and 25-node layer-local DAG frontiers. | `Kier73/E-CPU-Virtual-Processor` |
| **In-Formation-Space (`ifs`)** | 1,477 files · 667 Python modules · 303 docs | Wire contracts (`encounter-envelope`, `canonical-metadata-v1`, `temporal-coordinate-v2`), 9 cross-system bridges (`temporal-provenance`, `observation-path`, `observation-equivalence`, `hypothesis-revision`, `authority-lineage`, `proof-attention`, `model-native-operations`, etc.), Hugging Face bounded range reader (`HFR`), constructive piecewise-affine-to-ReLU compiler (`SNC-001`), Net2Wider dyadic neural formation (`PNF-001..003`), finite-field affine-kernel attention (`AKS-001..003`), and the `G01–G29` System Ambition Gap Ledger. | `Kier73/In-Formation-Space` |

---

## 2. Empirical Baseline Verification Record (2026-11-17, Windows / Python 3.14)

Rather than relying solely on historical run reports, representative test suites across all four codebases were executed locally under isolated `PYTHONPATH` and `PYTHONDONTWRITEBYTECODE=1`:

1. **`Virtual-Ground-Interpreter` (`vgi` v0.2.0):**
   - **Result:** **325/325 unit tests passed**; `python -m vgi verify` passed all **4,772 exhaustive finite `Z/8Z` Signal8 cases** (`ground_fingerprint = 4ba599c21ec7e446572dad61b2a66ecb9c0844fd3dee406e4d947e04db56c43b`).
   - **Verified Status:** Signal arithmetic, observer naturality, reframe round-trips, conditional traps, and multi-cell base-8 finite Ground compilation (`G01`) are production-ready for consolidation.

2. **`E-CPU-Virtual-Processor` (`vp`):**
   - **Result:** **27/27 unit and native-integration tests passed**.
   - **Verified Status:** Hash-chained journal integrity, tamper refusal, idempotent replay, dual affine derivation, and 235B MoE routing/frontier planning pass cleanly. Note that `VP-002` is a verified architectural/routing planner—it does not yet load pretrained weights or execute floating-point expert kernels.

3. **`In-Formation-Space` (`ifs`):**
   - **Result:** `bridges/model-native-operations` (**14/14 passed**), `experiments/semantic_neural_compilation` (**10/10 passed**), `experiments/procedural_neural_formation` (**13/17 passed**, 4 setup errors solely due to optional `jsonschema` package absence in that shell).
   - **Verified Status:** Independent stdlib verifiers, exact rational ReLU hinge synthesis (`SNC-001`), and dyadic Net2Wider neural block morphisms (`PNF-001`) are mathematically sound and reproducible.

4. **`E-CPU` (`mailcpu`):**
   - **Result:** Tensor streaming & atomic verified-prefix checkpoint tests (`test_tensor_paths.py`, `test_tensor_checkpoint.py`) passed **23/23**. Across 13 core Ground, DAG, VGI-session, weight-space, and Qwen3-layer test modules (`139` tests), **130 passed, 3 skipped (mTLS/OpenSSL fixture), and 6 failed**.
   - **Root-Cause Diagnosis of the 6 Failures (Addressed by Consolidation):**
     - *Defect A — Raw docstring escape + strict `stderr` framing (4 failures in `test_ground_native_realization.py`):* `mailcpu/computation_dag.py:274` used an unescaped `\d` inside a non-raw mathematical docstring, triggering a Python 3.14 `SyntaxWarning` on `stderr`, which `SubprocessQualificationEndpoint.exchange()` treated as a fatal framing violation. **Fix in `vld`:** Enforce raw docstrings `r"""..."""` and `-W error::SyntaxWarning` in `pyproject.toml` CI from day one, plus explicit line-ending normalization on subprocess channels.
     - *Defect B — Unwrapped progression identifiability leak (1 failure in `test_observed_orbit_pipeline.py`):* An unwrapped monotonic sample trace was certified as an unbounded linear `int` law (`53`) instead of requiring a modular wrap witness (`90`) or remaining `HYPOTHESIS`/`FITTED`. **Fix in `vld`:** Implemented directly in `vld.ground.propose_modular_progression` / `certify_modular_progression` with explicit regression tests.
     - *Defect C — Cross-module API drift (1 failure in `test_hf_model_ground.py`):* Caller invoked `dag.is_complete()` as a method while `NavigableComputationDAG.is_complete` is a `bool` `@property`. **Fix in `vld`:** Single monorepo type-checked API surface eliminates cross-PR/cross-repo commit-pin drift.

---

## 3. Structural Bottlenecks Solved by Consolidation

Across the four separate repositories, development velocity and reliability were constrained by four structural friction points:

1. **Multi-Repository Commit-Pin Drift:** `E-CPU` pinned an older `VGI` SHA; `VP` pinned an older `E-CPU` SHA; `IFS` experiments pinned dozens of distinct `E-CPU` and `VGI` SHAs across 147 workflow files. Fixing an interface in one repo could not be atomically tested across consumers.
2. **Duplicate Concept Implementations:** Canonical JSON hashing, hash-linked journals, affine LCG navigation, transition quotients, and neural graph descriptors were re-implemented slightly differently across `mailcpu`, `vgi`, `vp`, and `ifs`.
3. **Stacked Feature Branches Without Unified Gate:** Important capabilities (host formation `SAB`, model-operated `USM`, network-queue state `CAPACITY`, portable rooms `ROOM`) accumulated across separate branches without a single fast, deterministic local + CI gate.
4. **Historical Archive Coupling:** VGI's 31 frozen historical builds (including intentionally retained negative results like Build 12's onset-window artifact) lived alongside active runtime code. Separating local research archives (gitignored in the workspace) from the clean `src/vld/` production tree preserves every historical lesson while keeping the runtime crisp and fast.

---

## 4. Non-Negotiable Epistemic & Engineering Guardrails Carried Forward

Consolidation preserves the hard-won scientific discipline established across the research corpus:

- **Fitted $\neq$ Certified:** A formula matching a finite sample prefix is a `HYPOTHESIS`, not an admitted Ground law, unless identifiability conditions and domain bounds hold.
- **Described $\neq$ Executed $\neq$ Verified:** Constructing a 235B MoE computation DAG (`VP-002`) is architectural planning, not neural inference; transport `ACK` is frame delivery, not receiver state commit; self-recomputed hashes prove byte integrity, not mathematical correctness or independent authority.
- **Uncertainty Never Guesses:** Three-valued guards (`TRUE`, `FALSE`, `INDETERMINATE`) and partial transition tables (`UNKNOWN`, `AMBIGUOUS`) trap or trigger explicit inquiry rather than fabricating silent fallbacks.
- **Receiver Owns Authority:** Senders propose intentions and routes; only the state-owning receiver validates live capabilities, context freshness, and compare-and-swap commits.

---

## 5. Phase 8 Live Frontier & Unified Continuous Verification Record (2026-11-18)

Following completion of Phases 0–7 (`v0.4.0`), a second predecessor research pass across general Ground execution (`GGE-001..007`), `HeadlessOS` (`G27`, `G28`), partitioned tensor cuts (`TCR-001/002`, `G19`), observer-orbit / $\mathbb{F}_p$ commutant similarity blocks (`GCI-001..005`), and router/Internet Path Grounds (`PATH-GROUND-004`, `INTERNET-004`, `INTERNET-005`) was consolidated into `v0.5.0` and verified against live frontier checkpoints:

1. **Live `deepseek-ai/DeepSeek-V4.1-Flash` (`510.29 GB`, 48 shards, 96,085 tensors, SHA `314113f8a7dadb388215396ba5fbee2c610a04d6`):**
   - Glanced into a pinned `VirtualGroundSpec` (`1e83635d543c3ca7...`) by reading `7,325.2 KB` of JSON/header metadata (`0.0014%` of checkpoint), identifying `128.0x` single-KV Sink-MLA compression, `6/384` `sqrtsoftplus` `noaux_tc` routed `FP4` experts + `1` `FP8` shared expert, and `mHC` width $m=4$.
   - Executed full Layer 0 (`compress_ratio=0`) and full Layer 2 (`compress_ratio=2`, `Compressor` + `FP4` `Indexer` + `compress_rope_theta=160,000` YaRN) end-to-end over HTTPS `206` Range on prompt `"Virtual Layer Dynamics lifts computation."`, pooling 2 compressed KV latents (`compressed_kv_selected = [[-1, -1], [0, -1], [0, -1], [0, 1]]`), materializing only `12`–`15` of `384` `FP4` experts (`369`–`372` untouched on Hugging Face Hub), and transferring `395.69`–`446.60 MB` (`< 0.088%` of the `510.29 GB` checkpoint).
2. **Live `zai-org/GLM-5.3-BF16` (`1.507 TB`, 282 shards, 59,585 tensors, SHA `abf42e6c0e53bc29a875511301e7b7fcd220315b`):**
   - Glanced in `5,289.0 KB` (`0.00035%` of checkpoint), identifying `3` dense + `75` sigmoid `noaux_tc` MoE layers (`8/256` routed + `1` shared) and `56.89x` `MLA` KV compression (`576` latent floats vs `32,768` naive MHA floats per token-layer).
   - Executed full 64-head Layer 0 `MLA` Matrix Absorption verification over HTTPS `206` Range touching `1/282` shards (`0.104%` of checkpoint), achieving `0.0` binary32 `max_abs_diff` between materialized 64-head KV attention and latent-absorbed `(c_kv, k_pe)` attention.
3. **Unified Continuous Operation (`vld continuous`):**
   - Verified end-to-end across a 6-stage pipeline (`IPv4 Router + HTTP Path` $\to$ `HeadlessOS POSIX CoW + Pipe` $\to$ `Signal8 Congruence + F_97 Commutant Orbit` $\to$ `WASM 1.0 + Tiled E-GPU` $\to$ `4-Worker mTLS Tensor-Cut Fabric` $\to$ `DeepSeek-V4.1 + GLM-5.3`), surviving mid-pipeline host replacement and real `os._exit(73)` child process destruction with zero stage replay while `ContinuousLayerBuilderAgent` upgrades reduce Pass 2 cost from `(L=9.2, E=12012, B=11112, R=78)` to `(L=3.8, E=4332, B=2116, R=78)` (`Pareto dominance = True`).

---

## 6. Phase 9 Peer-Audit Hardening, Fluid In-Transit Substrate & Live 7-Model Arbitration Record (`v0.6.0`, 2026-11-18)

1. **Peer Read-Only Examination Findings Resolved & Regression-Tested:**
   - Fixed `.github/workflows/ci.yml` unquoted YAML colon (`lambda _: None`) via block scalar `run: |` and added automated workflow YAML validation in `tests/test_phase9_fluid_and_arbitration.py`.
   - Enforced atomic `0600` private-key file creation (`os.open(..., 0o600)`) and `0700` PKI directory permissions in `vld.fabric.mtls.generate_ephemeral_mtls_pki`.
   - Enforced 7-invariant inductive genesis-input & stage-spec recovery binding, mandatory independent `verify_fn` contracts, and separated empirical wall-clock/M1 byte telemetry (`MeasuredStageTelemetry`) in `vld.continuous.ContinuousVirtualSystem`, paired with standalone zero-import verification (`vld.independent_audit`).
2. **PyTorch `transformers` (`4.51.3`) FP32 Reference Parity Across Multi-Family Realizers:**
   - Verified `TransformerLayerRealizer` directly against `LlamaForCausalLM` (`max_abs_logit_diff = 1.2e-7`), `Qwen2ForCausalLM` (`6.0e-8`), and `Glm4ForCausalLM` (`1.0e-7`), alongside previously verified `Qwen3ForCausalLM` (`7.0e-6`), `DeepSeek-V4.1`, and `GLM-5.3`.
3. **Live 7-Model Recursive Arbitration Campaign Across `Virtual-Layer-Dynamics` $\leftrightarrow$ `In-Formation-Space` (`MVA-001`):**
   - Executed end-to-end 30-layer greedy decode on live `HuggingFaceTB/SmolLM2-135M-Instruct` (`5` coalesced HTTP `206` spans, `ckpt = d6ae996b...`), 28-layer local greedy decode on `Qwen/Qwen3-0.6B` (`ckpt = cd3b3540...`), live HTTP `206` Layer 0 execution on `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (`2.635%` of shard) and `THUDM/GLM-Z1-9B-0414` (`2.171%` of checkpoint), and live structural glances on `DeepSeek-V4.1-Flash` (`510.29 GB`), `GLM-5.3-BF16` (`1.507 TB`), and `Qwen3-30B-A3B` (`61.06 GB`), arbitrated by nested Depth-1 $\to$ Depth-2 `InLayerMetaArbitratorModel`s (`final_continuity_digest = b3437cfdbd8407e7...`, standalone audit `VERIFIED`).
