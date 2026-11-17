# Virtual Layer Dynamics

**A consolidated computational state-space architecture for lifting hardware, software, communication paths, and neural models into navigable, verified Virtual Grounds.**

---

## Vision & Core Question

Can communication paths, sensors, queues, services, and hosts—anything capable of receiving, transforming, and communicating information—form a connected computational fabric that operates virtual machines, processes, and dynamical systems *more effectively* than dedicated physical machines?

**Virtual Layer Dynamics (`vld`)** is the consolidated, production-grade implementation unifying four predecessor research programmes:

1. **E-CPU (`mailcpu`):** Computational Grounds, Structural Memory, receiver-owned Hypervisor authority, content-addressed M1 byte objects, authenticated vantage infrastructure, and neural weight/layer Grounds.
2. **Virtual Ground Interpreter (`vgi`):** Signal-valued cyclic phasor execution ($\mathbb{Z}/8\mathbb{Z}$), multi-domain observer-covariant backends, three-valued Kleene $\mathcal{K}_3$ guards, multi-digit predictive transition compilation, active perception, and `ProofGround` hypergraphs.
3. **E-CPU Virtual Processor (`vp`):** Persistent processor identity and hash-chained journals across disposable hosts, dual-derivation affine orbit navigation, and sparse Mixture-of-Experts (`Qwen3-30B-A3B` / `Qwen3-235B-A22B`) architectural routing frontiers.
4. **In-Formation-Space (`ifs`):** Portable information-flow envelopes, bounded Safetensors header/range observers (`HFR`), constructive semantic-to-neural hinge & FSM compilation (`SNC`), procedural Net2Wider neural formation (`PNF`), and capability-bounded model agents.

---

## Governing Mathematical Shape

### 1. Lifting Bytes into a Virtual Ground
Any hardware or software system representable in finite bytes $x \in (\mathbb{Z}/256\mathbb{Z})^N$—CPU/GPU state, OS pages, network datagrams, sensor streams, or neural weights—lifts through a declared, round-trip verified coordinate lens $\Lambda: (\mathbb{Z}/256\mathbb{Z})^N \rightharpoonup \Sigma_{\mathcal{G}}$ into a versioned **Virtual Ground**:

$$\mathcal{G} = (\mathcal{L}, \Sigma_{\mathcal{G}}, \Delta_{\mathcal{G}}, \mathcal{I}_{\mathcal{G}}, \mathcal{O}_{\mathcal{G}}, \mathcal{C}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}}, \nu_{\mathcal{G}})$$

### 2. Expressive Capacity vs. Predictive Quotient & Information Triad
Relative to an observer projection $P \in \mathcal{O}_{\mathcal{G}}$ and context $c$, states collapse under behavioral/predictive equivalence (Moore / Myhill-Nerode quotient):

$$s_1 \sim_{P,c} s_2 \iff \forall \pi \in \Delta_{\mathcal{G}}^{*},\; P(\pi(s_1), c) = P(\pi(s_2), c), \qquad \widetilde{\Sigma}_{P,c} = \Sigma_{\mathcal{G}} / {\sim_{P,c}}$$

### 3. Changing the Shape of Computation (Pareto Dominance Over Brute-Force Compute)
For an $n$-step trajectory or $M \times N$ operator evaluation, Virtual Layer Dynamics shifts asymptotic complexity whenever structural laws hold:
- **Sufficient-Statistic Operator Contraction (`vld.accel.StructuralLinearOperator`):** Contracts generative/structured matrices $W = W_{\text{bilinear}} + W_{\text{orbit}} + U V^\top + \Delta_{\text{sparse}}$ directly against input sufficient statistics in $O(M + N + M P + R(M + N) + |\operatorname{supp}(\Delta)|) \ll \Theta(MN)$ FLOPs and zero dense allocation.
- **Monoid & Semigroup Horizon Leaping (`AffineOrbitLaw`, `LinearRecurrenceGround`, `lfsr_jump_gf2`):** Advances scalar, $\operatorname{GF}(2)$, and contractive vector recurrences across $T$ coordinates in $O(\log T)$ binary compositions with zero intermediate state expansion.
- **Active-Expert-Only Sparse MoE & Range Vantage Execution (`vld.models`, `vld.fabric`):** Evaluates top-$k$ MoE layers ($k \ll E$) and remote `.safetensors` checkpoints via bounded M1 / HTTP `206 Partial Content` `Range` slices without materializing inactive experts.

---

## Documentation Map

- [**Mathematical Foundations**](docs/MATHEMATICAL_FOUNDATIONS.md) — Formal equations governing byte lifting, Virtual Grounds, Kleene $\mathcal{K}_3$ guards, multi-rung law certification, operator contraction, and neural/procedural model Grounds.
- [**Research State & Lineage**](docs/RESEARCH_STATE_AND_LINEAGE.md) — Audited status of all four predecessor codebases, verified capabilities, empirical baseline findings, and strict epistemic boundaries.
- [**Master Development Plan**](docs/DEVELOPMENT_PLAN.md) — Phased architecture and gate-driven engineering roadmap (`Phase 0` through `Phase 7`).

---

## Consolidated Repository Layout (`v0.2.0`)

```text
Virtual-Layer-Dynamics/
├── pyproject.toml                  # Zero-mandatory-dependency core (`virtual-layer-dynamics` / `vld`)
├── crates/vld_native/              # Optional LLVM-compiled Rust C-ABI accelerator kernels (`cdylib`)
├── docs/
│   ├── MATHEMATICAL_FOUNDATIONS.md
│   ├── RESEARCH_STATE_AND_LINEAGE.md
│   └── DEVELOPMENT_PLAN.md
├── src/vld/
│   ├── canonical.py                # Deterministic JSON & domain-separated SHA-256 addressing
│   ├── triad.py                    # Kleene K_3 guards, Myhill-Nerode quotient & erasure entropy
│   ├── ground.py                   # VirtualGroundSpec, ByteLens, dual-derivation AffineOrbitLaw, Pareto cost
│   ├── journal.py                  # Multi-process hash-chained receipt journal & idempotency fencing
│   ├── signal/                     # Signal8, ModularRing & ExactRational backends, footprints, base-8 compiler
│   ├── memory/                     # M1 PagedObjectStore, 5-rung Law Certifier, 2-adic inverses, demotion
│   ├── inquiry/                    # Greedy, beam & exact minimax probe planners + HypothesisRevisionLedger
│   ├── accel/                      # StructuralLinearOperator, LinearRecurrenceGround, NumPy + Rust CDLL
│   ├── fabric/                     # Multi-framing VantageEnvelope codec, StreamPipe & HTTP Range Vantage
│   ├── models/                     # Safetensors HFR, BPE Tokenizer, Qwen3 dense/MoE, orbit recovery, SNC/PNF
│   ├── agents/                     # ProofGround AND/OR hypergraph, ReceiverHypervisor & InLayerModelAgent
│   └── cli.py                      # `vld` CLI (`info`, `verify`, `navigate`, `benchmark`, `serve`)
└── tests/                          # 50+ invariant, differential, crash-recovery & hygiene tests
```

---

## Quickstart & Verification

Requires **Python 3.11+**. The core `vld` runtime requires zero third-party dependencies, and automatically activates `numpy` SIMD/BLAS and compiled `rustc` native C-ABI kernels when available on the host.

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
python -m vld verify
python -m vld benchmark --dim 1024 --steps 25000
python -m vld navigate --seed 73 --coordinate 1000000000000
python -m vld serve --host 127.0.0.1 --port 8730
```

---

## License

MIT License — Copyright (c) 2026 Kier73. See [LICENSE](LICENSE).
