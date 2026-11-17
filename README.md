# Virtual Layer Dynamics

**A consolidated computational state-space architecture for lifting hardware, software, communication paths, and neural models into navigable, verified Virtual Grounds.**

---

## Vision & Core Question

Can communication paths, sensors, queues, services, and hosts—anything capable of receiving, transforming, and communicating information—form a connected computational fabric that operates virtual machines, processes, and dynamical systems *more effectively* than dedicated physical machines?

**Virtual Layer Dynamics (`vld`)** is the consolidated, production-grade implementation unifying four predecessor research programmes:

1. **E-CPU (`mailcpu`):** Computational Grounds, Structural Memory, receiver-owned Hypervisor authority, content-addressed byte objects & quorum-fenced sessions, authenticated multi-hop fabric, SIMT/tensor recovery, and Internet/path-native substrates.
2. **Virtual Ground Interpreter (`vgi`):** Signal-valued cyclic phasor execution ($\mathbb{Z}/8\mathbb{Z}$), observer-coordinate reframing, three-valued phase-uncertainty guards (`TRUE | FALSE | INDETERMINATE`), predictive transition compilation, active perception, and `ProofGround` obligations.
3. **E-CPU Virtual Processor (`vp`):** Persistent processor identity and hash-chained journals across disposable hosts, dual-derivation affine orbit navigation, and sparse Mixture-of-Experts (`Qwen3-30B-A3B` / `Qwen3-235B-A22B`) architectural routing frontiers.
4. **In-Formation-Space (`ifs`):** Portable encounter & temporal provenance contracts, blind cold-start handoffs, bounded external model structure readers, constructive semantic-to-neural hinge compilation, procedural Net2Wider neural formation, and capability-bounded model controllers.

---

## Governing Mathematical Shape

### 1. Lifting Bytes into a Virtual Ground
Any hardware or software system representable in finite bytes $x \in (\mathbb{Z}/256\mathbb{Z})^N$—CPU/GPU state, OS pages, network datagrams, sensor streams, or neural weights—lifts through a declared, verifiable coordinate lens $\Lambda: (\mathbb{Z}/256\mathbb{Z})^N \rightharpoonup \Sigma_{\mathcal{G}}$ into a versioned **Virtual Ground**:

$$\mathcal{G} = (\mathcal{L}, \Sigma_{\mathcal{G}}, \Delta_{\mathcal{G}}, \mathcal{I}_{\mathcal{G}}, \mathcal{O}_{\mathcal{G}}, \mathcal{C}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}}, \nu_{\mathcal{G}})$$

where $\mathcal{L}$ is the representation language, $\Sigma_{\mathcal{G}}$ the denoted state space, $\Delta_{\mathcal{G}}$ admitted transitions, $\mathcal{I}_{\mathcal{G}}$ invariants and admission predicates, $\mathcal{O}_{\mathcal{G}}$ observer projections, $\mathcal{C}_{\mathcal{G}}$ multi-objective cost vectors, $\mathcal{E}_{\mathcal{G}}$ verification evidence, and $\nu_{\mathcal{G}}$ immutable schema version.

### 2. Expressive Capacity vs. Predictive Quotient & Information Triad
Relative to an observer projection $P \in \mathcal{O}_{\mathcal{G}}$ and context $c$, states collapse under behavioral/predictive equivalence (Moore / Myhill-Nerode quotient):

$$s_1 \sim_{P,c} s_2 \iff \forall \pi \in \Delta_{\mathcal{G}}^{*},\; P(\pi(s_1), c) = P(\pi(s_2), c), \qquad \widetilde{\Sigma}_{P,c} = \Sigma_{\mathcal{G}} / {\sim_{P,c}}$$

Every transition $(s, a) \mapsto \Delta_{\mathcal{G}}(s, a)$ is classified into an explicit predictability and information-flow triad:

- **Predictable & Information-Preserving ($\Delta_{\text{iso}}$):** Deterministic and locally injective ($|\delta_a^{-1}(s')| = 1$, $H_{\text{erasure}} = 0$). Admits algebraic/group shortcuts ($O(1)$ or $O(\log n)$ orbit jumps) without stepping intermediate states.
- **Predictable & Information-Removing ($\Delta_{\text{contract}}$):** Deterministic many-to-one maps ($|\delta_a^{-1}(s')| > 1$, erasing $H(S \mid \delta_a(S)) > 0$ bits of history). Enables compact sufficient-state checkpoints, cache coalescing, and quotient state reduction.
- **Information-Producing ($\Delta_{\text{open}}$):** Sensor, channel, or multi-host encounters injecting external mutual information $I(S'; \text{Env} \mid S, A) > 0$, governed by active probe planning.
- **Underdetermined / Ambiguous ($\Delta_{\text{amb}}$):** Unobserved (`UNKNOWN`), conflicting (`AMBIGUOUS`), or boundary-uncertain (`INDETERMINATE`) transitions—which trap atomically or request disambiguating observations rather than fabricating state.

### 3. When the Virtual Layer Outperforms Physical Step Execution
For an $n$-step trajectory, let $\mathbf{C}_{\text{phys}}(n)$ be the multi-objective cost vector (latency, compute operations, bytes transferred, verification work) of step-by-step physical emulation, and let $(\sigma, g)$ be an admitted Virtual Layer shortcut with guard $g$ and certified-pocket exit rate $p_{\text{exit}}$:

$$\mathbf{C}_{\text{VLD}}(\sigma, g, n) = \mathbf{C}_{\text{lift}} + \mathbf{C}_g + \mathbf{C}_\sigma(n) + p_{\text{exit}} \cdot \mathbf{C}_{\text{fallback}}(n)$$

Whenever algebraic orbit laws ($\mathbf{C}_\sigma(n) = O(\log n)$), predictive state quotients ($|\widetilde{\Sigma}_{P,c}| \ll |\Sigma_{\mathcal{G}}|$), sparse expert frontiers ($k \ll E$), or in-transit observer-host compositions ($F_\pi = T_{h_m} \circ \dots \circ T_{h_0}$) apply, the Virtual Layer achieves strict Pareto dominance $\mathbf{C}_{\text{VLD}} \prec \mathbf{C}_{\text{phys}}$ while preserving exact or tolerance-certified observations.

---

## Documentation Map

- [**Mathematical Foundations**](docs/MATHEMATICAL_FOUNDATIONS.md) — Formal equations governing byte lifting, Virtual Grounds, Kleene $\mathcal{K}_3$ guards, predictive quotients, cost dominance, path composition, and neural/procedural model Grounds.
- [**Research State & Lineage**](docs/RESEARCH_STATE_AND_LINEAGE.md) — Audited status of all four predecessor codebases, verified capabilities, empirical baseline findings, and strict epistemic boundaries.
- [**Master Development Plan**](docs/DEVELOPMENT_PLAN.md) — Phased architecture and gate-driven engineering roadmap (`Phase 0` through `Phase 7`) from core mathematical kernel to distributed substrates and Virtual-Layer-native LLMs/hypervisors.

---

## Repository Layout (`Phase 0` Foundation)

```text
Virtual-Layer-Dynamics/
├── pyproject.toml                  # Zero-dependency core package (`virtual-layer-dynamics` / `vld`)
├── docs/
│   ├── MATHEMATICAL_FOUNDATIONS.md # Formal equations & invariants
│   ├── RESEARCH_STATE_AND_LINEAGE.md
│   └── DEVELOPMENT_PLAN.md         # Consolidated Phase 0–7 engineering roadmap
├── src/vld/
│   ├── canonical.py                # Deterministic JSON & domain-separated SHA-256 addressing
│   ├── triad.py                    # Kleene K_3 guards, Myhill-Nerode quotient & erasure entropy
│   ├── ground.py                   # VirtualGroundSpec, ByteLens, dual-derivation AffineOrbitLaw, Pareto cost
│   ├── journal.py                  # Atomic hash-chained receipt journal with idempotency fencing
│   └── cli.py                      # `vld` command-line verifier & navigator
└── tests/                          # Unit, edge-case, regression & documentation-integrity tests
```

---

## Quickstart & Verification

Requires **Python 3.11+**. The core `vld` runtime uses only the Python standard library.

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
python -m vld verify
python -m vld navigate --seed 73 --coordinate 1000000000000
```

---

## License

MIT License — Copyright (c) 2026 Kier73. See [LICENSE](LICENSE).
