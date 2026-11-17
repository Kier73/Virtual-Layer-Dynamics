# Mathematical Foundations of Virtual Layer Dynamics

This document states the formal structures and proof obligations governing **Virtual Layer Dynamics (`vld`)**. Equations here serve two roles:
1. **Executable invariants** directly enforced in `src/vld/` and verified by `tests/`.
2. **Architectural contracts** constraining subsequent consolidation phases in [`DEVELOPMENT_PLAN.md`](DEVELOPMENT_PLAN.md).

Every mathematical expression carries explicit domain assumptions; a digest or formula never substitutes for an empirical measurement or an independent semantic check.

---

## 1. Byte Addressability and Coordinate Lenses

Any finite digital artifact—instruction image, process memory page, network packet, sensor buffer, or neural tensor shard—is natively a point in the product ring:

$$x \in (\mathbb{Z}/256\mathbb{Z})^N, \qquad N \in \mathbb{N}_0$$

Raw bytes carry exact content identity $h_{\text{raw}}(x) = \operatorname{SHA256}(x)$ and length $N$, without assuming semantic structure. Richer interpretation requires an explicit **coordinate lens**:

$$\Lambda = (\Sigma_\Lambda, \operatorname{lift}: \mathcal{D}_\Lambda \to \Sigma_\Lambda, \operatorname{lower}: \Sigma_\Lambda \to \mathcal{D}_\Lambda), \qquad \mathcal{D}_\Lambda \subseteq (\mathbb{Z}/256\mathbb{Z})^*$$

For invertible representations (`RAW_Z256`, fixed-width `UINT_LE`/`UINT_BE`, canonical `UTF8_EXACT`), `ByteLens.lift(x)` enforces exact round-trip conservation before issuing a receipt:

$$\forall x \in \mathcal{D}_\Lambda,\quad \operatorname{lower}(\operatorname{lift}(x)) = x$$

Multiple distinct lenses $\Lambda_1, \Lambda_2$ may interpret the same byte sequence $x$ into non-isomorphic Grounds (e.g., IEEE/BF16 bit patterns viewed under Structural Memory byte laws vs. floating-point linear algebra). VLD binds `(source_sha256, byte_length, lens_kind)` separately from downstream operators so representation never silently confers execution authority.

---

## 2. Computational Grounds and Constrained State Spaces

A **Virtual Ground** is a versioned eight-tuple:

$$\mathcal{G} = (\mathcal{L}, \Sigma_{\mathcal{G}}, \Delta_{\mathcal{G}}, \mathcal{I}_{\mathcal{G}}, \mathcal{O}_{\mathcal{G}}, \mathcal{C}_{\mathcal{G}}, \mathcal{E}_{\mathcal{G}}, \nu_{\mathcal{G}})$$

- $\mathcal{L}$: typed description grammar and operator signatures.
- $\Sigma_{\mathcal{G}}$: state space denoted by well-formed descriptions (potentially exponential or countably infinite while $\mathcal{G}$ remains finite).
- $\Delta_{\mathcal{G}} \subseteq \Sigma_{\mathcal{G}} \times A_{\mathcal{G}} \times \Sigma_{\mathcal{G}}$: admitted transition relation over typed action alphabet $A_{\mathcal{G}}$.
- $\mathcal{I}_{\mathcal{G}}: \Sigma_{\mathcal{G}} \to \{\text{true}, \text{false}\}$: state and domain invariants.
- $\mathcal{O}_{\mathcal{G}}$: family of bounded observer projections $P: \Sigma_{\mathcal{G}} \times C \to Y$.
- $\mathcal{C}_{\mathcal{G}}$: vector-valued transition, communication, and verification cost model.
- $\mathcal{E}_{\mathcal{G}}$: supporting witness certificates, tolerances, and counterexample boundaries.
- $\nu_{\mathcal{G}}$: schema version bound into the domain-separated Ground fingerprint:
  $$\operatorname{fp}(\mathcal{G}) = \operatorname{SHA256}(\text{"vld.ground-spec.v1"} \,\|\, \text{0x00} \,\|\, \operatorname{canonical\_bytes}(\mathcal{G}))$$

Given input $x$ and context $c$, the constrained admissible subspace is:

$$\Sigma_{\mathcal{G} \mid x, c} = \{ s \in \Sigma_{\mathcal{G}} \mid D_{x,c}(s) \land \mathcal{I}_{\mathcal{G}}(s) \}$$

---

## 3. Expressive Capacity, Predictive Quotients, and the Information Triad

Virtual Layer Dynamics navigates a fundamental duality between **state generation capacity** and **predictive compression**:

1. **Generative State Capacity:** Maximize $|\Sigma_{\mathcal{G}}|$ (or reachable horizon $|\Sigma_{\text{reach}}|$) addressable per byte of description $|\mathcal{G}|$, avoiding eager enumeration of $\Sigma_{\mathcal{G}}$.
2. **Predictive Equivalence Quotient:** Relative to an observer $P \in \mathcal{O}_{\mathcal{G}}$, two states that yield identical observations across all admissible action sequences $\pi \in A_{\mathcal{G}}^*$ carry no actionable distinction:
   $$s_1 \sim_P s_2 \iff \forall \pi \in A_{\mathcal{G}}^*,\quad P(\Delta_{\mathcal{G}}^*(\pi, s_1)) = P(\Delta_{\mathcal{G}}^*(\pi, s_2))$$
   Partition refinement computes the minimal quotient $\widetilde{\Sigma}_P = \Sigma_{\mathcal{G}} / {\sim_P}$, with predictive compression factor $\rho_P = |\widetilde{\Sigma}_P| / |\Sigma_{\mathcal{G}}| \in (0, 1]$.

### Transition Predictability & Information Classification

For each state-action pair $(s, a) \in \Sigma_{\mathcal{G}} \times A_{\mathcal{G}}$, let $T(s, a) = \{s' \mid (s, a, s') \in \Delta_{\mathcal{G}}\}$:

| Class | Defining Condition | Information Dynamics & Operational Role |
|---|---|---|
| `PREDICTABLE_PRESERVING` | $|T(s, a)| = 1$ (un-coupled) and $|\Delta_a^{-1}(s')| = 1$ | Locally invertible step; zero collision erasure ($H_{\text{erasure}} = 0$ when total). Supports group/orbit shortcuts and exact undo. |
| `PREDICTABLE_CONTRACTING` | $|T(s, a)| = 1$ (un-coupled) and $|\Delta_a^{-1}(s')| > 1$ | Deterministic many-to-one map; removes historical state bits: $H_{\text{erasure}}(a) = \sum_{s'} \frac{|\delta_a^{-1}(s')|}{|\Sigma|} \log_2 |\delta_a^{-1}(s')| > 0$. Justifies sufficient-state collapse and smaller checkpoints. |
| `INFORMATION_PRODUCING` | $(s, a) \in \mathcal{K}_{\text{ext}}$, $I(S'; \text{Env} \mid s, a) > 0$ | Active sensor, user, or network encounter introducing external entropy; requires observation receipts and active inquiry policies. |
| `AMBIGUOUS` | $|T(s, a)| > 1$, $(s, a) \notin \mathcal{K}_{\text{ext}}$ | Competing uncalibrated hypotheses; lowering to deterministic code is refused. |
| `UNKNOWN` | $|T(s, a)| = 0$ | Unwitnessed cell; triggers refusal or exploratory probing rather than extrapolation. |

---

## 4. Signal Phasor Grounds and Three-Valued Uncertainty ($\mathcal{K}_3$)

In signal-valued Grounds (such as cyclic phase domain $\mathbb{Z}/M\mathbb{Z}$, realized for $M = 8$ in VGI), semantic symbol $k \in \mathbb{Z}/M\mathbb{Z}$ under unit observer $u \in (\mathbb{Z}/M\mathbb{Z})^\times$ is encoded as the character:

$$\chi_u(k) = \exp\!\left(\frac{2\pi i u k}{M}\right), \qquad S = (z, u, \epsilon), \quad |z| = 1,\; \epsilon \in [0, \pi]$$

Observer coordinate reframing $u \to v$ raises the phasor to $m = v u^{-1} \bmod M$:

$$R_{u \to v}(z, u, \epsilon) = \left(\frac{z^m}{|z^m|},\; v,\; \min(\pi, m\epsilon)\right), \qquad R_{u \to v}(\chi_u(k)) = \chi_v(k)$$

Group addition and subtraction correspond to aligned phasor multiplication and conjugation:

$$\chi_u(a + b) = \chi_u(a)\chi_u(b), \qquad \chi_u(a - b) = \chi_u(a)\overline{\chi_u(b)}$$

### Conservative Interval Guards (Kleene $\mathcal{K}_3$)

Given phase separation $d = |\arg(z_a \overline{z_b})| \in [0, \pi]$ and combined phase-error bound $e = \min(\pi, \epsilon_a + \epsilon_b)$, the true angular separation lies in $[L, U] = [\max(0, d - e), \min(\pi, d + e)]$. Against threshold $\tau$:

$$\operatorname{Guard}_\tau(d, e) = \begin{cases}
\text{TRUE} & \text{if } U \le \tau, \\
\text{FALSE} & \text{if } L > \tau, \\
\text{INDETERMINATE} & \text{if } L \le \tau < U.
\end{cases}$$

`TruthDisposition` forms a bounded distributive Kleene algebra $(\{\text{FALSE}, \text{INDETERMINATE}, \text{TRUE}\}, \land, \lor, \lnot)$. Any branch instruction encountering `INDETERMINATE` raises `IndeterminateGuardTrap`, leaving machine state unmodified.

---

## 5. Algebraic Shortcuts, Dual Derivation, and Identifiability

### Dual-Derivation Affine Navigation
For modular affine dynamics $x_{t+1} = (a x_t + b) \bmod m$ ($m \ge 2$), advancing $n$ coordinates via step-by-step iteration costs $\Theta(n)$, whereas binary composition in the affine transformation monoid costs $O(\log n)$:

$$(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1 \bmod m,\; (a_2 b_1 + b_2) \bmod m)$$

To guard against implementation defects, `AffineOrbitLaw.navigate` independently evaluates homogeneous $2 \times 2$ matrix exponentiation over $(\mathbb{Z}/m\mathbb{Z})^{2 \times 2}$:

$$\begin{pmatrix} x_n \\ 1 \end{pmatrix} \equiv \begin{pmatrix} a & b \\ 0 & 1 \end{pmatrix}^n \begin{pmatrix} x_0 \\ 1 \end{pmatrix} \pmod m$$

and admits the result iff both derivations agree identically.

### Law Identifiability Boundary
Fitting a formula to finite observations does **not** automatically identify a Ground law. Specifically, for an observed additive sequence $x_0, x_1, \dots, x_{k-1}$ with constant positive step $x_{t+1} - x_t = \delta > 0$ and no witnessed wraparound:

$$\forall m > \max_{0 \le t < k} x_t,\quad x_{t+1} \equiv x_t + \delta \pmod m$$

Because infinitely many moduli (plus unbounded $\mathbb{Z}$) generate identical prefixes, `propose_modular_progression` classifies unwrapped traces as `EpistemicStatus.HYPOTHESIS` and `certify_modular_progression` refuses certification unless either:
1. Both a positive unwrapped step $\delta > 0$ and a modular wrap jump $x_{t^* + 1} - x_{t^*} = \delta - m < 0$ are directly witnessed across $N \ge 4$ samples (providing $df = (N - 1) - 2 = N - 3 \ge 1$ predictive degree of freedom beyond fitting $(\delta, m)$, preventing 3-sample overfitting), or
2. The domain modulus $m$ is explicitly declared by an upstream Ground contract ($df = (N - 1) - 1 = N - 2 \ge 1$ for $N \ge 3$).

---

## 6. Observer-Host Paths, Pareto Cost Vectors, and Durable Lineage

### Path Composition
A logical **host** is any identifiable locus $h = (I_h, C_h, S_h, T_h, K_h, L_h)$ possessing a declared observation or transition relation $T_h$ and local capability policy $K_h$. Over an admissible path $\pi = (h_0, h_1, \dots, h_m)$ with matching boundary contracts, the composite operation is:

$$F_\pi = T_{h_m} \circ T_{h_{m-1}} \circ \dots \circ T_{h_0}$$

Authority remains strictly receiver-local: routing reachability never grants execution permission, and transport acknowledgement never implies semantic state commit.

### Multi-Objective Cost Dominance
Each candidate realization carries a non-negative vector $\mathbf{C} = (L, E, B, R) \in \mathbb{R}_{\ge 0}^4$ (latency, compute operations, byte movement, verification/recovery cost). Strategy $\pi_1$ **Pareto-dominates** $\pi_2$ ($\mathbf{C}(\pi_1) \prec \mathbf{C}(\pi_2)$) iff:

$$\forall i \in \{1..4\},\; C_i(\pi_1) \le C_i(\pi_2) \quad \land \quad \exists j \in \{1..4\},\; C_j(\pi_1) < C_j(\pi_2)$$

For a guarded Virtual Layer shortcut $(\sigma, g)$ with pocket-exit probability $p_{\text{exit}} \in [0, 1]$, expected cost is:

$$\mathbf{C}_{\text{VLD}}(\sigma, g) = \mathbf{C}_{\text{lift}} + \mathbf{C}_g + \mathbf{C}_\sigma + p_{\text{exit}} \cdot \mathbf{C}_{\text{fallback}}$$

### Hash-Chained Journal & Quorum Safety
Durable processor/session receipts form a domain-separated hash chain anchored at $h_{-1} = 0^{64}$:

$$h_i = H_{\text{vld.journal-event.v1}}(e_i \setminus \{\text{event\_digest}\}), \qquad e_i.\text{prev\_digest} = h_{i-1}$$

Across distributed recovery authorities of fixed membership $N$, conflicting lineage roots in the same `(term, generation)` slot are prevented by strict majority quorums $2q > N \implies \forall Q_1, Q_2,\; |Q_1 \cap Q_2| \ge 1$.

---

## 7. Neural, Procedural, and Agentic Model Grounds

Virtual Layer Dynamics treats neural networks and Large Language Models as structured mathematical Grounds rather than monolithic device-bound binaries:

1. **Content-Addressed Parameter Manifolds:** An architecture $A$ defines parameter space $\mathcal{W}_A = \prod_{i=1}^M D_i^{s_i}$. A model snapshot $(A, \theta)$ has storage-independent identity derived from ordered per-tensor content digests, enabling lazy slice/expert materialization without loading full checkpoints.
2. **Sparse Mixture-of-Experts Frontiers:** Given router logits $\ell \in \mathbb{R}^E$ at layer $l$, deterministic top-$k$ selection ($k \ll E$, e.g., $8$ of $128$ experts in Qwen3-235B-A22B) activates only $\mathcal{S}(\ell) \subset \{0, \dots, E-1\}$ with normalized gates:
   $$p_e = \frac{\exp(\ell_e - \max_j \ell_j)}{\sum_{j=0}^{E-1} \exp(\ell_j - \max_{j'} \ell_{j'})}, \qquad w_e = \frac{p_e}{\sum_{j \in \mathcal{S}(\ell)} p_j} \quad (e \in \mathcal{S}(\ell))$$
   compiling a bounded per-layer dependency DAG without instantiating inactive expert weights.
3. **Constructive & Procedural Weight Synthesis:**
   - *Piecewise-Affine Hinge Compilation:* Any continuous 1D piecewise-affine law with breakpoints $t_1 < \dots < t_K$, leftmost slope $m_0$, intercept $b$, and interval slopes $m_1, \dots, m_K$ compiles constructively (zero gradient descent) to an exact $(K + 2)$-neuron ReLU network:
     $$f(x) = b + m_0(\operatorname{ReLU}(x) - \operatorname{ReLU}(-x)) + \sum_{k=1}^K (m_k - m_{k-1})\operatorname{ReLU}(x - t_k)$$
   - *Function-Preserving Morphisms (Dyadic Net2Wider):* Duplicating hidden unit $j$ with incoming column $W_{:,j}, b_j$ and halving outgoing rows $V_{j,:} \mapsto \frac{1}{2} V_{j,:}$ preserves exact real-valued block semantics $\sigma(x W + b)V + c$ while expanding representational geometry.
   - *Generative Weight Recipes:* Tensors may be reconstructed on demand from certified Structural Memory laws, sparse bit deltas, or verified coordinate bijections, all checked against exact per-chunk SHA-256 commitments.
4. **Models as Capability-Bounded Agents & Advisory Hypervisors:** Models operating inside the Virtual Layer emit typed proposals (`propose_ground`, `select_probe`, `route_frontier`), while deterministic receiver Hypervisors, law certifiers, and independent verifiers enforce capability bounds, mathematical equivalence, and state commits.
