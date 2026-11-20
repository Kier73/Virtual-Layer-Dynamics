# TRACE-002 — Active multi-observer inquiry and proof-carrying law acquisition

**Research status:** executable finite-field experiment, independent transcript audit, optional declared-operator proof. No claim of deducing the true external system from finite samples.

## Question

TRACE-001 showed that *given* enough observations, a common linear recurrence can become a logarithmic-time navigable Ground. TRACE-002 asks whether the Virtual Layer can **choose the next observation** based on mathematically guaranteed information gain, rather than request uniformly sampled history.

For a fixed candidate recurrence order d over a declared prime field, each observed scalar trajectory satisfies

`y(t) = c1*y(t-1) + ... + cd*y(t-d) (mod p)`.

The next regressor row `r_t = [y(t-1),...,y(t-d)]` is known **before** acquiring `y(t)`. Its increase in the coefficient matrix rank is therefore **exact and independent of the outcome**.

The active planner requests the cheapest affordable available rank-increasing observer measurement. If none exists, it can make an explicitly named **bridge** observation (rank gain zero now; potential access to novel future rows, not guaranteed). Once the law coefficients are uniquely identified, the system acquires at least two **new held-out observations from each observer** and refuses the law on the first counterexample.

This is a *greedy one-step policy*, not a proof of global minimum query cost. The fixed recurrence order and synchronized causal meaning of the input sequences are declared experimental assumptions.

## Running implementation

- `src/vld/inquiry/trace_inquiry.py`: exact finite-field rank acquisition and one-step cost-aware selection, explicit budget, bounded observations, nonce-like predecessor hash to reject stale proposals, refutation retention, status-based validation gate, optional pair of inquiry/navigation receipts.
- `src/vld/independent_trace_inquiry_audit.py`: zero-VLD-import independent replay of the **entire selection path**; checks exact matrix ranks, deterministic selection priorities, cost accounting, observed values, predecessor/digest chaining, model identifiability and withheld-data gates.
- `src/vld/navigation/trace_law.py` and `src/vld/independent_trace_audit.py` from TRACE-001 provide the already implemented, independently audited distant-coordinate projection.
- `examples/trace_inquiry_demo.py` and `tests/test_trace_inquiry.py` exercise a complete operator observation → hypothesis → held-out check → distant-state navigation path.

Commands:

~~~bash
python -m pytest -v tests/test_trace_inquiry.py tests/test_trace_law.py
python examples/trace_inquiry_demo.py
~~~

## Experiment: two individually inadequate observers

Let the **declared** dynamical operator be

`A = diag(1,2) mod 101`, initial state `s0=(1,1)`.

Observer A reads the first component: `y_A(t)=1`. Observer B reads the second: `y_B(t)=2^t mod 101`. No amount of additional measurements from **either one alone** can uniquely identify the shared two-coefficient recurrence—each geometric channel supplies only rank-one regressors.

The two initial two-sample prefixes are

- Observer A: `[1,1]`
- Observer B: `[1,2]`.

The active planner chooses one additional rank-increasing row from each channel, identifying the shared order-two law in exactly **two further observations**:

`y(t+2) = 3*y(t+1) - 2*y(t) mod 101`.

The implementation then requests **two more observations per channel** *after* the law first becomes uniquely identified, for a total of six newly acquired observations (two identification, four held-out). The rank-gain planner never needs to see the outcome before deciding whether its regressor is independent.

Under the declared operator, the pair of distant observer states at `10^100+17` is `[1,75]`, without replaying the intervening sequence. The navigation certificate is checked by TRACE-001's independent companion-matrix method.

A deliberate negative control acquires many additional observations from only the constant observer and **never attains rank two**. Other controls include cost/budget exhaustion, stale-probe refusal, inaccurate predicted held-out results retained as refuting evidence, forged greedy selections with recomputed hashes, and distinct-eigenvalue systems of order 2 or 3.

## What the evidence actually establishes

Three different questions must not be conflated:

1. **Was the query selected according to the promised rank/cost policy?** The independent inquiry verifier replays the entire query plan, selected observer, input, field rank and budget accounting.
2. **Do the observations support the inferred law, and is its distant-state arithmetic valid?** The transcript contains full observed prefixes and held-out checks; TRACE-001's separate verifier checks the resulting recurrence's extrapolation arithmetic.
3. **Is the declared operator the authentic external environment?** **Not established.** An evidence reference is plain attribution metadata, not a signature. Even held-out success cannot prove universal future behavior without an authenticated transition-ground witness.

TRACE-002's strongest outcome is consequently **a sound experimental selection policy under fixed order and declared observation semantics**, not an unrestricted theorem about unknown nonlinear systems.

## Toward distributed use

The existing VLD active-perception planners already support partial finite-state cell resolution, counterfactual scenarios, and a HypothesisRevisionLedger. TRACE-002 is an algebra-specific inquiry surface that reuses the recurrence solver and navigable Ground established in TRACE-001 rather than reimplementing a general transition planner.

Next gates: bind observer evidence refs to signed/verified transport receipts; select probes across heterogeneous observer clocks with temporal uncertainty; reason over multiple candidate recurrence orders and nonlinear regimes without premature promotion; and allow the IntentionToRealizationCompiler to ingest only qualified externally attributed laws.

**Key implication:** adding a complementary observer can increase a system's *identifiability* even when adding more computation or more samples from existing observers cannot.
