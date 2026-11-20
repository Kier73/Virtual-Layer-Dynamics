# TRACE-001 — Observation-derived laws as navigable computational Grounds

**Status:** experimental implementation with independently auditable arithmetic; not proof that a finite trace determines an unknown future.

**Question:** Can multiple observers jointly identify a dynamical relation that none can identify individually, then navigate an immense coordinate without replaying the trajectory?

## Implementation

1. **Discovery:** The new Python module at **src/vld/navigation/trace_law.py** synthesizes a common linear recurrence over a prime field from aligned observation channels. Every candidate order is solved with exact finite-field elimination; only uniquely identified coefficients are eligible. At least two held-out frames per observer are required. Contradictions and underidentification are distinct errors.
2. **Navigation:** A degree-d discovered law evaluates the state at coordinate t using binary exponentiation in the polynomial quotient ring Fp[x]/(x^d - c1 x^(d-1) - ... - cd). It uses O(d² log t) field arithmetic and never expands the intermediate trajectory.
3. **Independent audit:** **src/vld/independent_trace_audit.py** uses no VLD imports. It independently checks every observed recurrence constraint and recomputes the far coordinate by companion-matrix exponentiation. It verifies canonical residues, prime modulus, field/dimension constraints, exact epistemic scope, source traces, and domain-separated content digest.
4. **Optional global operator proof:** A declared matrix A, initial state s0, and observer rows h_i allow the independent checker to validate every observed frame, and the universal identity:

   h_i A^d = sum(j=1..d) c_j h_i A^(d-j) (mod p).

   Consequently the recurrence holds for **every state of the declared operator**, not just the sample. The proof is conditional on that operator actually describing the external system; a digest alone is not an authority credential.

## Measured local experiment

Observe A = [[1,1],[1,0]] with initial state [1,0] over the prime field F_65521, projecting both state coordinates separately. From only **three training frames per observer**, the shared order-two recurrence with coefficients [1,1] becomes uniquely identifiable. A single observer at that training length is still underdetermined.

There are 13 observations per channel, leaving **10 held-out frames** after training. At target coordinate **10^60 + 17**:

| Measurement | Local result |
|---|---|
| Learned coefficients | [1,1] modulo 65521 |
| Observer-1 projected value | 17164 |
| Observer-2 projected value | 51927 |
| Quotient-polynomial products | 262 per channel; not scalar FLOPs |
| Production and independent audit | ~1.48 ms combined, environment-specific |
| Separate matrix audit | VERIFIED_CONDITIONAL_ON_DECLARED_OPERATOR |
| Intermediate target states materialized | Zero (the auditor separately checks 13 witnessed frames) |

Reproduce with:

~~~bash
python -m pytest -q tests/test_trace_law.py
python examples/trace_law_demo.py
~~~

The isolated local test harness passed six pytest cases, including 15 deterministic random finite-field linear systems of dimension 2–4, each checked at seven target coordinates, plus several forged-receipt, contradictory-operator, and overfitted-suffix negative controls. Full-repository CI is a separate check.

## Epistemic distinction: observational fit is not global truth

A process can agree with every observed frame and change behavior on the very next tick. Thus the two receipt scopes are deliberately different:

- **OBSERVATIONAL_HYPOTHESIS:** finite evidence supports a recurrence; independent audit proves the arithmetic of its conditional projection, **not** future external behavior.
- **DECLARED_OPERATOR_CONDITIONAL:** an operator-ground witness is mathematically self-consistent and universally entails the recurrence. Binding that operator to the external system still requires independently attributable evidence.

Neither implies general neural inference nor independence from physical execution. The result establishes a specific bridge between information gained from multiple observers and cheaper addressability of remote states.

## Research frontier

TRACE-001 is a small runnable portion of **Discover_t -> Compile_t**. Next scientific gates: active counterexample-guided observer/time probing; authenticated operator-ground binding across host replacement; nonlinear and neural residual dynamics with interval enclosures; and qualified integration with the existing IntentionToRealizationCompiler.

**Negative control:** never promote a finite predictive fit into an unconditional global future-law certificate.
