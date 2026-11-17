r"""Phase 7 — Receiver-Owned Capability Hypervisor & Compare-and-Swap State Commit Kernel (`vld.agents.hypervisor`).

Mathematical contract:
Every model or external agent proposal $\pi$ must pass three independent gates at the state owner:
  1. **Capability Gate:** $\operatorname{cap}(\pi) \in K_{\text{agent}}$ and remaining budget $\ge \operatorname{cost}(\pi)$.
  2. **Verification Gate:** Deterministic certification (`StructuralMemoryRegistry`, `verify_compiled_capsule`,
     or `verify_widening_certificate`) independently validates the mathematical claim.
  3. **Compare-and-Swap Commit Gate:** $\operatorname{CAS}(d_{\text{expected}}, s')$ succeeds iff
     $d_{\text{expected}} = H(s_{\text{current}})$, advancing the epoch monotonically and recording
     both accepted and refused proposals in `HashChainedJournal`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.journal import HashChainedJournal
from vld.memory import LawCertificationRefused, StructuralMemoryError, StructuralMemoryRegistry
from vld.models.procedural import ProceduralModelError, verify_compiled_capsule, verify_multivariate_capsule


class HypervisorRefusalError(PermissionError):
    """Raised when the Receiver Hypervisor deterministically refuses an unauthorized, stale, or unverified proposal."""


class CASConflictError(HypervisorRefusalError):
    """Raised on compare-and-swap state-digest or epoch conflict."""


class AgentCapability(str, Enum):
    """Fine-grained capabilities granted to in-layer models and external controllers."""

    OBSERVE_TELEMETRY = "observe_telemetry"
    SELECT_PROBE = "select_probe"
    PROPOSE_LAW = "propose_law"
    NAVIGATE_ORBIT = "navigate_orbit"
    ADMIT_NEURAL_CAPSULE = "admit_neural_capsule"
    COMMIT_STATE = "commit_state"


@dataclass(frozen=True, slots=True)
class CapabilityGrant:
    """Immutable capability token scoping an agent's permitted Virtual Layer actions and probe budget."""

    agent_id: str
    capabilities: frozenset[AgentCapability]
    probe_budget: Fraction = Fraction(10)
    allowed_memory_prefix: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string.")
        if isinstance(self.probe_budget, (bool, float)) or Fraction(self.probe_budget) < 0:
            raise ValueError("probe_budget must be a non-negative exact Fraction or int.")
        object.__setattr__(self, "probe_budget", Fraction(self.probe_budget))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


class ReceiverHypervisor:
    """Deterministic receiver-local Hypervisor governing model-proposed laws, probes, capsules, and CAS commits."""

    def __init__(
        self,
        hypervisor_id: str = "vld.receiver.hypervisor",
        *,
        journal: HashChainedJournal | None = None,
        structural_memory: StructuralMemoryRegistry | None = None,
        initial_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.hypervisor_id = hypervisor_id
        self.journal = journal if journal is not None else HashChainedJournal()
        self.structural_memory = (
            structural_memory if structural_memory is not None else StructuralMemoryRegistry(self.journal)
        )
        self._grants: dict[str, CapabilityGrant] = {}
        self._spent_budget: dict[str, Fraction] = {}
        self._state: dict[str, Any] = parse_strict_json(canonical_json(dict(initial_state or {"epoch": 0})))
        self._epoch: int = 0
        self._admitted_capsules: dict[str, dict[str, Any]] = {}
        self._audit_seq = 0

    def _next_audit_id(self, tag: str) -> str:
        self._audit_seq += 1
        return f"hyp-{tag}-{self._audit_seq:06d}"

    @property
    def state(self) -> dict[str, Any]:
        return parse_strict_json(canonical_json(self._state))

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def state_digest(self) -> str:
        return domain_digest("vld.hypervisor-state.v1", {"epoch": self._epoch, "state": self._state})

    def register_agent(self, grant: CapabilityGrant) -> None:
        self._grants[grant.agent_id] = grant
        self._spent_budget.setdefault(grant.agent_id, Fraction(0))

    def remaining_budget(self, agent_id: str) -> Fraction:
        grant = self._require_grant(agent_id)
        return grant.probe_budget - self._spent_budget[agent_id]

    def _require_grant(self, agent_id: str) -> CapabilityGrant:
        grant = self._grants.get(agent_id)
        if grant is None:
            self._record_refusal(agent_id, "unknown_agent", {"agent_id": agent_id})
            raise HypervisorRefusalError(f"Unregistered agent {agent_id!r}.")
        return grant

    def _require_capability(self, agent_id: str, cap: AgentCapability) -> CapabilityGrant:
        grant = self._require_grant(agent_id)
        if cap not in grant.capabilities:
            self._record_refusal(agent_id, f"missing_capability:{cap.value}", {"required": cap.value})
            raise HypervisorRefusalError(f"Agent {agent_id!r} lacks capability {cap.value!r}.")
        return grant

    def _record_refusal(self, agent_id: str, reason: str, details: Mapping[str, Any]) -> None:
        self.journal.record(
            request_id=self._next_audit_id("refusal"),
            ground_fingerprint=self.state_digest,
            operation="hypervisor.refusal",
            payload={"agent_id": agent_id, "details": dict(details), "reason": reason},
            result_fn=lambda: {"admitted": False, "reason": reason},
        )

    def authorize_probe(self, agent_id: str, *, probe_label: str, cost: int | Fraction) -> Fraction:
        """Deduct `cost` from agent's probe budget if permitted, else deterministically refuse."""
        self._require_capability(agent_id, AgentCapability.SELECT_PROBE)
        if isinstance(cost, (bool, float)):
            self._record_refusal(agent_id, "invalid_probe_cost_type", {"probe_label": probe_label, "type": type(cost).__name__})
            raise HypervisorRefusalError("Probe cost must be an exact int or Fraction, not bool/float.")
        try:
            c = Fraction(cost)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            self._record_refusal(agent_id, "invalid_probe_cost", {"error": str(exc), "probe_label": probe_label})
            raise HypervisorRefusalError(f"Invalid probe cost: {exc}") from exc
        if c <= 0:
            self._record_refusal(agent_id, "non_positive_probe_cost", {"cost": str(c), "probe_label": probe_label})
            raise HypervisorRefusalError("Probe cost must be positive.")
        rem = self.remaining_budget(agent_id)
        if c > rem:
            self._record_refusal(agent_id, "probe_budget_exceeded", {"cost": str(c), "remaining": str(rem)})
            raise HypervisorRefusalError(f"Probe {probe_label!r} cost {c} exceeds remaining budget {rem}.")
        self._spent_budget[agent_id] += c
        new_rem = self.remaining_budget(agent_id)
        self.journal.record(
            request_id=self._next_audit_id("probe"),
            ground_fingerprint=self.state_digest,
            operation="hypervisor.authorize_probe",
            payload={"agent_id": agent_id, "cost": str(c), "probe_label": probe_label},
            result_fn=lambda: {"admitted": True, "remaining_budget": str(new_rem)},
        )
        return new_rem

    def propose_structural_law(
        self,
        agent_id: str,
        *,
        memory_id: str,
        samples: Sequence[int],
        domain: str = "Z",
        declared_modulus: int | None = None,
        allow_unbounded_poly: bool = False,
    ) -> dict[str, Any]:
        """Deterministically verify and admit an agent-proposed sequence law, or refuse if underdetermined."""
        grant = self._require_capability(agent_id, AgentCapability.PROPOSE_LAW)
        if grant.allowed_memory_prefix and not memory_id.startswith(grant.allowed_memory_prefix):
            self._record_refusal(agent_id, "memory_scope_violation", {"memory_id": memory_id})
            raise HypervisorRefusalError(
                f"memory_id {memory_id!r} outside agent scope prefix {grant.allowed_memory_prefix!r}."
            )
        try:
            spec, cert = self.structural_memory.certify_and_admit(
                memory_id,
                samples,
                domain=domain,
                declared_modulus=declared_modulus,
                allow_unbounded_poly=allow_unbounded_poly,
                context={"admitted_by": self.hypervisor_id, "proposer_agent": agent_id},
            )
        except (LawCertificationRefused, StructuralMemoryError, ValueError) as exc:
            self._record_refusal(agent_id, "law_certification_refused", {"error": str(exc), "memory_id": memory_id})
            raise HypervisorRefusalError(f"Proposed law refused certification: {exc}") from exc

        return {
            "certificate_digest": cert.digest,
            "family": spec.law.family,
            "memory_id": memory_id,
            "revision": spec.revision,
            "spec_digest": spec.digest,
        }

    def navigate_certified_orbit(
        self,
        agent_id: str,
        *,
        memory_id: str,
        coordinate: int,
    ) -> dict[str, Any]:
        """Navigate to distant coordinate `coordinate` using a certified Structural Memory Ground."""
        grant = self._require_capability(agent_id, AgentCapability.NAVIGATE_ORBIT)
        if grant.allowed_memory_prefix and not memory_id.startswith(grant.allowed_memory_prefix):
            self._record_refusal(agent_id, "memory_scope_violation", {"memory_id": memory_id})
            raise HypervisorRefusalError(
                f"memory_id {memory_id!r} outside agent scope prefix {grant.allowed_memory_prefix!r}."
            )
        try:
            val, receipt = self.structural_memory.read(memory_id, coordinate)
        except (StructuralMemoryError, LawCertificationRefused, ValueError) as exc:
            self._record_refusal(agent_id, "orbit_navigation_refused", {"coordinate": coordinate, "error": str(exc)})
            raise HypervisorRefusalError(f"Orbit navigation refused: {exc}") from exc
        return {
            "coordinate": coordinate,
            "expanded_intermediate_states": receipt.expanded_intermediate_states,
            "receipt_digest": receipt.receipt_digest,
            "spec_digest": receipt.spec_digest,
            "value": val,
        }

    def admit_neural_capsule(self, agent_id: str, capsule: Mapping[str, Any]) -> dict[str, Any]:
        """Independently verify whole-domain equivalence before admitting a compiled neural capsule."""
        self._require_capability(agent_id, AgentCapability.ADMIT_NEURAL_CAPSULE)
        try:
            if not isinstance(capsule, Mapping):
                raise ProceduralModelError("capsule must be a mapping.")
            if capsule.get("schema") == "vld.snc-multivariate-relu-capsule.v1":
                report = verify_multivariate_capsule(capsule)
            else:
                report = verify_compiled_capsule(capsule)
        except (ProceduralModelError, KeyError, TypeError, ValueError) as exc:
            self._record_refusal(agent_id, "neural_capsule_verification_failed", {"error": str(exc)})
            raise HypervisorRefusalError(f"Neural capsule failed independent verification: {exc}") from exc
        cdigest = str(report["capsule_digest"])
        self._admitted_capsules[cdigest] = parse_strict_json(canonical_json(dict(capsule)))
        self.journal.record(
            request_id=self._next_audit_id("capsule"),
            ground_fingerprint=self.state_digest,
            operation="hypervisor.admit_neural_capsule",
            payload={"agent_id": agent_id, "capsule_digest": cdigest},
            result_fn=lambda: {"admitted": True, "status": report["status"]},
        )
        return report

    def cas_commit(
        self,
        agent_id: str,
        *,
        expected_state_digest: str,
        next_state: Mapping[str, Any],
    ) -> str:
        """Atomically update receiver state iff `expected_state_digest == self.state_digest`."""
        self._require_capability(agent_id, AgentCapability.COMMIT_STATE)
        current = self.state_digest
        if expected_state_digest != current:
            self._record_refusal(
                agent_id,
                "cas_conflict",
                {"actual_digest": current, "expected_digest": expected_state_digest},
            )
            raise CASConflictError(
                f"CAS conflict: expected state_digest {expected_state_digest}, actual {current}."
            )
        clean_next = parse_strict_json(canonical_json(dict(next_state)))
        if not isinstance(clean_next, dict):
            raise HypervisorRefusalError("next_state must be a JSON object.")
        self._epoch += 1
        self._state = clean_next
        new_digest = self.state_digest
        self.journal.record(
            request_id=self._next_audit_id("cas"),
            ground_fingerprint=current,
            operation="hypervisor.cas_commit",
            payload={"agent_id": agent_id, "epoch": self._epoch, "prev_state_digest": current},
            result_fn=lambda: {"admitted": True, "new_state_digest": new_digest},
        )
        return new_digest


__all__ = [
    "AgentCapability",
    "CASConflictError",
    "CapabilityGrant",
    "HypervisorRefusalError",
    "ReceiverHypervisor",
]
