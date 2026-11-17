r"""Phase 7 — Models as Interactable Virtual-Layer Agents & Advisory Controllers (`vld.agents.controller`)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from vld.agents.hypervisor import (
    AgentCapability,
    CapabilityGrant,
    HypervisorRefusalError,
    ReceiverHypervisor,
)
from vld.models.procedural import (
    compile_piecewise_affine_relu,
    evaluate_compiled_capsule,
)


@dataclass(frozen=True, slots=True)
class AgentExplorationReport:
    """Complete verified transcript of an in-layer model agent's probe-certify-navigate session."""

    agent_id: str
    policy_capsule_digest: str
    probes_executed: int
    observed_samples: tuple[int, ...]
    certified_memory_id: str
    certified_spec_digest: str
    target_coordinate: int
    navigated_value: int
    navigation_receipt_digest: str
    final_hypervisor_state_digest: str


class InLayerModelAgent:
    """First-class Virtual-Layer neural agent interacting strictly through `ReceiverHypervisor` gates.

    Uses a verified procedural ReLU policy capsule inside the Virtual Layer to decide whether
    observations have witnessed a modular wrap (triggering law certification) or require additional
    active probes before proposing a Ground law.
    """

    def __init__(
        self,
        agent_id: str,
        hypervisor: ReceiverHypervisor,
        *,
        policy_capsule: Mapping[str, Any] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.hypervisor = hypervisor
        # Default procedural ReLU readiness policy over wrap_indicator x:
        # f(x) = 0 on (-inf, 0], linear ramp on [0, 1], saturated 1 on [1, +inf)
        default_atlas = [
            {"left": "-inf", "right": "0", "affines": [["0", "0"]]},
            {"left": "0", "right": "1", "affines": [["1", "0"]]},
            {"left": "1", "right": "+inf", "affines": [["0", "1"]]},
        ]
        capsule = dict(policy_capsule) if policy_capsule is not None else compile_piecewise_affine_relu(default_atlas)
        admitted = self.hypervisor.admit_neural_capsule(agent_id, capsule)
        self.policy_capsule = capsule
        self.policy_capsule_digest = str(admitted["capsule_digest"])

    def _neural_readiness_score(self, samples: Sequence[int]) -> Fraction:
        """Evaluate the in-layer ReLU policy on empirical wrap-and-dof feature."""
        if len(samples) < 4:
            return Fraction(0)
        diffs = [samples[i + 1] - samples[i] for i in range(len(samples) - 1)]
        has_neg = any(d < 0 for d in diffs)
        has_pos = any(d > 0 for d in diffs)
        feature = 1 if (has_neg and has_pos and len(samples) >= 4) else 0
        return evaluate_compiled_capsule(self.policy_capsule, feature)[0]

    def explore_certify_and_navigate(
        self,
        environment_probe: Callable[[int], int],
        *,
        memory_id: str,
        target_coordinate: int = 10**12,
        probe_cost: Fraction = Fraction(1),
        max_probes: int = 32,
    ) -> AgentExplorationReport:
        """Autonomously probe unknown environment until neural policy detects identifiability, certify, and jump."""
        samples: list[int] = []
        for t in range(max_probes):
            self.hypervisor.authorize_probe(
                self.agent_id,
                probe_label=f"env_step_{t}",
                cost=probe_cost,
            )
            samples.append(int(environment_probe(t)))
            if self._neural_readiness_score(samples) >= Fraction(1):
                break

        law_admission = self.hypervisor.propose_structural_law(
            self.agent_id,
            memory_id=memory_id,
            samples=samples,
            domain="modular_observed",
        )
        nav = self.hypervisor.navigate_certified_orbit(
            self.agent_id,
            memory_id=memory_id,
            coordinate=target_coordinate,
        )
        prev_digest = self.hypervisor.state_digest
        new_state_digest = self.hypervisor.cas_commit(
            self.agent_id,
            expected_state_digest=prev_digest,
            next_state={
                "certified_memory_id": memory_id,
                "last_agent": self.agent_id,
                "navigated_coordinate": target_coordinate,
                "navigated_value": nav["value"],
            },
        )
        return AgentExplorationReport(
            agent_id=self.agent_id,
            policy_capsule_digest=self.policy_capsule_digest,
            probes_executed=len(samples),
            observed_samples=tuple(samples),
            certified_memory_id=memory_id,
            certified_spec_digest=str(law_admission["spec_digest"]),
            target_coordinate=target_coordinate,
            navigated_value=int(nav["value"]),
            navigation_receipt_digest=str(nav["receipt_digest"]),
            final_hypervisor_state_digest=new_state_digest,
        )


def create_default_exploration_grant(
    agent_id: str,
    *,
    probe_budget: int | Fraction = Fraction(25),
    allowed_memory_prefix: str = "agent.",
) -> CapabilityGrant:
    return CapabilityGrant(
        agent_id=agent_id,
        capabilities=frozenset(
            {
                AgentCapability.OBSERVE_TELEMETRY,
                AgentCapability.SELECT_PROBE,
                AgentCapability.PROPOSE_LAW,
                AgentCapability.NAVIGATE_ORBIT,
                AgentCapability.ADMIT_NEURAL_CAPSULE,
                AgentCapability.COMMIT_STATE,
            }
        ),
        probe_budget=Fraction(probe_budget),
        allowed_memory_prefix=allowed_memory_prefix,
    )


__all__ = [
    "AgentExplorationReport",
    "HypervisorRefusalError",
    "InLayerModelAgent",
    "create_default_exploration_grant",
]
