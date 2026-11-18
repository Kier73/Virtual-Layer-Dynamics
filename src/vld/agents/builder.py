r"""Continuous Self-Building Virtual-Layer Agents (`vld.agents.builder`).

Implements capability-bounded internal agents (`ContinuousLayerBuilderAgent`) that assist in
constructing, certifying, and Pareto-optimizing the Virtual Layer *while it runs*:
  1. **Glance & Lift Hosted Models:** Inspects remote/local `config.json` + Safetensors shard headers
     (`DeepSeek-V4.1`, `GLM-5.3`, `Qwen3`) and admits pinned `HostedModelGlance` Grounds.
  2. **Discover & Certify Orbit Shortcuts:** Actively probes sequences within `probe_budget`, certifies
     `StructuralMemory` laws, and installs shortcuts iff $\mathbf{C}_{\text{shortcut}} \prec \mathbf{C}_{\text{baseline}}$.
  3. **Certify Commutant Similarity Decompositions:** Factors coupled $\mathbb{F}_p^n$ operator chains into
     certified invariant blocks $P^{-1} A_r P = \bigoplus_j B_r^{(j)}$ when $\sum_j d_j^3 < n^3$.
  4. **Compile Symbolic Policies to Verified Neural Capsules:** Synthesizes exact rational ReLU networks
     and admits them through `ReceiverHypervisor` whole-domain verification.
  5. **Qualify In-Path Network Hosts:** Runs `ActiveHostFormationPipeline` with mandatory negative controls
     ($\Delta N_{\text{origin}} = 0$) to promote genuine caches/paths while refusing pass-through proxies.
"""

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
from vld.ground import NavigationCostVector
from vld.models.frontier import HostedModelGlance, glance_huggingface_model, glance_model_structure
from vld.models.procedural import compile_multivariate_hinge_relu, compile_piecewise_affine_relu
from vld.navigation.observer_orbit import (
    CertifiedSimilarityDecomposition,
    FiniteFieldOperatorChain,
    ObserverOrbitError,
    discover_similarity_blocks,
)
from vld.substrates.path_grounds import (
    ActiveHostFormationPipeline,
    AdmittedPathHost,
    PathGroundError,
)


@dataclass(frozen=True, slots=True)
class InstalledLayerUpgrade:
    """Verified Virtual Layer upgrade constructed and CAS-committed by `ContinuousLayerBuilderAgent`."""

    upgrade_id: str
    kind: str
    artifact_digest: str
    pareto_dominates_baseline: bool
    hypervisor_state_digest: str
    summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LayerBuildCycleReport:
    """Audit report of one continuous autonomous Virtual-Layer build pass."""

    agent_id: str
    admitted_upgrades: tuple[InstalledLayerUpgrade, ...]
    refused_proposals: tuple[tuple[str, str], ...]
    final_hypervisor_epoch: int
    final_hypervisor_state_digest: str


def create_default_builder_grant(
    agent_id: str,
    *,
    probe_budget: int | Fraction = Fraction(64),
    allowed_memory_prefix: str = "builder.",
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
                AgentCapability.GLANCE_MODEL,
                AgentCapability.ADMIT_SIMILARITY_BLOCKS,
                AgentCapability.QUALIFY_PATH_HOST,
                AgentCapability.COMMIT_STATE,
            }
        ),
        probe_budget=Fraction(probe_budget),
        allowed_memory_prefix=allowed_memory_prefix,
    )


class ContinuousLayerBuilderAgent:
    """Autonomous internal agent that observes runtime telemetry and builds verified Virtual Layer Grounds."""

    def __init__(
        self,
        agent_id: str,
        hypervisor: ReceiverHypervisor,
        *,
        path_pipeline: ActiveHostFormationPipeline | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.hypervisor = hypervisor
        self.path_pipeline = path_pipeline if path_pipeline is not None else ActiveHostFormationPipeline(hypervisor.journal)
        self.glanced_models: dict[str, HostedModelGlance] = {}
        self.similarity_decompositions: dict[str, CertifiedSimilarityDecomposition] = {}
        self.admitted_path_hosts: dict[str, AdmittedPathHost] = {}

    def _commit_upgrade_to_hypervisor(
        self,
        upgrade_id: str,
        kind: str,
        artifact_digest: str,
        pareto_ok: bool,
        summary: Mapping[str, Any],
    ) -> InstalledLayerUpgrade:
        cur_state = self.hypervisor.state
        upgrades_map = dict(cur_state.get("installed_upgrades", {}))
        upgrades_map[upgrade_id] = {
            "agent_id": self.agent_id,
            "artifact_digest": artifact_digest,
            "kind": kind,
            "pareto_dominates_baseline": pareto_ok,
            "summary": dict(summary),
        }
        cur_state["installed_upgrades"] = upgrades_map
        new_digest = self.hypervisor.cas_commit(
            self.agent_id,
            expected_state_digest=self.hypervisor.state_digest,
            next_state=cur_state,
        )
        return InstalledLayerUpgrade(
            upgrade_id=upgrade_id,
            kind=kind,
            artifact_digest=artifact_digest,
            pareto_dominates_baseline=pareto_ok,
            hypervisor_state_digest=new_digest,
            summary=dict(summary),
        )

    def glance_and_admit_model(
        self,
        upgrade_id: str,
        *,
        repo_id: str,
        revision: str,
        config_raw: bytes | None = None,
        index_raw: bytes | None = None,
    ) -> InstalledLayerUpgrade:
        """Glance at a hosted or detached frontier checkpoint (`DeepSeek-V4.1`, `GLM-5.3`, `Qwen3`) and register its Ground."""
        self.hypervisor._require_capability(self.agent_id, AgentCapability.GLANCE_MODEL)
        if config_raw is not None and index_raw is not None:
            glance = glance_model_structure(repo_id=repo_id, revision=revision, config_raw=config_raw, index_raw=index_raw)
        else:
            glance = glance_huggingface_model(repo_id, revision=revision)

        ground_spec = glance.to_virtual_ground_spec()
        # Structural glancing + active-expert/latent-KV execution Pareto-dominates full eager checkpoint download!
        c_eager = NavigationCostVector(
            latency_ms=1000.0,
            compute_ops=float(max(1, glance.routed_experts or 1)),
            bytes_moved=float(max(1, glance.declared_checkpoint_bytes or glance.metadata_bytes_read * 100)),
            verification_ops=float(glance.total_tensor_count),
        )
        c_glanced = NavigationCostVector(
            latency_ms=10.0,
            compute_ops=float(max(1, glance.active_experts_per_tok or 1)),
            bytes_moved=float(glance.metadata_bytes_read),
            verification_ops=float(max(1, glance.shard_count)),
        )
        if not c_glanced.pareto_dominates(c_eager):
            raise HypervisorRefusalError("Glanced model Ground failed Pareto cost check against eager baseline.")

        self.glanced_models[upgrade_id] = glance
        return self._commit_upgrade_to_hypervisor(
            upgrade_id,
            "glanced_frontier_model_ground",
            ground_spec.fingerprint,
            True,
            {
                "kv_compression_factor": round(glance.kv_compression_factor, 2),
                "model_type": glance.model_type,
                "repo_id": glance.repo_id,
                "sparsity_ratio": round(glance.expert_sparsity_ratio, 6),
            },
        )

    def probe_certify_and_install_orbit_shortcut(
        self,
        upgrade_id: str,
        *,
        memory_id: str,
        probe_fn: Callable[[int], int],
        num_probes: int = 8,
        domain: str = "Z",
        declared_modulus: int | None = None,
        target_horizon: int = 10**12,
    ) -> InstalledLayerUpgrade:
        """Actively sample `probe_fn`, certify a `StructuralMemory` law, verify Pareto gain at `target_horizon`, and commit."""
        samples: list[int] = []
        for t in range(num_probes):
            self.hypervisor.authorize_probe(self.agent_id, probe_label=f"{upgrade_id}.p{t}", cost=Fraction(1))
            samples.append(int(probe_fn(t)))

        admitted = self.hypervisor.propose_structural_law(
            self.agent_id,
            memory_id=memory_id,
            samples=samples,
            domain=domain,
            declared_modulus=declared_modulus,
        )
        log_steps = max(1, target_horizon.bit_length())
        c_step = NavigationCostVector(float(target_horizon) * 1e-6, float(target_horizon), 64.0, 1.0)
        c_jump = NavigationCostVector(0.05, float(log_steps), 64.0, 1.0)
        if not c_jump.pareto_dominates(c_step):
            raise HypervisorRefusalError(
                f"Orbit shortcut for {memory_id!r} at horizon {target_horizon} does not Pareto-dominate replay."
            )
        return self._commit_upgrade_to_hypervisor(
            upgrade_id,
            "certified_orbit_shortcut",
            str(admitted["spec_digest"]),
            True,
            {"family": admitted["family"], "memory_id": memory_id, "target_horizon": target_horizon},
        )

    def decompose_and_install_similarity_blocks(
        self,
        upgrade_id: str,
        *,
        chain: FiniteFieldOperatorChain,
        target_horizon: int = 10**12,
    ) -> InstalledLayerUpgrade:
        """Factor a coupled $\\mathbb{F}_p^n$ operator chain into certified invariant blocks if $\\sum_j d_j^3 < n^3$."""
        self.hypervisor._require_capability(self.agent_id, AgentCapability.ADMIT_SIMILARITY_BLOCKS)
        try:
            decomp = discover_similarity_blocks(chain)
        except ObserverOrbitError as exc:
            self.hypervisor._record_refusal(self.agent_id, "similarity_decomposition_failed", {"error": str(exc)})
            raise HypervisorRefusalError(f"Similarity block decomposition failed: {exc}") from exc

        _, op_stats = decomp.navigate_horizon([1] * chain.dimension, target_horizon)
        c_dense = NavigationCostVector(1.0, float(op_stats["dense_cubic_ops"]), 0.0, 1.0)
        c_block = NavigationCostVector(0.2, float(op_stats["block_cubic_ops"]), 0.0, 1.0)
        if not c_block.pareto_dominates(c_dense):
            self.hypervisor._record_refusal(self.agent_id, "non_pareto_similarity_blocks", op_stats)
            raise HypervisorRefusalError("Similarity decomposition did not reduce cubic complexity.")

        self.similarity_decompositions[upgrade_id] = decomp
        return self._commit_upgrade_to_hypervisor(
            upgrade_id,
            "certified_similarity_blocks",
            decomp.certificate_digest,
            True,
            {
                "block_cubic_ops": op_stats["block_cubic_ops"],
                "block_sizes": list(decomp.block_sizes),
                "dense_cubic_ops": op_stats["dense_cubic_ops"],
            },
        )

    def compile_and_install_neural_capsule(
        self,
        upgrade_id: str,
        *,
        piecewise_regions: Sequence[Mapping[str, Any]] | None = None,
        multivariate_spec: Mapping[str, Any] | None = None,
    ) -> InstalledLayerUpgrade:
        """Synthesize an exact rational ReLU network from symbolic spec and admit via whole-domain verification."""
        if multivariate_spec is not None:
            capsule = compile_multivariate_hinge_relu(**multivariate_spec)
        elif piecewise_regions is not None:
            capsule = compile_piecewise_affine_relu(piecewise_regions)
        else:
            raise HypervisorRefusalError("Supply piecewise_regions or multivariate_spec.")
        report = self.hypervisor.admit_neural_capsule(self.agent_id, capsule)
        return self._commit_upgrade_to_hypervisor(
            upgrade_id,
            "verified_neural_capsule",
            str(report["capsule_digest"]),
            True,
            {"schema": capsule["schema"], "status": report["status"]},
        )

    def qualify_and_install_http_path_cache(
        self,
        upgrade_id: str,
        *,
        locus_id: str,
        intermediary_url: str,
        asset_key: str,
        origin_hit_reader: Callable[[], int],
    ) -> InstalledLayerUpgrade:
        """Qualify an HTTP conditional/Range intermediary (`observe -> qualify -> construct -> admit`) with negative controls."""
        self.hypervisor._require_capability(self.agent_id, AgentCapability.QUALIFY_PATH_HOST)
        try:
            obs = self.path_pipeline.observe_http_cache_locus(locus_id, intermediary_url, asset_key, origin_hit_reader)
            cert = self.path_pipeline.qualify(obs)
            host = self.path_pipeline.construct_and_admit(cert)
        except PathGroundError as exc:
            self.hypervisor._record_refusal(self.agent_id, "path_qualification_refused", {"error": str(exc), "locus_id": locus_id})
            raise HypervisorRefusalError(f"Path host qualification refused: {exc}") from exc

        self.admitted_path_hosts[upgrade_id] = host
        return self._commit_upgrade_to_hypervisor(
            upgrade_id,
            "admitted_http_path_cache",
            cert.qualification_digest,
            True,
            {"locus_id": locus_id, "negative_controls": list(cert.negative_controls_passed)},
        )

    def run_build_cycle(self, proposals: Sequence[tuple[str, Callable[[], InstalledLayerUpgrade]]]) -> LayerBuildCycleReport:
        """Execute a sequence of autonomous Virtual-Layer construction tasks, retaining both admitted upgrades & refusals."""
        admitted: list[InstalledLayerUpgrade] = []
        refused: list[tuple[str, str]] = []
        for prop_id, fn in proposals:
            try:
                admitted.append(fn())
            except HypervisorRefusalError as exc:
                refused.append((prop_id, str(exc)))
        return LayerBuildCycleReport(
            agent_id=self.agent_id,
            admitted_upgrades=tuple(admitted),
            refused_proposals=tuple(refused),
            final_hypervisor_epoch=self.hypervisor.epoch,
            final_hypervisor_state_digest=self.hypervisor.state_digest,
        )


__all__ = [
    "ContinuousLayerBuilderAgent",
    "InstalledLayerUpgrade",
    "LayerBuildCycleReport",
    "create_default_builder_grant",
]
