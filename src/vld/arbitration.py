r"""Phase 9 — Multi-Source Virtual-Layer Model Arbitration, Self-Hosted Meta-Arbitrator Models & Idea Ledger (`vld.arbitration`).

Mathematical & architectural contract:
Let $\mathcal{W} = \{W_1, \dots, W_K\}$ be heterogeneous model Grounds hosted across local checkpoints,
live Hugging Face HTTP `206` Range shards (`DeepSeek-V4.1`, `DeepSeek-R1-Distill`, `GLM-5.3`, `GLM-Z1`,
`Qwen3`, `SmolLM2`), external HTTP inference endpoints, and procedural in-layer neural capsules.

An **In-Layer Meta-Arbitrator Model** $M_{\text{arb}}$ running *inside* the Virtual Layer evaluates a
whole-domain-verified multivariate ReLU utility functional over $\mathbb{Q}^4$:
$$U(c, s, b, h) = 4c + 2\operatorname{ReLU}\!\left(s - \frac{1}{2}\right) - 2\operatorname{ReLU}\!\left(b - \frac{1}{4}\right) - 8\operatorname{ReLU}(1 - h)$$
where $c \in [0, 1]$ is capability alignment, $s \in [0, 1]$ is structural sparsity / KV compression gain,
$b \ge 0$ is normalized wire-byte footprint, and $h \in [0, 1]$ is source health ( optionally co-scored
with a pretrained LLM Ground's token/hidden-state affinity).

Every arbitration decision passes through:
  1. `ReceiverHypervisor.authorize_model_arbitration` (`AgentCapability.ARBITRATE_MODEL`, bounded depth
     $1 \le d \le d_{\max}$, anti-self-loop $M_{\text{arb}} \neq W_i$, exact `Fraction` budget deduction),
  2. Quorum-fenced `ContinuousVirtualSystem` stage execution (`2q > N`, `require_independent_verifier=True`),
  3. Standalone `In-Formation-Space`-compatible Encounter Envelopes & epistemic claim classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.error
import urllib.request

from vld.agents.hypervisor import (
    AgentCapability,
    CapabilityGrant,
    HypervisorRefusalError,
    ReceiverHypervisor,
)
from vld.authority import (
    EncounterEnvelope,
    QuorumAuthorityMember,
    QuorumLineageCoordinator,
    QuorumMembership,
    TemporalCoordinate,
    VectorClock,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.continuous import (
    ContinuousPipelineResult,
    ContinuousStageSpec,
    ContinuousVirtualSystem,
    build_compact_deepseek_v41_fixture,
    build_compact_glm53_fixture,
)
from vld.fabric.mtls import MTLSPeerClient, MutualTLSError
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.journal import GENESIS_PREV_DIGEST, HashChainedJournal
from vld.memory import DiskPageBacking, InMemoryPageBacking, PagedObjectStore
from vld.models import (
    DeepSeekV41LayerRealizer,
    DeepSeekV41ReferenceOracle,
    Glm53LayerRealizer,
    TransformerArchitectureSpec,
    TransformerLayerRealizer,
    compile_multivariate_hinge_relu,
    evaluate_multivariate_capsule,
    verify_mla_matrix_absorption,
)


class ArbitrationError(RuntimeError):
    """Raised when model worker registration, arbitration routing, or artifact verification fails."""


def create_meta_arbitrator_grant(
    agent_id: str,
    *,
    probe_budget: int | Fraction = Fraction(128),
    max_arbitration_depth: int = 2,
    allowed_memory_prefix: str = "arb.",
) -> CapabilityGrant:
    """Create a `CapabilityGrant` permitting an in-layer model to arbitrate downstream models."""
    return CapabilityGrant(
        agent_id=agent_id,
        capabilities=frozenset(
            {
                AgentCapability.OBSERVE_TELEMETRY,
                AgentCapability.SELECT_PROBE,
                AgentCapability.ADMIT_NEURAL_CAPSULE,
                AgentCapability.GLANCE_MODEL,
                AgentCapability.ARBITRATE_MODEL,
                AgentCapability.COMMIT_STATE,
            }
        ),
        probe_budget=Fraction(probe_budget),
        allowed_memory_prefix=allowed_memory_prefix,
        max_arbitration_depth=max_arbitration_depth,
    )


@dataclass(frozen=True, slots=True)
class ExplorationTaskSpec:
    """Declarative research, synthesis, or navigation task routed through the Virtual Layer."""

    task_id: str
    title: str
    required_tags: tuple[str, ...]
    prompt_text: str
    prompt_tokens: tuple[int, ...] = (3, 7, 11, 19)
    desired_worker_count: int = 2
    epistemic_target: str = "Experiment"
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.title:
            raise ArbitrationError("task_id and title must be non-empty.")
        if self.desired_worker_count <= 0:
            raise ArbitrationError("desired_worker_count must be positive.")


@dataclass(frozen=True, slots=True)
class ArbitratedModelWorker:
    """Uniform Virtual-Layer descriptor and execution callback for any hosted or in-layer model."""

    worker_id: str
    model_family: str
    source_kind: str
    source_uri: str
    capabilities: frozenset[str]
    sparsity_gain: Fraction
    normalized_byte_cost: Fraction
    health: Fraction
    ground_spec: VirtualGroundSpec
    run_fn: Callable[[ExplorationTaskSpec, Mapping[str, Any]], tuple[dict[str, Any], NavigationCostVector]]
    verify_fn: Callable[[ExplorationTaskSpec, Mapping[str, Any], Mapping[str, Any]], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class WorkerDispatchScore:
    """Exact rational score assigned to a candidate model worker by `InLayerMetaArbitratorModel`."""

    worker_id: str
    model_family: str
    capability_match: Fraction
    neural_score: Fraction
    llm_affinity_bonus: Fraction
    final_score: Fraction


@dataclass(frozen=True, slots=True)
class ArbitrationTaskReport:
    """Verified outcome of arbitrating a task across selected Virtual-Layer model workers."""

    cycle_index: int
    task_id: str
    arbitrator_id: str
    arbitrator_capsule_digest: str
    arbitration_depth: int
    ranked_scores: tuple[WorkerDispatchScore, ...]
    selected_workers: tuple[str, ...]
    hypervisor_authorizations: tuple[Mapping[str, Any], ...]
    pipeline_result: ContinuousPipelineResult
    encounter_envelope_digest: str


class InLayerMetaArbitratorModel:
    r"""Neural Meta-Arbitrator running *inside* `vld` that schedules & arbitrates other models in `vld`."""

    def __init__(
        self,
        agent_id: str,
        hypervisor: ReceiverHypervisor,
        *,
        arbitration_capsule: Mapping[str, Any] | None = None,
        co_scorer_fn: Callable[[ExplorationTaskSpec, ArbitratedModelWorker], Fraction] | None = None,
        arbitration_depth: int = 1,
    ) -> None:
        self.agent_id = agent_id
        self.hypervisor = hypervisor
        self.co_scorer_fn = co_scorer_fn
        self.arbitration_depth = arbitration_depth

        # Compile exact 4D multivariate ReLU utility model over (cap_match, sparsity, byte_cost, health):
        # U(c, s, b, h) = 4*c + 2*ReLU(s - 1/2) - 2*ReLU(b - 1/4) - 8*ReLU(1 - h)
        default_capsule = compile_multivariate_hinge_relu(
            input_dim=4,
            base_linear=[["4", "0", "0", "0"]],
            base_bias=["0"],
            hinges=[
                {"normal": ["0", "1", "0", "0"], "offset": "-1/2", "weights": ["2"]},
                {"normal": ["0", "0", "1", "0"], "offset": "-1/4", "weights": ["-2"]},
                {"normal": ["0", "0", "0", "-1"], "offset": "1", "weights": ["-8"]},
            ],
        )
        cap_obj = dict(arbitration_capsule) if arbitration_capsule is not None else default_capsule
        admitted = self.hypervisor.admit_neural_capsule(agent_id, cap_obj)
        self.capsule = cap_obj
        self.capsule_digest = str(admitted["capsule_digest"])

    def score_workers(
        self,
        task: ExplorationTaskSpec,
        workers: Sequence[ArbitratedModelWorker],
    ) -> tuple[WorkerDispatchScore, ...]:
        if not workers:
            raise ArbitrationError("No model workers registered for arbitration.")
        req = set(task.required_tags)
        scored: list[WorkerDispatchScore] = []
        for w in workers:
            if w.worker_id == self.agent_id:
                continue
            if not req:
                c_match = Fraction(1)
            else:
                hits = len(req & w.capabilities)
                c_match = Fraction(hits, len(req))
            features = [c_match, Fraction(w.sparsity_gain), Fraction(w.normalized_byte_cost), Fraction(w.health)]
            neural_u = evaluate_multivariate_capsule(self.capsule, features)[0]
            bonus = Fraction(0)
            if self.co_scorer_fn is not None:
                bonus = Fraction(self.co_scorer_fn(task, w))
            scored.append(
                WorkerDispatchScore(
                    worker_id=w.worker_id,
                    model_family=w.model_family,
                    capability_match=c_match,
                    neural_score=neural_u,
                    llm_affinity_bonus=bonus,
                    final_score=neural_u + bonus,
                )
            )
        scored.sort(key=lambda s: (-s.final_score, s.worker_id))
        return tuple(scored)

    def arbitrate_and_execute(
        self,
        task: ExplorationTaskSpec,
        workers: Mapping[str, ArbitratedModelWorker],
        *,
        cycle_index: int,
        store: PagedObjectStore,
        quorum: QuorumLineageCoordinator,
        active_quorum_members: Sequence[str],
        journal: HashChainedJournal,
        host_id: str = "vld.arbitration.host",
    ) -> ArbitrationTaskReport:
        ranked = self.score_workers(task, tuple(workers.values()))
        chosen_scores = [s for s in ranked if s.final_score > Fraction(-4)][: task.desired_worker_count]
        if not chosen_scores:
            raise ArbitrationError(f"No healthy eligible model workers found for task {task.task_id!r}.")

        authorizations: list[Mapping[str, Any]] = []
        stages: list[ContinuousStageSpec] = []
        prev_rep = "task_genesis"
        for s_idx, w_score in enumerate(chosen_scores):
            worker = workers[w_score.worker_id]
            auth = self.hypervisor.authorize_model_arbitration(
                self.agent_id,
                target_worker_id=worker.worker_id,
                task_id=f"{task.task_id}.c{cycle_index}.w{s_idx}",
                arbitration_depth=self.arbitration_depth,
                cost=Fraction(1),
                idempotent=True,
            )
            authorizations.append(auth)
            out_rep = f"model_out_{worker.model_family}_{s_idx}"

            def _make_exec(
                wk: ArbitratedModelWorker,
            ) -> Callable[[Mapping[str, Any]], tuple[dict[str, Any], NavigationCostVector, str]]:
                def _exec(cur_state: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                    out_payload, cost_vec = wk.run_fn(task, cur_state)
                    merged = dict(cur_state)
                    outputs_map = dict(merged.get("worker_outputs", {}))
                    outputs_map[wk.worker_id] = out_payload
                    merged["worker_outputs"] = outputs_map
                    merged["last_worker"] = wk.worker_id
                    return merged, cost_vec, f"PASS:{wk.worker_id}"

                return _exec

            def _make_ver(
                wk: ArbitratedModelWorker,
            ) -> Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[bool, str]]:
                def _ver(inp_state: Mapping[str, Any], out_state: Mapping[str, Any]) -> tuple[bool, str]:
                    w_map = out_state.get("worker_outputs", {})
                    if wk.worker_id not in w_map:
                        return False, "missing_worker_output"
                    return wk.verify_fn(task, inp_state, w_map[wk.worker_id])

                return _ver

            stages.append(
                ContinuousStageSpec(
                    stage_id=f"arb.{worker.worker_id}",
                    ground_spec=worker.ground_spec,
                    input_representation=prev_rep,
                    output_representation=out_rep,
                    scheduler_id=f"sched.arb.{worker.model_family}",
                    execute_fn=_make_exec(worker),
                    verify_fn=_make_ver(worker),
                )
            )
            prev_rep = out_rep

        pipeline_id = f"arb.{task.task_id}.cycle.{cycle_index}"
        cvs = ContinuousVirtualSystem(
            pipeline_id,
            store=store,
            quorum=quorum,
            active_quorum_members=active_quorum_members,
            journal=journal,
            require_independent_verifier=True,
            reverify_recovered_stages=True,
        )
        init_input = {
            "arbitrator_id": self.agent_id,
            "cycle_index": cycle_index,
            "prompt_text": task.prompt_text,
            "prompt_tokens": list(task.prompt_tokens),
            "task_id": task.task_id,
            "worker_outputs": {},
        }
        pipe_res = cvs.execute_stages(stages, init_input, host_id=host_id)

        raw_final = canonical_bytes(pipe_res.final_state)
        enc = EncounterEnvelope(
            encounter_id=f"enc.{pipeline_id}",
            observer_id=self.agent_id,
            temporal_coord=TemporalCoordinate(
                scheduler_id="sched.meta.arbitrator",
                epoch=cycle_index + 1,
                clock=VectorClock.from_mapping({self.agent_id: cycle_index + 1}),
            ),
            raw_sha256=hashlib.sha256(raw_final).hexdigest(),
            byte_length=len(raw_final),
            channel_meta={
                "continuity_digest": pipe_res.final_continuity_digest,
                "epistemic_target": task.epistemic_target,
                "selected_workers": [s.worker_id for s in chosen_scores],
            },
        )
        return ArbitrationTaskReport(
            cycle_index=cycle_index,
            task_id=task.task_id,
            arbitrator_id=self.agent_id,
            arbitrator_capsule_digest=self.capsule_digest,
            arbitration_depth=self.arbitration_depth,
            ranked_scores=ranked,
            selected_workers=tuple(s.worker_id for s in chosen_scores),
            hypervisor_authorizations=tuple(authorizations),
            pipeline_result=pipe_res,
            encounter_envelope_digest=enc.encounter_digest,
        )


# ============================================================================
# Built-in Multi-Family Model Worker Constructors (DeepSeek, GLM, Qwen/Llama, HTTP, Sub-Arbitrator)
# ============================================================================


def build_deepseek_v41_worker(worker_id: str = "worker.deepseek.v41", *, seed: int = 7) -> ArbitratedModelWorker:
    arch, weights = build_compact_deepseek_v41_fixture(seed=seed)
    spec = VirtualGroundSpec(
        worker_id, "1.0", "deepseek_v41", "mhc+fp8_fp4_moe", ("forward",), ("oracle_parity",), ("top1_token",)
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        toks = [int(t) % arch.vocab_size for t in task.prompt_tokens]
        runner = DeepSeekV41LayerRealizer(arch, weights)
        hc_s, pm_s = runner.initial_hc_stream(toks)
        routed_seen: set[int] = set()
        for l_i in range(arch.num_hidden_layers):
            out = runner.forward_layer(l_i, hc_s, pm_s)
            hc_s, pm_s = out.hc_stream, out.next_pre_mix
            for rec in out.moe_receipts:
                routed_seen.update(rec.selected_experts)
        logits = runner.collapse_and_logits(hc_s, pm_s)
        top1 = max(range(arch.vocab_size), key=lambda idx: logits[-1][idx])
        return (
            {
                "active_unique_experts": sorted(routed_seen),
                "last_logit_max": round(float(logits[-1][top1]), 6),
                "model_family": "deepseek_v41",
                "tokens_in": toks,
                "top1_token": top1,
                "total_routed_experts": arch.n_routed_experts,
            },
            NavigationCostVector(0.9, 1024.0, 512.0, 8.0),
        )

    def _ver(task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        toks = [int(t) % arch.vocab_size for t in task.prompt_tokens]
        ref_logits = DeepSeekV41ReferenceOracle.forward_full(arch, weights, toks)
        exp_top1 = max(range(arch.vocab_size), key=lambda idx: ref_logits[-1][idx])
        ok = int(w_out.get("top1_token", -1)) == exp_top1 and abs(float(w_out["last_logit_max"]) - ref_logits[-1][exp_top1]) < 1e-3
        return ok, f"deepseek_v41_oracle_top1={exp_top1}"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family="deepseek_v41",
        source_kind="hf_frontier_deepseek",
        source_uri=f"vld://{arch.source_model}@{arch.revision[:12]}",
        capabilities=frozenset({"sparse_moe", "fp4_quant", "sink_mla", "frontier_reasoning"}),
        sparsity_gain=Fraction(3, 4),
        normalized_byte_cost=Fraction(1, 8),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


def build_glm_53_worker(worker_id: str = "worker.glm.53", *, seed: int = 11) -> ArbitratedModelWorker:
    arch, weights = build_compact_glm53_fixture(seed=seed)
    spec = VirtualGroundSpec(
        worker_id, "1.0", "glm_moe_dsa", "latent_mla+sigmoid_moe", ("forward",), ("mla_absorption",), ("top1_token",)
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        toks = [int(t) % arch.vocab_size for t in task.prompt_tokens]
        runner = Glm53LayerRealizer(arch, weights, mla_mode="latent_absorbed")
        h = runner.embed_tokens(toks)
        abs_rep = verify_mla_matrix_absorption(arch, weights, h, layer_index=0)
        for l_i in range(arch.num_hidden_layers):
            h = runner.forward_layer(l_i, h).hidden_states
        logits = runner.final_logits(h)
        top1 = max(range(arch.vocab_size), key=lambda idx: logits[-1][idx])
        return (
            {
                "kv_compression_ratio": round(2.0 * arch.num_attention_heads * arch.v_head_dim / arch.latent_kv_width, 2),
                "mla_absorption_max_diff": float(abs_rep["max_abs_diff"]),
                "model_family": "glm_53",
                "tokens_in": toks,
                "top1_token": top1,
            },
            NavigationCostVector(0.8, 896.0, 384.0, 8.0),
        )

    def _ver(task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        toks = [int(t) % arch.vocab_size for t in task.prompt_tokens]
        mat_runner = Glm53LayerRealizer(arch, weights, mla_mode="materialized")
        h = mat_runner.embed_tokens(toks)
        for l_i in range(arch.num_hidden_layers):
            h = mat_runner.forward_layer(l_i, h).hidden_states
        ref_logits = mat_runner.final_logits(h)
        exp_top1 = max(range(arch.vocab_size), key=lambda idx: ref_logits[-1][idx])
        ok = int(w_out.get("top1_token", -1)) == exp_top1 and float(w_out.get("mla_absorption_max_diff", 1.0)) < 1e-5
        return ok, f"glm53_materialized_parity_top1={exp_top1}"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family="glm_53",
        source_kind="hf_frontier_glm",
        source_uri=f"vld://{arch.source_model}@{arch.revision[:12]}",
        capabilities=frozenset({"mla_absorption", "sparse_moe", "kv_compression", "frontier_reasoning"}),
        sparsity_gain=Fraction(4, 5),
        normalized_byte_cost=Fraction(1, 10),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


def build_transformer_ground_worker(
    worker_id: str,
    *,
    arch: TransformerArchitectureSpec,
    snapshot: Any,
    capabilities: Sequence[str],
    max_layers_to_run: int | None = None,
) -> ArbitratedModelWorker:
    """Wrap any `qwen3`, `qwen3_moe`, `qwen2` (`DeepSeek-R1`), `llama` (`SmolLM2`), or `glm4` (`GLM-Z1`) Ground."""
    layers_cap = arch.num_hidden_layers if max_layers_to_run is None else min(arch.num_hidden_layers, max_layers_to_run)
    spec = VirtualGroundSpec(
        worker_id,
        "1.0",
        arch.model_type,
        f"layers_0..{layers_cap - 1}",
        ("forward_layers",),
        ("finite_hidden_norm",),
        ("hidden_digest",),
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        toks = [int(t) % arch.vocab_size for t in task.prompt_tokens]
        realizer = TransformerLayerRealizer(arch, snapshot)
        h = realizer.embed_tokens(toks)
        tensors_loaded = 0
        for l_i in range(layers_cap):
            out = realizer.forward_layer(l_i, h)
            h = out.hidden_states
            tensors_loaded += len(out.loaded_tensor_names)
        top1 = -1
        if layers_cap == arch.num_hidden_layers:
            logits = realizer.final_logits(h)
            top1 = max(range(arch.vocab_size), key=lambda idx: logits[-1][idx])
        rms_last = math.sqrt(math.fsum(float(v) * float(v) for v in h[-1]) / len(h[-1]))
        h_dg = hashlib.sha256(canonical_bytes([[round(float(x), 5) for x in row[:8]] for row in h])).hexdigest()
        return (
            {
                "arch_fingerprint": arch.fingerprint,
                "hidden_prefix_digest": h_dg,
                "last_token_rms": round(rms_last, 6),
                "layers_executed": layers_cap,
                "model_type": arch.model_type,
                "source_model": arch.source_model,
                "tensors_loaded": tensors_loaded,
                "top1_token": top1,
            },
            NavigationCostVector(1.0, float(layers_cap * arch.hidden_size), float(tensors_loaded * 64), 4.0),
        )

    def _ver(_task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        rms = float(w_out.get("last_token_rms", -1.0))
        ok = (
            str(w_out.get("arch_fingerprint", "")) == arch.fingerprint
            and int(w_out.get("layers_executed", 0)) == layers_cap
            and math.isfinite(rms)
            and rms > 0.0
        )
        return ok, f"{arch.model_type}_finite_rms={rms:.4f}"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family=arch.model_type,
        source_kind="hf_transformer",
        source_uri=f"hf://{arch.source_model}@{arch.revision[:12]}",
        capabilities=frozenset(capabilities),
        sparsity_gain=Fraction(3, 5) if arch.is_moe else Fraction(1, 2),
        normalized_byte_cost=Fraction(1, 5),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


def build_nested_sub_arbitrator_worker(
    worker_id: str,
    *,
    sub_arbitrator: InLayerMetaArbitratorModel,
    sub_workers: Mapping[str, ArbitratedModelWorker],
) -> ArbitratedModelWorker:
    """Wrap a Depth-2 `InLayerMetaArbitratorModel` AS a worker so a Depth-1 model in `vld` arbitrates a model in `vld`!"""
    spec = VirtualGroundSpec(
        worker_id,
        "1.0",
        "in_layer_meta_arbitrator",
        "recursive_arbitration",
        ("arbitrate_subfleet",),
        ("quorum_continuity",),
        ("sub_continuity_digest",),
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        q_mem = QuorumMembership(f"sub.{worker_id}.{task.task_id}", ("sq0", "sq1", "sq2"))
        secs = {m: f"sub-sec-{m}-0123456789".encode("utf-8") for m in q_mem.member_ids}
        nodes = [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids]
        sub_quorum = QuorumLineageCoordinator(q_mem, nodes, secs)
        sub_store = PagedObjectStore(InMemoryPageBacking(4096))
        sub_journal = HashChainedJournal()
        sub_task = ExplorationTaskSpec(
            task_id=f"{task.task_id}.nested",
            title=f"Nested: {task.title}",
            required_tags=task.required_tags,
            prompt_text=task.prompt_text,
            prompt_tokens=task.prompt_tokens,
            desired_worker_count=min(2, len(sub_workers)),
        )
        rep = sub_arbitrator.arbitrate_and_execute(
            sub_task,
            sub_workers,
            cycle_index=0,
            store=sub_store,
            quorum=sub_quorum,
            active_quorum_members=("sq0", "sq1"),
            journal=sub_journal,
            host_id=f"host.{worker_id}",
        )
        return (
            {
                "nested_arbitrator_capsule": rep.arbitrator_capsule_digest,
                "nested_arbitrator_id": sub_arbitrator.agent_id,
                "nested_continuity_digest": rep.pipeline_result.final_continuity_digest,
                "nested_selected_workers": list(rep.selected_workers),
                "nested_worker_outputs": dict(rep.pipeline_result.final_state.get("worker_outputs", {})),
            },
            rep.pipeline_result.total_cost,
        )

    def _ver(_task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        sel = list(w_out.get("nested_selected_workers", ()))
        outs = dict(w_out.get("nested_worker_outputs", {}))
        ok = (
            len(sel) >= 1
            and all(s in outs for s in sel)
            and len(str(w_out.get("nested_continuity_digest", ""))) == 64
        )
        return ok, f"nested_arbitrated={len(sel)}_models"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family="meta_arbitrator_vld",
        source_kind="in_layer_meta_model",
        source_uri=f"vld://agent/{sub_arbitrator.agent_id}",
        capabilities=frozenset({"meta_arbitration", "sparse_moe", "mla_absorption", "frontier_reasoning"}),
        sparsity_gain=Fraction(9, 10),
        normalized_byte_cost=Fraction(1, 16),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


def build_external_http_endpoint_worker(
    worker_id: str,
    *,
    endpoint_url: str,
    model_name: str,
    capabilities: Sequence[str] = ("text_generation",),
    api_key: str | None = None,
    timeout_sec: float = 15.0,
) -> ArbitratedModelWorker:
    """Connect any OpenAI-compatible (`/v1/chat/completions`) or Ollama (`/api/generate`) endpoint to `vld`."""
    spec = VirtualGroundSpec(
        worker_id, "1.0", "http-llm-endpoint", "json_stream", ("chat_completion",), ("utf8_sha256",), ("response_text",)
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        headers = {"Content-Type": "application/json", "User-Agent": "vld-arbitrator/0.6"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if endpoint_url.rstrip("/").endswith("/api/generate"):
            body = canonical_bytes({"model": model_name, "prompt": task.prompt_text, "stream": False})
        else:
            body = canonical_bytes(
                {
                    "messages": [{"content": task.prompt_text, "role": "user"}],
                    "model": model_name,
                    "temperature": 0.0,
                }
            )
        req = urllib.request.Request(endpoint_url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise ArbitrationError(f"External HTTP model endpoint {endpoint_url!r} failed: {exc}") from exc
        doc = parse_strict_json(raw)
        if not isinstance(doc, Mapping):
            raise ArbitrationError("External HTTP model endpoint returned non-object JSON.")
        choices = doc.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], Mapping) else {}
        msg_obj = first_choice.get("message") if isinstance(first_choice.get("message"), Mapping) else {}
        text = str(doc.get("response") or msg_obj.get("content") or "")
        if not text:
            raise ArbitrationError(f"External HTTP model endpoint {endpoint_url!r} returned empty completion.")
        return (
            {
                "model_name": model_name,
                "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "response_text": text,
                "wire_bytes": len(body) + len(raw),
            },
            NavigationCostVector(5.0, float(max(1, len(text))), float(len(body) + len(raw)), 2.0),
        )

    def _ver(_task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        txt = str(w_out.get("response_text", ""))
        ok = bool(txt) and hashlib.sha256(txt.encode("utf-8")).hexdigest() == w_out.get("response_sha256")
        return ok, "http_endpoint_sha256_verified"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family="external_http",
        source_kind="http_endpoint",
        source_uri=endpoint_url,
        capabilities=frozenset(capabilities),
        sparsity_gain=Fraction(1, 2),
        normalized_byte_cost=Fraction(1, 4),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


def build_mtls_fabric_worker(
    worker_id: str,
    *,
    mtls_client: MTLSPeerClient,
    target_store: PagedObjectStore,
    model_family: str = "mtls_fabric_model",
    capabilities: Sequence[str] = ("mtls_fabric", "frontier_reasoning"),
    independent_verifier_fn: Callable[[ExplorationTaskSpec, Mapping[str, Any]], tuple[bool, str]] | None = None,
) -> ArbitratedModelWorker:
    """Connect a remote X.509 v3 mutually-authenticated TLS VLD peer (`MTLSPeerServer`) as an arbitrated worker."""
    spec = VirtualGroundSpec(
        worker_id,
        "1.0",
        model_family,
        "mtls_control+m1_range_stream",
        ("invoke_control", "fetch_m1_object_resumable"),
        ("m1_sha256_manifest",),
        ("remote_result",),
    )

    def _run(task: ExplorationTaskSpec, _st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector]:
        try:
            ctrl_resp = mtls_client.invoke_control(
                {
                    "op": "execute_arbitrated_model",
                    "prompt_text": task.prompt_text,
                    "prompt_tokens": list(task.prompt_tokens),
                    "task_id": task.task_id,
                }
            )
            manifest_dg = str(ctrl_resp["result_manifest_digest"])
            manifest, reconnects = mtls_client.fetch_m1_object_resumable(manifest_dg, target_store)
            result_doc = parse_strict_json(target_store.read_bytes(manifest))
            if not isinstance(result_doc, Mapping):
                raise ArbitrationError("Remote mTLS worker M1 payload must be a JSON object.")
        except (MutualTLSError, KeyError, ValueError) as exc:
            raise ArbitrationError(f"mTLS fabric worker {worker_id!r} failed: {exc}") from exc
        return (
            {
                "m1_manifest_digest": manifest.manifest_digest,
                "m1_size_bytes": manifest.size_bytes,
                "peer_dns": mtls_client.expected_server_dns,
                "reconnects": reconnects,
                "remote_result": dict(result_doc),
            },
            NavigationCostVector(1.5, float(max(1, manifest.size_bytes)), float(manifest.size_bytes), 4.0),
        )

    def _ver(task: ExplorationTaskSpec, _inp: Mapping[str, Any], w_out: Mapping[str, Any]) -> tuple[bool, str]:
        m_dg = str(w_out.get("m1_manifest_digest", ""))
        rem = w_out.get("remote_result")
        if not m_dg or not isinstance(rem, Mapping):
            return False, "missing_mtls_m1_result"
        try:
            m = target_store.get_manifest(m_dg)
            if parse_strict_json(target_store.read_bytes(m)) != dict(rem):
                return False, "mtls_m1_payload_mismatch"
        except Exception:
            return False, "mtls_m1_manifest_unverified"
        if independent_verifier_fn is not None:
            return independent_verifier_fn(task, rem)
        return True, f"mtls_m1_verified:{mtls_client.expected_server_dns}"

    return ArbitratedModelWorker(
        worker_id=worker_id,
        model_family=model_family,
        source_kind="mtls_vld_fabric",
        source_uri=f"mtls://{mtls_client.expected_server_dns}:{mtls_client.port}",
        capabilities=frozenset(capabilities),
        sparsity_gain=Fraction(3, 4),
        normalized_byte_cost=Fraction(1, 8),
        health=Fraction(1),
        ground_spec=spec,
        run_fn=_run,
        verify_fn=_ver,
    )


# ============================================================================
# Continuous Autonomous Arbitration Service & Hash-Chained Idea Ledger
# ============================================================================


@dataclass(frozen=True, slots=True)
class IdeaEntry:
    """Single immutable entry in the Virtual Layer's emergent Idea & Hypothesis Ledger."""

    idea_id: str
    timestamp_utc: str
    epistemic_status: str  # Observation | Inference | Hypothesis | Experiment | Verified | Unknown
    title: str
    equation_or_invariant: str
    rationale_and_next_step: str
    origin_agent: str
    prev_digest: str
    entry_digest: str = ""

    SCHEMA = "vld.idea-ledger-entry.v1"

    def __post_init__(self) -> None:
        if self.epistemic_status not in (
            "Observation",
            "Inference",
            "Hypothesis",
            "Experiment",
            "Verified",
            "Unknown",
        ):
            raise ArbitrationError(f"Invalid epistemic_status {self.epistemic_status!r}.")
        computed = domain_digest(
            self.SCHEMA,
            {
                "equation_or_invariant": self.equation_or_invariant,
                "epistemic_status": self.epistemic_status,
                "idea_id": self.idea_id,
                "origin_agent": self.origin_agent,
                "prev_digest": self.prev_digest,
                "rationale_and_next_step": self.rationale_and_next_step,
                "timestamp_utc": self.timestamp_utc,
                "title": self.title,
            },
        )
        if self.entry_digest and self.entry_digest != computed:
            raise ArbitrationError("IdeaEntry digest tampering detected.")
        object.__setattr__(self, "entry_digest", computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_digest": self.entry_digest,
            "equation_or_invariant": self.equation_or_invariant,
            "epistemic_status": self.epistemic_status,
            "idea_id": self.idea_id,
            "origin_agent": self.origin_agent,
            "prev_digest": self.prev_digest,
            "rationale_and_next_step": self.rationale_and_next_step,
            "timestamp_utc": self.timestamp_utc,
            "title": self.title,
        }


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Atomically write `payload` to `target` via fsync'd temp file + `os.replace` (zero torn-cursor window)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


class IdeaLedger:
    """Append-only, hash-chained ledger of unasked-for ideas, conjectures, negative controls & designs."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path is not None else None
        self._entries: list[IdeaEntry] = []
        self._by_id: dict[str, IdeaEntry] = {}
        if self._path is not None and self._path.is_file():
            prev = GENESIS_PREV_DIGEST
            for line in self._path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                doc = parse_strict_json(line)
                entry = IdeaEntry(**doc)
                if entry.prev_digest != prev:
                    raise ArbitrationError("Broken hash chain in persisted IdeaLedger.")
                self._entries.append(entry)
                self._by_id.setdefault(entry.idea_id, entry)
                prev = entry.entry_digest

    @property
    def entries(self) -> tuple[IdeaEntry, ...]:
        return tuple(self._entries)

    @property
    def head_digest(self) -> str:
        return self._entries[-1].entry_digest if self._entries else GENESIS_PREV_DIGEST

    def record_idea(
        self,
        *,
        idea_id: str,
        epistemic_status: str,
        title: str,
        equation_or_invariant: str,
        rationale_and_next_step: str,
        origin_agent: str,
        timestamp_utc: str | None = None,
        deduplicate_by_id: bool = True,
    ) -> IdeaEntry:
        if deduplicate_by_id and idea_id in self._by_id:
            return self._by_id[idea_id]
        ts = timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = IdeaEntry(
            idea_id=idea_id,
            timestamp_utc=ts,
            epistemic_status=epistemic_status,
            title=title,
            equation_or_invariant=equation_or_invariant,
            rationale_and_next_step=rationale_and_next_step,
            origin_agent=origin_agent,
            prev_digest=self.head_digest,
        )
        self._entries.append(entry)
        self._by_id[idea_id] = entry
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(canonical_json(entry.to_dict()) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return entry

    def to_markdown(self) -> str:
        lines = [
            "# Emergent Virtual Layer Idea & Hypothesis Ledger",
            "",
            f"- **Entries:** `{len(self._entries)}`",
            f"- **Head SHA-256:** `{self.head_digest}`",
            "",
        ]
        for idx, e in enumerate(self._entries, start=1):
            lines.extend(
                [
                    f"## {idx}. [{e.epistemic_status}] {e.title} (`{e.idea_id}`)",
                    f"- **Origin Agent:** `{e.origin_agent}` · **Timestamp:** `{e.timestamp_utc}`",
                    f"- **Governing Equation / Invariant:** {e.equation_or_invariant}",
                    f"- **Rationale & Next Step:** {e.rationale_and_next_step}",
                    f"- **Entry Digest:** `{e.entry_digest[:16]}...` (prev `{e.prev_digest[:16]}...`)",
                    "",
                ]
            )
        return "\n".join(lines)


class ContinuousArbitrationService:
    """Crash-resumable continuous service letting in-layer models arbitrate multi-model campaigns over time."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        arbitrator_id: str = "agent.meta.arbitrator.primary",
        workers: Mapping[str, ArbitratedModelWorker] | None = None,
        ifs_output_dir: Path | str | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.ifs_output_dir = Path(ifs_output_dir) if ifs_output_dir is not None else None

        self.store = PagedObjectStore(DiskPageBacking(self.state_dir / "m1_pages", page_size=4096))
        self.journal = HashChainedJournal(self.state_dir / "arbitration_journal.jsonl")
        self.hypervisor_journal = HashChainedJournal(self.state_dir / "hypervisor_journal.jsonl")
        self.idea_ledger = IdeaLedger(self.state_dir / "idea_ledger.jsonl")

        self.hypervisor = ReceiverHypervisor("vld.arbitration.hypervisor", journal=self.hypervisor_journal)
        self.hypervisor.register_agent(create_meta_arbitrator_grant(arbitrator_id, probe_budget=4096, max_arbitration_depth=2))

        if workers is None:
            # Default self-contained fleet INCLUDING a nested Depth-2 in-layer sub-arbitrator!
            self.hypervisor.register_agent(
                create_meta_arbitrator_grant("agent.meta.sub_arbitrator", probe_budget=2048, max_arbitration_depth=2)
            )
            sub_arb = InLayerMetaArbitratorModel("agent.meta.sub_arbitrator", self.hypervisor, arbitration_depth=2)
            w_ds = build_deepseek_v41_worker("worker.deepseek.v41")
            w_glm = build_glm_53_worker("worker.glm.53")
            w_nested = build_nested_sub_arbitrator_worker(
                "worker.in_layer.sub_arbitrator",
                sub_arbitrator=sub_arb,
                sub_workers={w_ds.worker_id: w_ds, w_glm.worker_id: w_glm},
            )
            self.workers: dict[str, ArbitratedModelWorker] = {
                w_ds.worker_id: w_ds,
                w_glm.worker_id: w_glm,
                w_nested.worker_id: w_nested,
            }
        else:
            self.workers = dict(workers)

        self.meta_arbitrator = InLayerMetaArbitratorModel(arbitrator_id, self.hypervisor, arbitration_depth=1)
        self._cursor_file = self.state_dir / "service_cursor.json"

    def completed_cycles(self) -> int:
        if not self._cursor_file.is_file():
            return 0
        doc = parse_strict_json(self._cursor_file.read_bytes())
        return int(doc.get("completed_cycles", 0))

    def run_cycles(
        self,
        tasks: Sequence[ExplorationTaskSpec],
        *,
        num_cycles: int = 1,
        interval_sec: float = 0.0,
        _fault_before_cursor_write: Callable[[int], None] | None = None,
    ) -> tuple[ArbitrationTaskReport, ...]:
        if not tasks or num_cycles <= 0:
            raise ArbitrationError("tasks and num_cycles must be non-empty / positive.")
        start_cycle = self.completed_cycles()
        reports: list[ArbitrationTaskReport] = []

        for offset in range(num_cycles):
            cyc = start_cycle + offset
            task = tasks[cyc % len(tasks)]
            q_mem = QuorumMembership(f"arb.lineage.{cyc}", ("q0", "q1", "q2"))
            secs = {m: f"arb-qsec-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
            nodes = [
                QuorumAuthorityMember(m, secs[m], journal_path=self.state_dir / f"quorum_{cyc}_{m}.jsonl")
                for m in q_mem.member_ids
            ]
            quorum = QuorumLineageCoordinator(q_mem, nodes, secs)

            rep = self.meta_arbitrator.arbitrate_and_execute(
                task,
                self.workers,
                cycle_index=cyc,
                store=self.store,
                quorum=quorum,
                active_quorum_members=("q0", "q1", "q2"),
                journal=self.journal,
                host_id=f"vld.continuous.arb.{cyc}",
            )
            reports.append(rep)

            # Record idempotent cycle observation in IdeaLedger BEFORE cursor write (crash-safe deduplication by idea_id!)
            self.idea_ledger.record_idea(
                idea_id=f"CYCLE-OBS-{cyc:04d}-{task.task_id}",
                epistemic_status="Verified",
                title=f"Cycle {cyc}: {task.title}",
                equation_or_invariant=f"continuity_digest == {rep.pipeline_result.final_continuity_digest[:16]}...",
                rationale_and_next_step=(
                    f"Selected workers {list(rep.selected_workers)} under capsule {rep.arbitrator_capsule_digest[:12]}...; "
                    f"recovered={rep.pipeline_result.stages_recovered_from_quorum}, executed={rep.pipeline_result.stages_executed_this_invocation}."
                ),
                origin_agent=rep.arbitrator_id,
                timestamp_utc="2026-11-18T00:00:00Z",
                deduplicate_by_id=True,
            )

            if self.ifs_output_dir is not None:
                artifact_path = self.ifs_output_dir / f"cycle_{cyc:04d}_{task.task_id}.json"
                _atomic_write_bytes(
                    artifact_path,
                    canonical_bytes(
                        {
                            "arbitrator_capsule_digest": rep.arbitrator_capsule_digest,
                            "arbitrator_id": rep.arbitrator_id,
                            "continuity_digest": rep.pipeline_result.final_continuity_digest,
                            "cycle_index": cyc,
                            "encounter_envelope_digest": rep.encounter_envelope_digest,
                            "epistemic_status": "Verified",
                            "final_state": dict(rep.pipeline_result.final_state),
                            "measured_wall_ms": round(rep.pipeline_result.measured_wall_ms, 4),
                            "model_cost": rep.pipeline_result.total_cost.to_dict(),
                            "selected_workers": list(rep.selected_workers),
                            "task_id": task.task_id,
                        }
                    ),
                )

            if _fault_before_cursor_write is not None:
                _fault_before_cursor_write(cyc)

            cursor_doc = {
                "completed_cycles": cyc + 1,
                "idea_ledger_head_digest": self.idea_ledger.head_digest,
                "last_continuity_digest": rep.pipeline_result.final_continuity_digest,
                "last_encounter_digest": rep.encounter_envelope_digest,
                "last_selected_workers": list(rep.selected_workers),
                "last_task_id": task.task_id,
            }
            _atomic_write_bytes(self._cursor_file, canonical_bytes(cursor_doc))

            if interval_sec > 0.0 and offset + 1 < num_cycles:
                time.sleep(interval_sec)

        return tuple(reports)


__all__ = [
    "ArbitratedModelWorker",
    "ArbitrationError",
    "ArbitrationTaskReport",
    "ContinuousArbitrationService",
    "ExplorationTaskSpec",
    "IdeaEntry",
    "IdeaLedger",
    "InLayerMetaArbitratorModel",
    "WorkerDispatchScore",
    "build_deepseek_v41_worker",
    "build_external_http_endpoint_worker",
    "build_glm_53_worker",
    "build_mtls_fabric_worker",
    "build_nested_sub_arbitrator_worker",
    "build_transformer_ground_worker",
    "create_meta_arbitrator_grant",
]
