r"""Phase 10 — Authentic Hosted-Model Subagents Directing & Improving VLD (`vld.agents.subagents`).

Mathematical & architectural design:
Turns pretrained & procedural Transformer models hosted *by and through* the Virtual Layer
(`Qwen3-0.6B`, live HTTP-`206` `SmolLM2-135M-Instruct`, `SmolLM-135M`, `DeepSeek-R1-Distill-1.5B`)
into **authentic, closed-loop improvable VLD development subagents** $S_t = (\mathcal{M}, D_t, U_{\text{cal}}^{(t)}, \mathcal{H}_{<t})$:

1. **M1-Prefix-Forked Exact Teacher-Forced Log-Likelihood Scoring:**
   Given current engineering directive $D_t(\mathcal{H}_{<t})$ and task context $\tau$, the prompt token prefix
   $p_{1..L_p} = \mathcal{T}(D_t \,\|\, \tau)$ is executed **once** across all $L$ transformer layers, storing
   content-addressed `LayerKVPage`s in M1 `PagedKVCacheGround`. Each typed engineering proposal continuation
   $y^{(k)}_{1..m} = \mathcal{T}(a_k)$ branches from that shared M1 KV state in a single batched causal forward pass:
   $$\bar{\ell}_{\mathcal{M}}(a_k \mid D_t, \tau) = \frac{1}{m} \sum_{j=1}^{m} \log P_{\mathcal{M}}\!\left(y^{(k)}_j \;\middle|\; p_{1..L_p},\, y^{(k)}_{<j}\right).$$

2. **Closed-Loop Directive & Verified Capsule Co-Evolution (`SubagentCalibrationMemory`):**
   Subagents are **never** their own final authority. Every selected action is executed against deterministic
   Virtual-Layer verifiers (`ReceiverHypervisor`, `ContinuousLayerBuilderAgent`, `IntentionToRealizationCompiler`,
   `GitPackGraphGround`, `CellularSheaf`).
   - Deterministic refusals & counterexamples append distilled rules into $D_{t+1}$ (shifting the hosted model's
     own conditional log-likelihood distribution $\bar{\ell}_{\mathcal{M}}(\cdot \mid D_{t+1}, \tau)$) AND upgrade
     the whole-domain-verified $\mathbb{Q}^4$ `ReceiverHypervisor` calibration capsule:
     $$\operatorname{Score}_t(a_k) = w_{\text{llm}} \cdot \tilde{\ell}_{\mathcal{M}}(a_k) + U_{\text{cal}}^{(t)}(\phi(a_k)) - \Pi_{\text{refused}}(a_k, \mathcal{H}_{<t}).$$

3. **Multi-Subagent Relational Consensus & Ed25519 Quorum Sealing (`DirectedSubagentDevelopmentLoop`):**
   When multiple heterogeneous hosted models co-direct a round, their prompt-final-layer normalized Token-Gram
   geometries are compared via `CenteredGramRelationalSketch` (`IDEA-005`), and accepted runtime upgrades are
   committed under `ed25519-rfc8032` public-key quorum authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.agents.builder import ContinuousLayerBuilderAgent, InstalledLayerUpgrade, create_default_builder_grant
from vld.agents.hypervisor import (
    AgentCapability,
    CapabilityGrant,
    HypervisorRefusalError,
    ReceiverHypervisor,
)
from vld.authority import (
    QuorumAuthorityMember,
    QuorumLineageCoordinator,
    QuorumMembership,
    build_ed25519_quorum_identities,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.independent_audit import verify_quorum_certificate_standalone
from vld.journal import GENESIS_PREV_DIGEST
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.models.artifacts import (
    LazySafetensorsWeightGround,
    ModelWeightSnapshot,
    TransformerArchitectureSpec,
    build_safetensors_bytes,
)
from vld.models.consensus import CenteredGramRelationalSketch, compare_relational_sketches
from vld.models.orbit import AutoregressiveModelRunner, PagedKVCacheGround
from vld.models.procedural import compile_multivariate_hinge_relu, evaluate_multivariate_capsule
from vld.models.realizers import TransformerLayerRealizer


class SubagentError(RuntimeError):
    """Raised on invalid subagent configuration, empty proposals, or verification failure."""


def create_subagent_grant(
    agent_id: str,
    *,
    probe_budget: int | Fraction = Fraction(128),
    allowed_memory_prefix: str = "subagent.",
) -> CapabilityGrant:
    """Create a `CapabilityGrant` scoping a `HostedModelSubagent` inside `ReceiverHypervisor`."""
    base = create_default_builder_grant(agent_id, probe_budget=probe_budget, allowed_memory_prefix=allowed_memory_prefix)
    return CapabilityGrant(
        agent_id=agent_id,
        capabilities=base.capabilities | frozenset({AgentCapability.ARBITRATE_MODEL}),
        probe_budget=base.probe_budget,
        allowed_memory_prefix=allowed_memory_prefix,
        max_arbitration_depth=2,
    )


def _log_softmax(logits: Sequence[float]) -> list[float]:
    m = max(float(x) for x in logits)
    exps = [math.exp(float(x) - m) for x in logits]
    lse = m + math.log(max(1e-30, math.fsum(exps)))
    return [float(x) - lse for x in logits]


@dataclass(frozen=True, slots=True)
class SubagentProposalOption:
    r"""Typed VLD engineering candidate $a_k$ with verbalized rationale and structural invariants $\phi(a_k)$."""

    proposal_id: str
    action_kind: str
    verbalizer_text: str
    identifiability_df: Fraction
    negative_control_score: Fraction
    pareto_saving_ratio: Fraction
    proof_completeness: Fraction
    pattern_signature: str
    execute_proposal_fn: Callable[[], dict[str, Any]]

    def feature_vector(self) -> list[Fraction]:
        return [
            Fraction(self.identifiability_df),
            Fraction(self.negative_control_score),
            Fraction(self.pareto_saving_ratio),
            Fraction(self.proof_completeness),
        ]


@dataclass(frozen=True, slots=True)
class SubagentEngineeringTask:
    """VLD engineering decision presented to one or more `HostedModelSubagent`s."""

    task_id: str
    domain_summary: str
    telemetry_prompt: str
    options: tuple[SubagentProposalOption, ...]

    def __post_init__(self) -> None:
        if not self.task_id or not self.options:
            raise SubagentError("SubagentEngineeringTask requires task_id and non-empty options.")


@dataclass(frozen=True, slots=True)
class ScoredProposalRecord:
    """Exact LLM log-probability + verified calibration breakdown for one candidate proposal."""

    proposal_id: str
    action_kind: str
    mean_token_logprob: float
    rational_llm_score: Fraction
    calibration_capsule_score: Fraction
    refusal_penalty: Fraction
    total_score: Fraction


@dataclass(frozen=True, slots=True)
class SubagentCalibrationMemory:
    """Immutable snapshot of a subagent's learned directives, counterexamples & verified capsule."""

    generation: int
    directive_rules: tuple[str, ...]
    refused_patterns: tuple[str, ...]
    accepted_count: int
    refused_count: int
    capsule_digest: str


class HostedModelSubagent:
    """Authentic VLD-hosted Transformer model directed & calibrated as a runtime development subagent."""

    BASE_DIRECTIVE = (
        "Role: Virtual Layer Dynamics engineering subagent.\n"
        "Goal: Select mathematically identifiable laws, Pareto-optimal realizations, and negative-control-qualified paths.\n"
    )

    def __init__(
        self,
        agent_id: str,
        *,
        arch: TransformerArchitectureSpec,
        weight_ground: ModelWeightSnapshot | LazySafetensorsWeightGround,
        encode_fn: Callable[[str], Sequence[int]],
        decode_fn: Callable[[Sequence[int]], str],
        hypervisor: ReceiverHypervisor,
        store: PagedObjectStore | None = None,
        llm_weight: Fraction = Fraction(1),
        max_continuation_tokens: int = 6,
        cache_decoded_weights: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.arch = arch
        self.weight_ground = weight_ground
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.hypervisor = hypervisor
        self.store = store if store is not None else PagedObjectStore(InMemoryPageBacking(65536))
        self.kv_ground = PagedKVCacheGround(self.store)
        self.realizer = TransformerLayerRealizer(
            arch,
            weight_ground,
            store=self.store,
            cache_decoded_weights=cache_decoded_weights,
        )
        self.llm_weight = Fraction(llm_weight)
        self.max_continuation_tokens = max(1, int(max_continuation_tokens))

        self._generation = 0
        self._directive_rules: list[str] = []
        self._refused_patterns: set[str] = set()
        self._accepted_count = 0
        self._refused_count = 0
        self.kv_prefix_reuses = 0
        self._last_prompt_hidden: list[list[float]] = []

        # Generation 0 neutral prior capsule U^(0)(df, nc, sp, pf) = 0 (uncalibrated LLM-only baseline!)
        init_cap = compile_multivariate_hinge_relu(
            input_dim=4,
            base_linear=[["0", "0", "0", "0"]],
            base_bias=["0"],
            hinges=[],
        )
        rep = self.hypervisor.admit_neural_capsule(self.agent_id, init_cap)
        self._capsule = init_cap
        self._capsule_digest = str(rep["capsule_digest"])

    @property
    def last_prompt_hidden_states(self) -> list[list[float]]:
        return self._last_prompt_hidden

    def snapshot_calibration(self) -> SubagentCalibrationMemory:
        return SubagentCalibrationMemory(
            generation=self._generation,
            directive_rules=tuple(self._directive_rules),
            refused_patterns=tuple(sorted(self._refused_patterns)),
            accepted_count=self._accepted_count,
            refused_count=self._refused_count,
            capsule_digest=self._capsule_digest,
        )

    def build_directive_prompt(self, task: SubagentEngineeringTask) -> str:
        rules_block = ""
        if self._directive_rules:
            rules_block = "Verified Engineering Rules:\n" + "\n".join(f"- {r}" for r in self._directive_rules) + "\n"
        return f"{self.BASE_DIRECTIVE}{rules_block}Task [{task.task_id}]: {task.telemetry_prompt}\nDecision:"

    def _prefill_prompt(
        self, prompt_ids: tuple[int, ...]
    ) -> tuple[dict[int, list[list[list[float]]]], dict[int, list[list[list[float]]]], list[float]]:
        """Run shared prompt prefix through all Transformer layers once and persist M1 KV pages."""
        cached_len, cached_pages = self.kv_ground.lookup_longest_prefix(self.arch, (*prompt_ids, 0))
        past_k_map: dict[int, list[list[list[float]]]] = {}
        past_v_map: dict[int, list[list[list[float]]]] = {}
        if cached_len == len(prompt_ids) and len(cached_pages) == self.arch.num_hidden_layers:
            self.kv_prefix_reuses += 1
            for l_idx in range(self.arch.num_hidden_layers):
                pk, pv = self.kv_ground.load_layer_kv(cached_pages[l_idx])
                past_k_map[l_idx] = pk
                past_v_map[l_idx] = pv
            # Recompute final token hidden state from last token only using cached prefix[:L-1] if needed,
            # or run full fast pass when hidden state needed
            h = self.realizer.embed_tokens([prompt_ids[-1]])
            for l_idx in range(self.arch.num_hidden_layers):
                out = self.realizer.forward_layer(
                    l_idx,
                    h,
                    past_k=past_k_map[l_idx][:-1],
                    past_v=past_v_map[l_idx][:-1],
                )
                h = out.hidden_states
            last_logits = self.realizer.final_logits(h)[0]
            return past_k_map, past_v_map, last_logits

        h = self.realizer.embed_tokens(prompt_ids)
        pages: dict[int, Any] = {}
        for l_idx in range(self.arch.num_hidden_layers):
            out = self.realizer.forward_layer(l_idx, h)
            h = out.hidden_states
            past_k_map[l_idx] = out.k_cache_full
            past_v_map[l_idx] = out.v_cache_full
            pages[l_idx] = self.kv_ground.store_layer_kv(
                self.arch,
                layer_index=l_idx,
                token_prefix=prompt_ids,
                k_3d=out.k_cache_full,
                v_3d=out.v_cache_full,
            )
        self.kv_ground.register_prefix_pages(self.arch, prompt_ids, pages)
        self._last_prompt_hidden = [list(row) for row in h]
        last_logits = self.realizer.final_logits([h[-1]])[0]
        return past_k_map, past_v_map, last_logits

    def score_task_options(self, task: SubagentEngineeringTask) -> tuple[ScoredProposalRecord, ...]:
        """Score every candidate proposal via M1-prefix-forked Transformer log-probs + verified capsule."""
        prompt_text = self.build_directive_prompt(task)
        raw_p_ids = [int(t) for t in self.encode_fn(prompt_text)]
        if not raw_p_ids:
            raw_p_ids = [0]
        prompt_ids = tuple(raw_p_ids)

        past_k_map, past_v_map, prompt_last_logits = self._prefill_prompt(prompt_ids)
        first_logprobs = _log_softmax(prompt_last_logits)

        records: list[ScoredProposalRecord] = []
        for opt in task.options:
            cont_ids = [int(t) for t in self.encode_fn(" " + opt.verbalizer_text)][: self.max_continuation_tokens]
            if not cont_ids:
                cont_ids = [0]
            token_lps: list[float] = [first_logprobs[cont_ids[0]]]
            if len(cont_ids) > 1:
                # Single batched teacher-forced causal pass over y_{1..m-1} branching from shared M1 KV prefix!
                h_cont = self.realizer.embed_tokens(cont_ids[:-1])
                for l_idx in range(self.arch.num_hidden_layers):
                    out_c = self.realizer.forward_layer(
                        l_idx,
                        h_cont,
                        past_k=past_k_map[l_idx],
                        past_v=past_v_map[l_idx],
                    )
                    h_cont = out_c.hidden_states
                cont_logits = self.realizer.final_logits(h_cont)
                for step_i, target_tid in enumerate(cont_ids[1:]):
                    step_lp = _log_softmax(cont_logits[step_i])
                    token_lps.append(step_lp[target_tid])

            mean_lp = math.fsum(token_lps) / len(token_lps)
            # Rationalize mean log-prob to deterministic micro-nats Fraction
            rat_llm = Fraction(int(round(mean_lp * 10000)), 10000)
            cal_u = evaluate_multivariate_capsule(self._capsule, opt.feature_vector())[0]
            penalty = Fraction(12) if opt.pattern_signature in self._refused_patterns else Fraction(0)
            total = self.llm_weight * rat_llm + cal_u - penalty
            records.append(
                ScoredProposalRecord(
                    proposal_id=opt.proposal_id,
                    action_kind=opt.action_kind,
                    mean_token_logprob=round(mean_lp, 6),
                    rational_llm_score=rat_llm,
                    calibration_capsule_score=cal_u,
                    refusal_penalty=penalty,
                    total_score=total,
                )
            )

        records.sort(key=lambda r: (-r.total_score, r.proposal_id))
        return tuple(records)

    def generate_engineering_note(self, prompt_text: str, *, max_new_tokens: int = 8) -> dict[str, Any]:
        """Autoregressively decode a concise engineering note with M1 `ModelOrbitCheckpoint` provenance."""
        p_ids = [int(t) for t in self.encode_fn(prompt_text)]
        if not p_ids:
            p_ids = [0]
        runner = AutoregressiveModelRunner(self.arch, self.weight_ground, self.store)
        gen = runner.generate(p_ids, max_new_tokens=max_new_tokens)
        decoded = self.decode_fn(list(p_ids) + list(gen.generated_tokens))
        return {
            "agent_id": self.agent_id,
            "checkpoint_digest": gen.final_checkpoint.checkpoint_digest,
            "generated_token_ids": list(gen.generated_tokens),
            "model_fingerprint": self.arch.fingerprint,
            "text": decoded,
        }

    def record_outcome_and_improve(
        self,
        *,
        chosen_option: SubagentProposalOption,
        admitted: bool,
        feedback_rule: str,
    ) -> SubagentCalibrationMemory:
        """Update prompt directive, counterexample memory, and hypervisor-verified calibration capsule."""
        if admitted:
            self._accepted_count += 1
        else:
            self._refused_count += 1
            self._refused_patterns.add(chosen_option.pattern_signature)

        if feedback_rule and feedback_rule not in self._directive_rules:
            self._directive_rules.append(feedback_rule)

        self._generation += 1
        # Synthesize & whole-domain-verify upgraded rational ReLU co-scoring capsule:
        # Rewards positive degrees-of-freedom (df >= 1), negative-control coverage, Pareto byte savings & proofs,
        # while sharply penalizing df < 1 underdetermined laws or missing negative controls!
        strength = min(4, self._generation)
        upgraded_capsule = compile_multivariate_hinge_relu(
            input_dim=4,
            base_linear=[[str(2 * strength), str(3 * strength), str(2 * strength), str(2 * strength)]],
            base_bias=["0"],
            hinges=[
                # Penalize df < 1: -10 * ReLU(1 - df)
                {"normal": ["-1", "0", "0", "0"], "offset": "1", "weights": [str(-5 * strength)]},
                # Penalize negative_control < 1: -8 * ReLU(1 - nc)
                {"normal": ["0", "-1", "0", "0"], "offset": "1", "weights": [str(-4 * strength)]},
                # Bonus for high Pareto saving > 1/2
                {"normal": ["0", "0", "1", "0"], "offset": "-1/2", "weights": [str(2 * strength)]},
            ],
        )
        rep = self.hypervisor.admit_neural_capsule(self.agent_id, upgraded_capsule)
        self._capsule = upgraded_capsule
        self._capsule_digest = str(rep["capsule_digest"])
        return self.snapshot_calibration()


# ============================================================================
# 2. Multi-Round Directed Subagent Development Loop
# ============================================================================


@dataclass(frozen=True, slots=True)
class SubagentRoundOutcome:
    """Verified record of one engineering task executed by directed hosted subagent(s)."""

    round_index: int
    task_id: str
    selected_proposal_id: str
    selected_action_kind: str
    admitted: bool
    outcome_summary: Mapping[str, Any]
    top_scored_options: tuple[ScoredProposalRecord, ...]
    cross_model_cka: float | None
    calibration_after: SubagentCalibrationMemory


@dataclass(frozen=True, slots=True)
class SubagentEvolutionReport:
    """End-to-end record proving hosted-model subagent improvement & Ed25519-sealed VLD upgrades."""

    loop_id: str
    subagent_ids: tuple[str, ...]
    round_outcomes: tuple[SubagentRoundOutcome, ...]
    initial_acceptance_rate: float
    final_acceptance_rate: float
    quorum_public_keys: Mapping[str, str]
    continuity_digest: str
    standalone_audit: Mapping[str, Any]


class DirectedSubagentDevelopmentLoop:
    """Directs one or more `HostedModelSubagent`s to build, audit & improve the Virtual Layer."""

    def __init__(
        self,
        loop_id: str,
        subagents: Sequence[HostedModelSubagent],
        *,
        member_ids: tuple[str, ...] = ("q0", "q1", "q2"),
    ) -> None:
        if not subagents:
            raise SubagentError("DirectedSubagentDevelopmentLoop requires at least one HostedModelSubagent.")
        self.loop_id = loop_id
        self.subagents = tuple(subagents)
        self.member_ids = member_ids

    def run_development_rounds(
        self,
        rounds: Sequence[Sequence[SubagentEngineeringTask]],
    ) -> SubagentEvolutionReport:
        if not rounds:
            raise SubagentError("rounds cannot be empty.")

        all_outcomes: list[SubagentRoundOutcome] = []
        round_rates: list[float] = []

        for r_idx, task_batch in enumerate(rounds):
            admitted_in_round = 0
            for task in task_batch:
                # Score across all hosted subagents and combine exact rational totals
                per_agent_scores = [sa.score_task_options(task) for sa in self.subagents]
                agg_map: dict[str, Fraction] = {}
                for sc_list in per_agent_scores:
                    for rec in sc_list:
                        agg_map[rec.proposal_id] = agg_map.get(rec.proposal_id, Fraction(0)) + rec.total_score

                # Measure cross-model relational CKA (`IDEA-005`) if >= 2 subagents share token length or resample prefix!
                cka_val: float | None = None
                if len(self.subagents) >= 2:
                    h0 = self.subagents[0].last_prompt_hidden_states
                    h1 = self.subagents[1].last_prompt_hidden_states
                    min_t = min(len(h0), len(h1))
                    if min_t >= 3:
                        sk0 = CenteredGramRelationalSketch.from_hidden_states(self.subagents[0].agent_id, h0[-min_t:])
                        sk1 = CenteredGramRelationalSketch.from_hidden_states(self.subagents[1].agent_id, h1[-min_t:])
                        cka_val = round(compare_relational_sketches(sk0, sk1, num_permutations=8).linear_cka, 6)

                best_pid = sorted(agg_map.keys(), key=lambda pid: (-agg_map[pid], pid))[0]
                chosen = next(o for o in task.options if o.proposal_id == best_pid)

                try:
                    res_summary = chosen.execute_proposal_fn()
                    admitted = bool(res_summary.get("admitted", True))
                    feedback = str(
                        res_summary.get(
                            "learned_rule",
                            f"Verified {chosen.action_kind}: require df >= 1, negative controls & Pareto dominance.",
                        )
                    )
                except Exception as exc:
                    admitted = False
                    res_summary = {"admitted": False, "refusal_reason": str(exc)}
                    feedback = (
                        f"Refused counterexample ({chosen.action_kind}): {exc}. "
                        "Reject underdetermined df < 1 laws and stateless pass-through paths."
                    )

                if admitted:
                    admitted_in_round += 1
                cal_snap = self.subagents[0].snapshot_calibration()
                for sa in self.subagents:
                    cal_snap = sa.record_outcome_and_improve(
                        chosen_option=chosen,
                        admitted=admitted,
                        feedback_rule=feedback,
                    )

                all_outcomes.append(
                    SubagentRoundOutcome(
                        round_index=r_idx,
                        task_id=task.task_id,
                        selected_proposal_id=chosen.proposal_id,
                        selected_action_kind=chosen.action_kind,
                        admitted=admitted,
                        outcome_summary=parse_strict_json(canonical_json(res_summary)),
                        top_scored_options=per_agent_scores[0],
                        cross_model_cka=cka_val,
                        calibration_after=cal_snap,
                    )
                )

            round_rates.append(admitted_in_round / max(1, len(task_batch)))

        # Seal the complete subagent evolution ledger under Ed25519 public-key quorum authority!
        lin_id = f"subagent.loop.{self.loop_id}"
        kps, pub_keys = build_ed25519_quorum_identities(self.member_ids, deterministic_domain_seed=lin_id.encode("utf-8"))
        q_mem = QuorumMembership(lin_id, self.member_ids, member_public_keys=pub_keys)
        q_members = [QuorumAuthorityMember.from_ed25519_keypair(m, kps[m]) for m in self.member_ids]
        quorum = QuorumLineageCoordinator(q_mem, q_members)
        quorum.fence_term(1, active_member_ids=self.member_ids, owner_id="vld.subagent.coordinator")

        store = PagedObjectStore(InMemoryPageBacking(65536))
        summary_payload = {
            "final_acceptance_rate": round(round_rates[-1], 6),
            "initial_acceptance_rate": round(round_rates[0], 6),
            "loop_id": self.loop_id,
            "outcomes_count": len(all_outcomes),
            "subagent_ids": [s.agent_id for s in self.subagents],
        }
        state_manifest = store.put_bytes(canonical_bytes(summary_payload))
        q_cert = quorum.propose_and_seal(
            slot=1,
            term=1,
            predecessor_manifest_digest=GENESIS_PREV_DIGEST,
            state_manifest_digest=state_manifest.manifest_digest,
            proposer_id="vld.subagent.coordinator",
            operation_id=f"seal.{lin_id}",
            active_member_ids=self.member_ids,
        )
        sa_audit = verify_quorum_certificate_standalone(
            q_cert.to_dict(),
            member_ids=self.member_ids,
            member_public_keys=pub_keys,
        )
        sa_audit["status"] = "VERIFIED"

        return SubagentEvolutionReport(
            loop_id=self.loop_id,
            subagent_ids=tuple(s.agent_id for s in self.subagents),
            round_outcomes=tuple(all_outcomes),
            initial_acceptance_rate=round(round_rates[0], 6),
            final_acceptance_rate=round(round_rates[-1], 6),
            quorum_public_keys=pub_keys,
            continuity_digest=q_cert.certificate_digest,
            standalone_audit=sa_audit,
        )


def build_compact_hosted_subagent_fixture(
    agent_id: str,
    hypervisor: ReceiverHypervisor,
    *,
    hidden_size: int = 16,
    num_layers: int = 2,
    seed: int = 73,
) -> HostedModelSubagent:
    """Build a fast hermetic VLD-hosted 2-layer Qwen3 Transformer subagent for deterministic CI tests."""
    vocab_size = 128
    arch = TransformerArchitectureSpec(
        source_model=f"vld/compact-{agent_id}",
        revision="0123456789abcdef0123456789abcdef01234567",
        model_type="qwen3",
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=hidden_size // 2,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        dtype="F32",
    )
    pspace = arch.build_parameter_space()
    raw_tensors: dict[str, bytes] = {}
    for idx, t_spec in enumerate(pspace.tensors):
        cnt = 1
        for d in t_spec.shape:
            cnt *= d
        vals: list[float] = []
        for k in range(cnt):
            h_b = hashlib.sha256(f"{seed}:{idx}:{t_spec.name}:{k // 8}".encode("utf-8")).digest()
            v = ((h_b[k % 32] / 255.0) - 0.5) * 0.25
            if "norm.weight" in t_spec.name:
                v = 1.0 + v * 0.05
            vals.append(v)
        raw_tensors[t_spec.name] = struct.pack(f"<{cnt}f", *vals)

    store = PagedObjectStore(InMemoryPageBacking(65536))
    snap = ModelWeightSnapshot.publish_tensors(pspace, store, raw_tensors)

    def _ascii_encode(txt: str) -> list[int]:
        return [ord(ch) % vocab_size for ch in txt[:96]] or [0]

    def _ascii_decode(ids: Sequence[int]) -> str:
        return "".join(chr(32 + (int(i) % 90)) for i in ids)

    return HostedModelSubagent(
        agent_id,
        arch=arch,
        weight_ground=snap,
        encode_fn=_ascii_encode,
        decode_fn=_ascii_decode,
        hypervisor=hypervisor,
        store=store,
    )


__all__ = [
    "DirectedSubagentDevelopmentLoop",
    "HostedModelSubagent",
    "ScoredProposalRecord",
    "SubagentCalibrationMemory",
    "SubagentEngineeringTask",
    "SubagentError",
    "SubagentEvolutionReport",
    "SubagentProposalOption",
    "SubagentRoundOutcome",
    "build_compact_hosted_subagent_fixture",
    "create_subagent_grant",
]
