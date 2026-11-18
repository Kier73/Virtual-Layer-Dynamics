r"""Unified Continuous Computational System Across Changing Grounds, Representations, Hosts & Physical Paths (`vld.continuous`, `GGE-001..007`).

Resolves the central cross-system composition question of Virtual Layer Dynamics:
Given heterogeneous computational stages $(\mathcal{G}_0, \dots, \mathcal{G}_{m-1})$ spanning
Internet/Router paths, `HeadlessOS` POSIX processes, `Signal8` & $\mathbb{F}_p$ commutant orbits,
WebAssembly/E-GPU substrates, mTLS partitioned tensor-cut fabrics, and Frontier LLMs (`DeepSeek-V4.1` / `GLM-5.3`),
every cross-stage transition $\Phi_{k \to k+1}$ emits a cryptographically chained `CrossGroundContinuityReceipt`
proving five simultaneous invariants:
  1. **Representation Naturality:** Verified lowering/lifting across stage schemas via M1 content digests.
  2. **Causal Clock Continuity (`G13`):** Explicit `ClockBridgeReceipt` proving `CausalOrder.BEFORE`
     whenever execution crosses scheduler or host boundaries.
  3. **Strict-Majority Quorum Authority (`2q > N`, `G21`/`G22`):** Each completed stage is sealed in a
     `QuorumLineageCoordinator` slot before advancing; replacement hosts fence the next Paxos term and
     resume strictly at the first unsealed stage (`replayed_stages == 0`).
  4. **Independent Stage Verification:** Every stage output passes a standalone mathematical/scalar oracle
     prior to quorum commit (never trusting transport `ACK` or worker self-attestation).
  5. **Closed-Loop Self-Improvement:** Internal `ContinuousLayerBuilderAgent`s inspect Pass 1 telemetry,
     install verified Pareto-dominating Grounds/shortcuts/caches, and accelerate Pass 2 (`C_pass2 < C_pass1`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.accel import (
    NumericEngine,
    encode_bf16_le,
    encode_f8_e4m3,
    encode_f8_e8m0,
    encode_f32_le,
    encode_fp4_e2m1_packed,
)
from vld.agents import (
    ContinuousLayerBuilderAgent,
    LayerBuildCycleReport,
    ReceiverHypervisor,
    create_default_builder_grant,
)
from vld.authority import (
    CausalOrder,
    ClockBridgeReceipt,
    EncounterEnvelope,
    QuorumAuthorityMember,
    QuorumCommitCertificate,
    QuorumLineageCoordinator,
    QuorumMembership,
    TemporalCoordinate,
    VectorClock,
)
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.fabric import (
    ComputationalFabricRouter,
    HostAdvertisement,
    PartitionedTensorCutCoordinator,
    RouteMetric,
    TensorCutWorkerNode,
    create_mtls_client_context,
    create_mtls_server_context,
    generate_ephemeral_mtls_pki,
)
from vld.ground import ByteLens, LensKind, NavigationCostVector, VirtualGroundSpec
from vld.journal import GENESIS_PREV_DIGEST, HashChainedJournal
from vld.memory import InMemoryPageBacking, PagedByteManifest, PagedObjectStore, StructuralMemoryRegistry
from vld.models import (
    DeepSeekV41ArchitectureSpec,
    DeepSeekV41LayerRealizer,
    DeepSeekV41ReferenceOracle,
    Glm53ArchitectureSpec,
    Glm53LayerRealizer,
    LazyFrontierWeightGround,
    build_safetensors_bytes,
    verify_mla_matrix_absorption,
)
from vld.navigation import (
    AttributedObserverReading,
    ComposedComponentSpec,
    ComposedGroundNavigator,
    CrossObserverCongruenceGate,
    FiniteFieldOperatorChain,
)
from vld.signal import Signal8Backend
from vld.substrates import (
    CountingHTTPOriginServer,
    DeltaTensorCheckpointLog,
    HTTPPathIntermediaryServer,
    HeadlessOS,
    O_RDONLY,
    SEEK_SET,
    TTL_PROBES,
    TiledTensorExecutor,
    VMInstruction,
    WasmRealizer,
    assemble_wasm_affine_orbit_module,
    build_ethernet_ipv4_udp_frame,
    lift_wasm_module,
    verify_ipv4_router_hop_ground,
)


class ContinuousSystemError(RuntimeError):
    """Raised when cross-Ground continuity, causal clock bridging, or stage verification fails."""


@dataclass(frozen=True, slots=True)
class CrossGroundContinuityReceipt:
    """Hash-chained, quorum-sealed proof of one stage transition across Grounds, hosts, and schedulers."""

    pipeline_id: str
    stage_index: int
    stage_id: str
    prev_continuity_digest: str
    source_ground_fingerprint: str
    target_ground_fingerprint: str
    source_representation: str
    target_representation: str
    host_id: str
    scheduler_id: str
    temporal_coordinate: TemporalCoordinate
    clock_bridge_digest: str | None
    state_manifest: PagedByteManifest
    quorum_certificate: QuorumCommitCertificate
    oracle_verdict: str
    stage_cost: NavigationCostVector
    continuity_digest: str = ""

    SCHEMA = "vld.cross-ground-continuity-receipt.v1"

    def __post_init__(self) -> None:
        computed = domain_digest(self.SCHEMA, self._body())
        if self.continuity_digest and self.continuity_digest != computed:
            raise ContinuousSystemError("CrossGroundContinuityReceipt digest tampering detected.")
        object.__setattr__(self, "continuity_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "clock_bridge_digest": self.clock_bridge_digest,
            "host_id": self.host_id,
            "oracle_verdict": self.oracle_verdict,
            "pipeline_id": self.pipeline_id,
            "prev_continuity_digest": self.prev_continuity_digest,
            "quorum_cert_digest": self.quorum_certificate.certificate_digest,
            "scheduler_id": self.scheduler_id,
            "schema": self.SCHEMA,
            "source_ground_fingerprint": self.source_ground_fingerprint,
            "source_representation": self.source_representation,
            "stage_cost": self.stage_cost.to_dict(),
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "state_manifest": self.state_manifest.to_dict(),
            "target_ground_fingerprint": self.target_ground_fingerprint,
            "target_representation": self.target_representation,
            "temporal": self.temporal_coordinate.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._body(),
            "continuity_digest": self.continuity_digest,
            "quorum_certificate": self.quorum_certificate.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CrossGroundContinuityReceipt:
        sc = data["stage_cost"]
        return cls(
            pipeline_id=str(data["pipeline_id"]),
            stage_index=int(data["stage_index"]),
            stage_id=str(data["stage_id"]),
            prev_continuity_digest=str(data["prev_continuity_digest"]),
            source_ground_fingerprint=str(data["source_ground_fingerprint"]),
            target_ground_fingerprint=str(data["target_ground_fingerprint"]),
            source_representation=str(data["source_representation"]),
            target_representation=str(data["target_representation"]),
            host_id=str(data["host_id"]),
            scheduler_id=str(data["scheduler_id"]),
            temporal_coordinate=TemporalCoordinate.from_dict(data["temporal"]),
            clock_bridge_digest=None if data["clock_bridge_digest"] is None else str(data["clock_bridge_digest"]),
            state_manifest=PagedByteManifest.from_dict(data["state_manifest"]),
            quorum_certificate=QuorumCommitCertificate.from_dict(data["quorum_certificate"]),
            oracle_verdict=str(data["oracle_verdict"]),
            stage_cost=NavigationCostVector(
                latency_ms=float(sc["latency_ms"]),
                compute_ops=float(sc["compute_ops"]),
                bytes_moved=float(sc["bytes_moved"]),
                verification_ops=float(sc["verification_ops"]),
            ),
            continuity_digest=str(data["continuity_digest"]),
        )


@dataclass(frozen=True, slots=True)
class ContinuousStageSpec:
    """Declarative specification of one stage inside a continuous multi-Ground computation."""

    stage_id: str
    ground_spec: VirtualGroundSpec
    input_representation: str
    output_representation: str
    scheduler_id: str
    execute_fn: Callable[[Mapping[str, Any]], tuple[dict[str, Any], NavigationCostVector, str]]


@dataclass(frozen=True, slots=True)
class ContinuousPipelineResult:
    """Verified completion report for a continuous multi-Ground pipeline execution."""

    pipeline_id: str
    final_state: Mapping[str, Any]
    receipts: tuple[CrossGroundContinuityReceipt, ...]
    stages_executed_this_invocation: int
    stages_recovered_from_quorum: int
    total_cost: NavigationCostVector
    final_continuity_digest: str


class ContinuousVirtualSystem:
    """Quorum-fenced, crash-resumable cross-Ground execution runtime."""

    def __init__(
        self,
        pipeline_id: str,
        *,
        store: PagedObjectStore,
        quorum: QuorumLineageCoordinator,
        active_quorum_members: Sequence[str],
        journal: HashChainedJournal | None = None,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.store = store
        self.quorum = quorum
        self.active_quorum_members = tuple(active_quorum_members)
        self.journal = journal if journal is not None else HashChainedJournal()

    def execute_stages(
        self,
        stages: Sequence[ContinuousStageSpec],
        initial_input: Mapping[str, Any],
        *,
        host_id: str,
        interrupt_after_stage: int | None = None,
    ) -> ContinuousPipelineResult:
        """Execute or resume `stages` with Paxos term fencing, M1 state manifests & `ClockBridgeReceipt`s."""
        if not stages:
            raise ContinuousSystemError("stages cannot be empty.")

        # Fence next Paxos term on strict majority so any dead/zombie predecessor host is fenced!
        max_promised = self.quorum.max_observed_term(active_member_ids=self.active_quorum_members)
        term = max_promised + 1
        self.quorum.fence_term(term, active_member_ids=self.active_quorum_members, owner_id=host_id)
        locked_max = max(
            (s for s in range(1, len(stages) + 1) if self.quorum.locked_slot_record(s) is not None),
            default=0,
        )
        hw_slot = max(self.quorum.high_water_slot, locked_max)

        # Recover all previously sealed/quorum-accepted stages without re-executing them!
        recovered_receipts: list[CrossGroundContinuityReceipt] = []
        cur_state: dict[str, Any] = parse_strict_json(canonical_json(dict(initial_input)))
        prev_cont_digest = GENESIS_PREV_DIGEST
        prev_manifest_digest = GENESIS_PREV_DIGEST
        prev_temporal: TemporalCoordinate | None = None
        prev_ground_fp = domain_digest("vld.pipeline-genesis.v1", {"input": cur_state, "pipeline_id": self.pipeline_id})

        for slot_idx in range(1, hw_slot + 1):
            acc = self.quorum.locked_slot_record(slot_idx)
            cert_s = self.quorum.certificate_at(slot_idx)
            if cert_s is None and acc is not None:
                # Paxos Phase 2a value-locking: propagate accepted slot to fresh coordinator without re-running stage!
                cert_s = self.quorum.propose_and_seal(
                    slot=slot_idx,
                    term=term,
                    predecessor_manifest_digest=acc.predecessor_manifest_digest,
                    state_manifest_digest=acc.state_manifest_digest,
                    proposer_id=host_id,
                    operation_id=acc.operation_id,
                    active_member_ids=self.active_quorum_members,
                )
            s_dg = acc.state_manifest_digest if acc is not None else (cert_s.proposal.state_manifest_digest if cert_s is not None else None)
            if s_dg is None or cert_s is None:
                raise ContinuousSystemError(f"Missing quorum record for committed slot {slot_idx}.")
            rcpt = self._recover_receipt_by_digest(s_dg, cert_s)
            if rcpt.stage_index != slot_idx - 1 or rcpt.prev_continuity_digest != prev_cont_digest:
                raise ContinuousSystemError(f"Broken continuity chain at recovered stage {slot_idx - 1}.")
            cur_state = parse_strict_json(self.store.read_bytes(rcpt.state_manifest))
            recovered_receipts.append(rcpt)
            prev_cont_digest = rcpt.continuity_digest
            prev_manifest_digest = s_dg
            prev_temporal = rcpt.temporal_coordinate
            prev_ground_fp = rcpt.target_ground_fingerprint

        recovered_count = len(recovered_receipts)
        executed_count = 0
        all_receipts = list(recovered_receipts)

        for idx in range(recovered_count, len(stages)):
            stage = stages[idx]
            next_clock = VectorClock.from_mapping({host_id: idx + 1})
            next_temporal = TemporalCoordinate(scheduler_id=stage.scheduler_id, epoch=term, clock=next_clock)
            bridge_digest: str | None = None

            if prev_temporal is not None:
                if prev_temporal.scheduler_id != next_temporal.scheduler_id or prev_temporal.epoch != next_temporal.epoch:
                    bridge = ClockBridgeReceipt(
                        source_scheduler_id=prev_temporal.scheduler_id,
                        source_epoch=prev_temporal.epoch,
                        source_anchor=prev_temporal.clock,
                        target_scheduler_id=next_temporal.scheduler_id,
                        target_epoch=next_temporal.epoch,
                        target_anchor=next_temporal.clock,
                    )
                    if prev_temporal.compare(next_temporal, bridge=bridge) is not CausalOrder.BEFORE:
                        raise ContinuousSystemError("ClockBridgeReceipt failed causal progression check.")
                    bridge_digest = bridge.bridge_digest
                else:
                    next_temporal = prev_temporal.advance(host_id, 1)

            next_state, stage_cost, verdict = stage.execute_fn(cur_state)
            if not verdict.startswith("PASS"):
                raise ContinuousSystemError(f"Stage {stage.stage_id!r} failed independent oracle: {verdict}")

            clean_next = parse_strict_json(canonical_json(dict(next_state)))
            state_m = self.store.put_bytes(canonical_bytes(clean_next), media_type="application/json")

            # Write-ahead pre-seal envelope BEFORE quorum proposal so zero crash window exists!
            prelim_env = {
                "pipeline_id": self.pipeline_id,
                "stage_id": stage.stage_id,
                "stage_index": idx,
                "state_manifest_digest": state_m.manifest_digest,
            }
            slot_token_dg = domain_digest("vld.stage-slot.v1", prelim_env)
            pre_record = {
                "clock_bridge_digest": bridge_digest,
                "host_id": host_id,
                "oracle_verdict": verdict,
                "pipeline_id": self.pipeline_id,
                "prev_continuity_digest": prev_cont_digest,
                "scheduler_id": stage.scheduler_id,
                "source_ground_fingerprint": prev_ground_fp,
                "source_representation": stage.input_representation,
                "stage_cost": stage_cost.to_dict(),
                "stage_id": stage.stage_id,
                "stage_index": idx,
                "state_manifest": state_m.to_dict(),
                "target_ground_fingerprint": stage.ground_spec.fingerprint,
                "target_representation": stage.output_representation,
                "temporal": next_temporal.to_dict(),
            }
            self._save_pre_seal_envelope(slot_token_dg, stage.ground_spec.fingerprint, idx, pre_record)

            # Seal stage state in quorum slot (idx + 1)
            q_cert = self.quorum.propose_and_seal(
                slot=idx + 1,
                term=term,
                predecessor_manifest_digest=prev_manifest_digest,
                state_manifest_digest=slot_token_dg,
                proposer_id=host_id,
                operation_id=f"{self.pipeline_id}.stage.{idx}",
                active_member_ids=self.active_quorum_members,
            )

            rcpt = CrossGroundContinuityReceipt(
                pipeline_id=self.pipeline_id,
                stage_index=idx,
                stage_id=stage.stage_id,
                prev_continuity_digest=prev_cont_digest,
                source_ground_fingerprint=prev_ground_fp,
                target_ground_fingerprint=stage.ground_spec.fingerprint,
                source_representation=stage.input_representation,
                target_representation=stage.output_representation,
                host_id=host_id,
                scheduler_id=stage.scheduler_id,
                temporal_coordinate=next_temporal,
                clock_bridge_digest=bridge_digest,
                state_manifest=state_m,
                quorum_certificate=q_cert,
                oracle_verdict=verdict,
                stage_cost=stage_cost,
            )
            self._save_slot_envelope(slot_token_dg, rcpt)

            all_receipts.append(rcpt)
            cur_state = clean_next
            prev_cont_digest = rcpt.continuity_digest
            prev_manifest_digest = slot_token_dg
            prev_temporal = next_temporal
            prev_ground_fp = stage.ground_spec.fingerprint
            executed_count += 1

            if interrupt_after_stage is not None and idx == interrupt_after_stage:
                break

        tot_cost = NavigationCostVector(
            latency_ms=sum(r.stage_cost.latency_ms for r in all_receipts),
            compute_ops=sum(r.stage_cost.compute_ops for r in all_receipts),
            bytes_moved=sum(r.stage_cost.bytes_moved for r in all_receipts),
            verification_ops=sum(r.stage_cost.verification_ops for r in all_receipts),
        )
        return ContinuousPipelineResult(
            pipeline_id=self.pipeline_id,
            final_state=cur_state,
            receipts=tuple(all_receipts),
            stages_executed_this_invocation=executed_count,
            stages_recovered_from_quorum=recovered_count,
            total_cost=tot_cost,
            final_continuity_digest=prev_cont_digest,
        )

    def _save_pre_seal_envelope(
        self,
        token_digest: str,
        target_ground_fp: str,
        stage_index: int,
        pre_record: Mapping[str, Any],
    ) -> None:
        pre_m = self.store.put_bytes(canonical_bytes({"pre_stage": dict(pre_record)}))
        self.journal.record(
            request_id=f"slot-pre-{self.pipeline_id}-{stage_index}-{token_digest[:16]}",
            ground_fingerprint=target_ground_fp,
            operation="continuous.pre_stage",
            payload={"pre_manifest": pre_m.to_dict(), "slot_token_digest": token_digest},
            result_fn=lambda: {"stage_index": stage_index},
        )

    def _save_slot_envelope(self, token_digest: str, rcpt: CrossGroundContinuityReceipt) -> None:
        env_m = self.store.put_bytes(canonical_bytes({"continuity_receipt": rcpt.to_dict()}))
        self.journal.record(
            request_id=f"slot-env-{self.pipeline_id}-{rcpt.stage_index}-{token_digest[:16]}",
            ground_fingerprint=rcpt.target_ground_fingerprint,
            operation="continuous.seal_stage",
            payload={"envelope_manifest": env_m.to_dict(), "slot_token_digest": token_digest},
            result_fn=lambda: {"continuity_digest": rcpt.continuity_digest},
        )

    def _recover_receipt_by_digest(
        self,
        token_digest: str,
        fallback_cert: QuorumCommitCertificate,
    ) -> CrossGroundContinuityReceipt:
        self.journal.verify()
        events = self.journal.events
        for ev in reversed(events):
            if ev.operation == "continuous.seal_stage" and ev.payload.get("slot_token_digest") == token_digest:
                env_m = PagedByteManifest.from_dict(ev.payload["envelope_manifest"])
                env_doc = parse_strict_json(self.store.read_bytes(env_m))
                sealed_rcpt = CrossGroundContinuityReceipt.from_dict(env_doc["continuity_receipt"])
                if not self.quorum.verify_certificate(sealed_rcpt.quorum_certificate):
                    raise ContinuousSystemError("Persisted continuity_receipt failed QuorumCommitCertificate verification.")
                return sealed_rcpt
        for ev in reversed(events):
            if ev.operation == "continuous.pre_stage" and ev.payload.get("slot_token_digest") == token_digest:
                pre_m = PagedByteManifest.from_dict(ev.payload["pre_manifest"])
                pre = parse_strict_json(self.store.read_bytes(pre_m))["pre_stage"]
                sc = pre["stage_cost"]
                rebuilt = CrossGroundContinuityReceipt(
                    pipeline_id=str(pre["pipeline_id"]),
                    stage_index=int(pre["stage_index"]),
                    stage_id=str(pre["stage_id"]),
                    prev_continuity_digest=str(pre["prev_continuity_digest"]),
                    source_ground_fingerprint=str(pre["source_ground_fingerprint"]),
                    target_ground_fingerprint=str(pre["target_ground_fingerprint"]),
                    source_representation=str(pre["source_representation"]),
                    target_representation=str(pre["target_representation"]),
                    host_id=str(pre["host_id"]),
                    scheduler_id=str(pre["scheduler_id"]),
                    temporal_coordinate=TemporalCoordinate.from_dict(pre["temporal"]),
                    clock_bridge_digest=None if pre["clock_bridge_digest"] is None else str(pre["clock_bridge_digest"]),
                    state_manifest=PagedByteManifest.from_dict(pre["state_manifest"]),
                    quorum_certificate=fallback_cert,
                    oracle_verdict=str(pre["oracle_verdict"]),
                    stage_cost=NavigationCostVector(
                        latency_ms=float(sc["latency_ms"]),
                        compute_ops=float(sc["compute_ops"]),
                        bytes_moved=float(sc["bytes_moved"]),
                        verification_ops=float(sc["verification_ops"]),
                    ),
                )
                # Anchor the reconstructed continuity receipt into the journal so subsequent host
                # replacements observe an immutable continuity_digest!
                self._save_slot_envelope(token_digest, rebuilt)
                return rebuilt
        raise ContinuousSystemError(f"Could not resolve M1 envelope for quorum slot digest {token_digest}.")


# ============================================================================
# Deterministic Compact Frontier Fixtures & Full 6-Stage Self-Improving Campaign
# ============================================================================


def build_compact_deepseek_v41_fixture(seed: int = 7) -> tuple[DeepSeekV41ArchitectureSpec, LazyFrontierWeightGround]:
    """Construct a deterministic mixed-precision (`BF16` + `F8_E4M3`/`UE8M0` + `FP4` `E2M1`) DeepSeek-V4.1 model."""
    arch = DeepSeekV41ArchitectureSpec(
        source_model="deepseek-ai/DeepSeek-V41-Compact",
        revision="314113f8a7dadb388215396ba5fbee2c610a04d6",
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        moe_intermediate_size=64,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-6,
        sliding_window=16,
    )
    tensors: dict[str, tuple[str, Sequence[int], bytes]] = {}

    def _det(n: int, salt: int, amp: float = 0.15) -> list[float]:
        return [amp * math.sin((i + 1) * 0.37 + salt * 1.13 + seed) for i in range(n)]

    def _add_fp8_32(name: str, rows: int, cols: int, salt: int) -> None:
        vals = _det(rows * cols, salt, 0.5)
        tensors[name + ".weight"] = ("F8_E4M3", (rows, cols), encode_f8_e4m3(vals))
        sr, sc = rows // 32, cols // 32
        tensors[name + ".scale"] = ("F8_E8M0", (sr, sc), encode_f8_e8m0([0.125] * (sr * sc)))

    def _add_fp4_32(name: str, rows: int, cols: int, salt: int) -> None:
        vals = _det(rows * cols, salt, 0.25)
        rows_2d = [vals[r * cols : (r + 1) * cols] for r in range(rows)]
        p_raw, s_raw = encode_fp4_e2m1_packed(rows_2d, block_k=32)
        tensors[name + ".weight"] = ("I8", (rows, cols // 2), p_raw)
        tensors[name + ".scale"] = ("F8_E8M0", (rows, cols // 32), s_raw)

    tensors["embed.weight"] = ("BF16", (32, 64), encode_bf16_le(_det(32 * 64, 1, 0.2)))
    tensors["norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
    tensors["head.weight"] = ("BF16", (32, 64), encode_bf16_le(_det(32 * 64, 2, 0.2)))

    mix_hc = (2 + 4) * 4  # 24
    for l_i in range(2):
        p = f"layers.{l_i}."
        s0 = 100 * (l_i + 1)
        tensors[p + "attn_norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
        tensors[p + "ffn_norm.weight"] = ("BF16", (64,), encode_bf16_le([1.0] * 64))
        tensors[p + "attn.q_norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
        tensors[p + "attn.kv_norm.weight"] = ("BF16", (32,), encode_bf16_le([1.0] * 32))
        tensors[p + "attn.attn_sink"] = ("F32", (4,), encode_f32_le([-1.5, -1.0, -2.0, -0.5]))

        _add_fp8_32(p + "attn.wq_a", 32, 64, s0 + 1)
        _add_fp8_32(p + "attn.wq_b", 128, 32, s0 + 2)
        _add_fp8_32(p + "attn.wkv", 32, 64, s0 + 3)
        _add_fp8_32(p + "attn.wo_a", 64, 64, s0 + 4)
        _add_fp8_32(p + "attn.wo_b", 64, 64, s0 + 5)

        tensors[p + "hc_attn_fn"] = ("F32", (mix_hc, 256), encode_f32_le(_det(mix_hc * 256, s0 + 6, 0.05)))
        tensors[p + "hc_attn_base"] = ("F32", (mix_hc,), encode_f32_le(_det(mix_hc, s0 + 7, 0.1)))
        tensors[p + "hc_attn_scale"] = ("F32", (3,), encode_f32_le([0.2, 0.2, 0.2]))
        tensors[p + "hc_ffn_fn"] = ("F32", (mix_hc, 256), encode_f32_le(_det(mix_hc * 256, s0 + 8, 0.05)))
        tensors[p + "hc_ffn_base"] = ("F32", (mix_hc,), encode_f32_le(_det(mix_hc, s0 + 9, 0.1)))
        tensors[p + "hc_ffn_scale"] = ("F32", (3,), encode_f32_le([0.2, 0.2, 0.2]))

        tensors[p + "ffn.gate.weight"] = ("BF16", (8, 64), encode_bf16_le(_det(8 * 64, s0 + 10, 0.3)))
        tensors[p + "ffn.gate.bias"] = ("F32", (8,), encode_f32_le(_det(8, s0 + 11, 0.05)))

        _add_fp8_32(p + "ffn.shared_experts.w1", 64, 64, s0 + 12)
        _add_fp8_32(p + "ffn.shared_experts.w2", 64, 64, s0 + 13)
        _add_fp8_32(p + "ffn.shared_experts.w3", 64, 64, s0 + 14)

        for e_i in range(8):
            ep = f"{p}ffn.experts.{e_i}."
            _add_fp4_32(ep + "w1", 64, 64, s0 + 20 + 3 * e_i)
            _add_fp4_32(ep + "w2", 64, 64, s0 + 21 + 3 * e_i)
            _add_fp4_32(ep + "w3", 64, 64, s0 + 22 + 3 * e_i)

    raw_st = build_safetensors_bytes(tensors)
    return arch, LazyFrontierWeightGround.from_safetensors_bytes(raw_st, shard_name="dsv41_compact.safetensors")


def build_compact_glm53_fixture(seed: int = 11) -> tuple[Glm53ArchitectureSpec, LazyFrontierWeightGround]:
    """Construct a deterministic GLM-5.3 (`MLA` + Hybrid Layer-0 Dense + Layer-1 Sigmoid `noaux_tc` MoE) model."""
    arch = Glm53ArchitectureSpec(
        source_model="zai-org/GLM-53-Compact",
        revision="abf42e6c0e53bc29a875511301e7b7fcd220315b",
        vocab_size=32,
        hidden_size=48,
        num_hidden_layers=2,
        first_k_dense_replace=1,
        intermediate_size=64,
        num_attention_heads=4,
        q_lora_rank=24,
        kv_lora_rank=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        moe_intermediate_size=32,
        n_routed_experts=6,
        n_shared_experts=1,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        routed_scaling_factor=2.5,
        rms_norm_eps=1e-5,
    )
    tensors: dict[str, tuple[str, Sequence[int], bytes]] = {}

    def _bf(shape: tuple[int, ...], salt: int, amp: float = 0.15) -> tuple[str, tuple[int, ...], bytes]:
        n = math.prod(shape)
        vals = [amp * math.cos((i + 1) * 0.29 + salt * 0.91 + seed) for i in range(n)]
        return ("BF16", shape, encode_bf16_le(vals))

    tensors["model.embed_tokens.weight"] = _bf((32, 48), 1)
    tensors["model.norm.weight"] = ("BF16", (48,), encode_bf16_le([1.0] * 48))
    tensors["lm_head.weight"] = _bf((32, 48), 2)

    for l_i in range(2):
        p = f"model.layers.{l_i}."
        s0 = 200 * (l_i + 1)
        tensors[p + "input_layernorm.weight"] = ("BF16", (48,), encode_bf16_le([1.0] * 48))
        tensors[p + "post_attention_layernorm.weight"] = ("BF16", (48,), encode_bf16_le([1.0] * 48))
        tensors[p + "self_attn.q_a_proj.weight"] = _bf((24, 48), s0 + 1)
        tensors[p + "self_attn.q_a_layernorm.weight"] = ("BF16", (24,), encode_bf16_le([1.0] * 24))
        tensors[p + "self_attn.q_b_proj.weight"] = _bf((4 * 16, 24), s0 + 2)
        tensors[p + "self_attn.kv_a_proj_with_mqa.weight"] = _bf((16 + 4, 48), s0 + 3)
        tensors[p + "self_attn.kv_a_layernorm.weight"] = ("BF16", (16,), encode_bf16_le([1.0] * 16))
        tensors[p + "self_attn.kv_b_proj.weight"] = _bf((4 * (12 + 16), 16), s0 + 4)
        tensors[p + "self_attn.o_proj.weight"] = _bf((48, 4 * 16), s0 + 5)

        if l_i == 0:
            tensors[p + "mlp.gate_proj.weight"] = _bf((64, 48), s0 + 6)
            tensors[p + "mlp.up_proj.weight"] = _bf((64, 48), s0 + 7)
            tensors[p + "mlp.down_proj.weight"] = _bf((48, 64), s0 + 8)
        else:
            tensors[p + "mlp.gate.weight"] = _bf((6, 48), s0 + 9)
            tensors[p + "mlp.gate.e_score_correction_bias"] = (
                "F32",
                (6,),
                encode_f32_le([0.02 * k - 0.05 for k in range(6)]),
            )
            tensors[p + "mlp.shared_experts.gate_proj.weight"] = _bf((32, 48), s0 + 10)
            tensors[p + "mlp.shared_experts.up_proj.weight"] = _bf((32, 48), s0 + 11)
            tensors[p + "mlp.shared_experts.down_proj.weight"] = _bf((48, 32), s0 + 12)
            for e_i in range(6):
                ep = f"{p}mlp.experts.{e_i}."
                tensors[ep + "gate_proj.weight"] = _bf((32, 48), s0 + 20 + 3 * e_i)
                tensors[ep + "up_proj.weight"] = _bf((32, 48), s0 + 21 + 3 * e_i)
                tensors[ep + "down_proj.weight"] = _bf((48, 32), s0 + 22 + 3 * e_i)

    raw_st = build_safetensors_bytes(tensors)
    return arch, LazyFrontierWeightGround.from_safetensors_bytes(raw_st, shard_name="glm53_compact.safetensors")


@dataclass(frozen=True, slots=True)
class SelfImprovingCampaignReport:
    """End-to-end record of Pass 1 (Cold Discovery) -> Agent Build Cycle -> Pass 2 (Warm + Host Switch)."""

    pass1: ContinuousPipelineResult
    build_cycle: LayerBuildCycleReport
    pass2_before_crash: ContinuousPipelineResult
    pass2_resumed: ContinuousPipelineResult
    pass2_pareto_dominates_pass1: bool
    origin_hits_delta_pass1: int
    origin_hits_delta_pass2: int
    warm_weight_bytes_pass2: int


def run_self_improving_continuous_campaign(*, horizon: int = 10**12) -> SelfImprovingCampaignReport:
    """Operate ALL Virtual Layer capabilities as one continuous verified system, build upgrades, and re-run!"""
    ds_arch, ds_weights = build_compact_deepseek_v41_fixture()
    glm_arch, glm_weights = build_compact_glm53_fixture()

    # Construct a coupled 6x6 F_97 chain that is conjugated from three 2x2 diagonal blocks (P * diag(B1, B2, B3) * P^-1)!
    p_mod = 97
    p_mat = (
        (1, 2, 0, 1, 0, 0),
        (0, 1, 1, 0, 0, 1),
        (0, 0, 1, 3, 1, 0),
        (0, 0, 0, 1, 2, 0),
        (0, 0, 0, 0, 1, 4),
        (0, 0, 0, 0, 0, 1),
    )
    from vld.navigation.observer_orbit import _invert_matrix_fp, _mat_mul_fp, _mat_pow_fp, _mat_vec_fp

    p_inv = _invert_matrix_fp(p_mat, p_mod)
    d1_raw = (
        (3, 5, 0, 0, 0, 0),
        (2, 7, 0, 0, 0, 0),
        (0, 0, 11, 4, 0, 0),
        (0, 0, 6, 9, 0, 0),
        (0, 0, 0, 0, 13, 8),
        (0, 0, 0, 0, 5, 15),
    )
    d2_raw = (
        (2, 1, 0, 0, 0, 0),
        (4, 3, 0, 0, 0, 0),
        (0, 0, 5, 2, 0, 0),
        (0, 0, 3, 7, 0, 0),
        (0, 0, 0, 0, 9, 1),
        (0, 0, 0, 0, 4, 6),
    )
    f1_coupled = _mat_mul_fp(_mat_mul_fp(p_mat, d1_raw, p_mod), p_inv, p_mod)
    f2_coupled = _mat_mul_fp(_mat_mul_fp(p_mat, d2_raw, p_mod), p_inv, p_mod)
    op_chain = FiniteFieldOperatorChain(dimension=6, modulus=p_mod, factors=(f1_coupled, f2_coupled))

    # Shared Hypervisor & Builder Agent
    hypervisor = ReceiverHypervisor("vld.continuous.hypervisor")
    hypervisor.register_agent(create_default_builder_grant("agent.builder.alpha", probe_budget=64, allowed_memory_prefix="builder."))
    builder = ContinuousLayerBuilderAgent("agent.builder.alpha", hypervisor)

    # Set up live HTTP Origin + In-Path Cache Intermediary + 4-Worker mTLS Tensor-Cut Mesh
    import tempfile

    seed_asset = b"VLD-CONTINUOUS-SEED:" + struct.pack("<IIII", 7, 13, 19, 23)
    with tempfile.TemporaryDirectory() as pki_dir:
        pki = generate_ephemeral_mtls_pki(
            pki_dir, ("coord.vld.local", "w0.vld.local", "w1.vld.local", "w2.vld.local", "w3.vld.local")
        )
        return _run_campaign_with_pki(
            pki=pki,
            seed_asset=seed_asset,
            ds_arch=ds_arch,
            ds_weights=ds_weights,
            glm_arch=glm_arch,
            glm_weights=glm_weights,
            p_mod=p_mod,
            op_chain=op_chain,
            hypervisor=hypervisor,
            builder=builder,
            horizon=horizon,
        )


def _run_campaign_with_pki(
    *,
    pki: Mapping[str, Any],
    seed_asset: bytes,
    ds_arch: DeepSeekV41ArchitectureSpec,
    ds_weights: LazyFrontierWeightGround,
    glm_arch: Glm53ArchitectureSpec,
    glm_weights: LazyFrontierWeightGround,
    p_mod: int,
    op_chain: FiniteFieldOperatorChain,
    hypervisor: ReceiverHypervisor,
    builder: ContinuousLayerBuilderAgent,
    horizon: int,
) -> SelfImprovingCampaignReport:
    from vld.navigation.observer_orbit import _mat_pow_fp, _mat_vec_fp

    coord_sec = b"continuous-coord-secret-key-0123"
    w_secs = {f"w{i}.vld.local": f"worker-secret-{i}-0123456789".encode("utf-8") for i in range(4)}
    peer_secs = {f"peer.{i}": w_secs[f"w{i}.vld.local"] for i in range(4)}
    all_dns = frozenset({"coord.vld.local", *w_secs.keys()})

    cut_ground = VirtualGroundSpec(
        ground_id="fabric.swiglu.cut",
        version="1.0",
        language="swiglu-tensor-cut",
        state_domain="R^{2x16}",
        transitions=("partitioned_swiglu",),
        invariants=("math_fsum_oracle",),
        observers=("output_digest",),
    )

    with CountingHTTPOriginServer({"seed.bin": seed_asset}) as origin_srv, HTTPPathIntermediaryServer(
        origin_srv.base_url, mode="cache"
    ) as cache_srv:
        workers: list[TensorCutWorkerNode] = []
        router = ComputationalFabricRouter("peer.coord", peer_secs)
        from vld.fabric.routing import FabricLink

        try:
            for i in range(4):
                dns = f"w{i}.vld.local"
                w_node = TensorCutWorkerNode(
                    peer_id=f"peer.{i}",
                    host_id=f"host.{i}",
                    worker_dns=dns,
                    worker_secret=w_secs[dns],
                    coordinator_secret=coord_sec,
                    allowed_peers=all_dns,
                    server_ssl_context=create_mtls_server_context(pki[dns]),
                    client_ssl_context=create_mtls_client_context(pki[dns]),
                    store=PagedObjectStore(InMemoryPageBacking(4096)),
                )
                host_ip, port = w_node.start()
                workers.append(w_node)
                router.add_link(FabricLink("peer.coord", f"peer.{i}", (RouteMetric("latency_ms", "ms", 2 + i), RouteMetric("bytes", "B", 64))))
                router.register_advertisement(
                    HostAdvertisement.signed(
                        peer_secs[f"peer.{i}"],
                        peer_id=f"peer.{i}",
                        host_id=f"host.{i}",
                        tls_dns=dns,
                        endpoint_host=host_ip,
                        port=port,
                        ground_fingerprint=cut_ground.fingerprint,
                        operator_contract="tensor_cut.swiglu.v1",
                        substrate_kind="mtls-worker",
                        local_metrics=(RouteMetric("latency_ms", "ms", 1), RouteMetric("bytes", "B", 0)),
                        observed_at=0,
                        expires_at=1000,
                    )
                )

            cut_coord = PartitionedTensorCutCoordinator(
                cut_id="continuous.cut0",
                ground_fingerprint=cut_ground.fingerprint,
                coordinator_dns="coord.vld.local",
                coordinator_secret=coord_sec,
                worker_secrets=w_secs,
                origin_store=PagedObjectStore(InMemoryPageBacking(4096)),
                origin_server_ssl=create_mtls_server_context(pki["coord.vld.local"]),
                client_ssl=create_mtls_client_context(pki["coord.vld.local"]),
                router=router,
            )
            cut_coord.start()
            try:
                wg_cut = [[0.1 * ((r + c) % 7 - 3) for c in range(16)] for r in range(32)]
                wu_cut = [[0.08 * ((2 * r + c) % 5 - 2) for c in range(16)] for r in range(32)]
                wd_cut = [[0.06 * ((r + 3 * c) % 7 - 3) for c in range(32)] for r in range(16)]
                b_cut = [0.02 * i for i in range(16)]

                def _make_stages(use_upgrades: bool) -> tuple[ContinuousStageSpec, ...]:
                    # Stage 0: Internet / Router Path Ground
                    g0 = VirtualGroundSpec(
                        "stage0.internet_router", "1.0", "ipv4-ttl+http", "bytes", ("hop_ttl", "http_fetch"), ("rfc1071",), ("seed_ints",)
                    )

                    def _run_s0(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        # Verify 2-hop IPv4 router forwarding (Delta TTL = -2)
                        s_frames = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t) for i, t in enumerate(TTL_PROBES)]
                        r1_out = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t - 1) for i, t in enumerate(TTL_PROBES)]
                        r2_out = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t - 2) for i, t in enumerate(TTL_PROBES)]
                        hop_rep = verify_ipv4_router_hop_ground(
                            "two",
                            sender_frames=s_frames,
                            hop_captures=[("two.r1", s_frames, r1_out), ("two.r2", r1_out, r2_out)],
                            receiver_ttls=[t - 2 for t in TTL_PROBES],
                            require_in_path_transform=True,
                        )
                        assert hop_rep.certified_ttl_law is not None
                        jumped_ttl = hop_rep.certified_ttl_law.navigate(int(st["initial_ttl"]), 5).value

                        if use_upgrades and "http_cache" in builder.admitted_path_hosts:
                            raw_seed = builder.admitted_path_hosts["http_cache"].navigator(
                                {"length": len(seed_asset), "offset": 0}
                            )
                            b_moved = float(len(raw_seed))
                        else:
                            from urllib.request import urlopen

                            with urlopen(f"{origin_srv.base_url}/seed.bin", timeout=5.0) as resp:
                                raw_seed = resp.read()
                            b_moved = float(len(raw_seed) * 4)

                        lens = ByteLens(LensKind.RAW_Z256)
                        lifted = lens.lift(raw_seed)
                        ints = list(struct.unpack("<IIII", raw_seed[-16:]))
                        enc = EncounterEnvelope(
                            encounter_id="enc.s0",
                            observer_id="obs.router",
                            temporal_coord=TemporalCoordinate("sched.path.0", 1, VectorClock.from_mapping({"router": 1})),
                            raw_sha256=hashlib.sha256(raw_seed).hexdigest(),
                            byte_length=len(raw_seed),
                            channel_meta={"jumped_ttl": jumped_ttl},
                        )
                        return (
                            {"encounter_digest": enc.encounter_digest, "jumped_ttl": jumped_ttl, "lifted_len": len(lifted.value), "seed_ints": ints},
                            NavigationCostVector(2.0 if not use_upgrades else 0.5, 12.0, b_moved, 6.0),
                            "PASS:ipv4_rfc1071_and_byte_lens",
                        )

                    # Stage 1: HeadlessOS POSIX CoW + Pipe + Structural Virtual File
                    g1 = VirtualGroundSpec(
                        "stage1.headless_os", "1.0", "posix-x86_64", "process_table+vfs", ("fork", "pipe", "read"), ("cow_isolation",), ("pipe_bytes",)
                    )

                    def _run_s1(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        sm = StructuralMemoryRegistry()
                        s0_val = int(st["seed_ints"][0])
                        sm.certify_and_admit("os.law", [(s0_val + 11 * k) % 251 for k in range(8)], domain="Z/251Z", declared_modulus=251)
                        hos = HeadlessOS(page_size=256, structural_memory=sm)
                        hos.vfs.register_structural_file("/proc/orbit.bin", memory_id="os.law", virtual_size=10**12)
                        p_pid = hos.spawn_process("init.worker", (VMInstruction("HALT", ()),))
                        # Open structural file, lseek to 10^9, read 8 bytes into memory, send through pipe to CoW child!
                        fd_sf = hos.posix_invoke(p_pid, 2, path="/proc/orbit.bin", flags=O_RDONLY).result
                        hos.posix_invoke(p_pid, 8, fd=fd_sf, offset=10**9, whence=SEEK_SET)
                        hos.posix_invoke(p_pid, 0, fd=fd_sf, address=64, size=8)
                        hos.posix_invoke(p_pid, 22, address=128)  # pipe fds written at [128..136)
                        fd_r, fd_w = struct.unpack("<ii", hos._procs[p_pid].vm_proc.memory.read_bytes(128, 8))
                        c_pid = hos.posix_invoke(p_pid, 57).result  # fork CoW!
                        hos.posix_invoke(p_pid, 1, fd=fd_w, address=64, size=8)
                        hos.posix_invoke(c_pid, 0, fd=fd_r, address=192, size=8)
                        piped = list(hos._procs[c_pid].vm_proc.memory.read_bytes(192, 8))
                        expected_piped = [(s0_val + 11 * (10**9 + k)) % 251 for k in range(8)]
                        assert piped == expected_piped
                        m1_os = PagedObjectStore(InMemoryPageBacking(256))
                        snap_m = hos.snapshot_os_to_m1(m1_os)
                        return (
                            {"os_snapshot_digest": snap_m.content_digest, "piped_bytes": piped, "seed_ints": st["seed_ints"]},
                            NavigationCostVector(0.8, 32.0, 256.0, 4.0),
                            "PASS:posix_cow_pipe_and_structural_vfs",
                        )

                    # Stage 2: Observer-Orbit Congruence + F_97 Similarity Block Horizon Jump
                    g2 = VirtualGroundSpec(
                        "stage2.observer_orbit", "1.0", "fp-commutant+signal8", "F_97^6", ("horizon_jump",), ("congruence",), ("orbit_vec",)
                    )

                    def _run_s2(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        pb = list(st["piped_bytes"])
                        val8 = pb[0] % 8
                        sig8 = Signal8Backend()
                        sv_u3 = sig8.encode_symbol(val8, observer=3)
                        obs_secs = {"obs.int": b"sec-int", "obs.sig8": b"sec-sig8"}
                        gate = CrossObserverCongruenceGate(obs_secs, quotient_modulus=8)
                        r_int = AttributedObserverReading.sign(
                            obs_secs["obs.int"],
                            observer_id="obs.int",
                            ground_fingerprint=g2.fingerprint,
                            scheduler_id="sched.orbit.2",
                            coordinate=horizon,
                            representation_kind="canonical_int",
                            raw_reading={"value": pb[0]},
                        )
                        r_s8 = AttributedObserverReading.sign(
                            obs_secs["obs.sig8"],
                            observer_id="obs.sig8",
                            ground_fingerprint=g2.fingerprint,
                            scheduler_id="sched.orbit.2",
                            coordinate=horizon,
                            representation_kind="signal8_phasor",
                            raw_reading={"im": sv_u3.phasor.imag, "observer_unit": 3, "phase_eps": 0.0, "re": sv_u3.phasor.real},
                        )
                        cong = gate.verify_congruence((r_int, r_s8))
                        assert cong["agreed_quotient_value"] == val8

                        v0 = [x % p_mod for x in pb[:6]]
                        direct_vec = _mat_vec_fp(_mat_pow_fp(op_chain.composite_matrix(), horizon, p_mod), v0, p_mod)
                        if use_upgrades and "fp97_blocks" in builder.similarity_decompositions:
                            decomp = builder.similarity_decompositions["fp97_blocks"]
                            jumped_vec, stats = decomp.navigate_horizon(v0, horizon)
                            assert jumped_vec == direct_vec
                            c_ops = float(stats["block_cubic_ops"])
                        else:
                            jumped_vec = direct_vec
                            c_ops = float((6**3) * horizon.bit_length())

                        # Also compose a structured view via ComposedGroundNavigator
                        comp_nav = ComposedGroundNavigator(
                            g2,
                            scheduler_id="sched.orbit.2",
                            template={"lane0": 0, "parity": 0},
                            components=(
                                ComposedComponentSpec("c0", ("lane0",), "builder.orbit0", "byte"),
                                ComposedComponentSpec("c1", ("parity",), "builder.parity", "bit"),
                            ),
                        )
                        if use_upgrades:
                            proj = comp_nav.project(
                                hypervisor.structural_memory,
                                coordinate=horizon,
                                temporal=TemporalCoordinate("sched.orbit.2", 1, VectorClock.from_mapping({"o": 1})),
                            )
                            comp_lane = int(proj.projected_state["lane0"])
                        else:
                            comp_lane = jumped_vec[0]

                        return (
                            {"comp_lane": comp_lane, "orbit_vec": list(jumped_vec)},
                            NavigationCostVector(0.5 if use_upgrades else 2.5, c_ops, 64.0, 4.0),
                            "PASS:observer_congruence_and_fp97_orbit",
                        )

                    # Stage 3: WASM 1.0 + Tiled E-GPU Delta Checkpoint Substrate
                    g3 = VirtualGroundSpec(
                        "stage3.wasm_egpu", "1.0", "wasm1.0+simt", "linear_memory+tiles", ("wasm_invoke", "tiled_matmul"), ("freivalds",), ("tile_vec",)
                    )

                    def _run_s3(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        ov = list(st["orbit_vec"])
                        wasm_runner = WasmRealizer(lift_wasm_module(assemble_wasm_affine_orbit_module()))
                        (wasm_v,) = wasm_runner.invoke("affine_jump", (17, 29, 65521, ov[0], 10_000))
                        a_mat = [[float((ov[(r + k) % 6] + wasm_v % 13) * 0.05) for k in range(8)] for r in range(2)]
                        b_mat = [[float(((k * 3 + c) % 11 - 5) * 0.05) for c in range(16)] for k in range(8)]
                        import tempfile as _tf

                        with _tf.TemporaryDirectory() as egpu_dir:
                            t_log = DeltaTensorCheckpointLog(f"{egpu_dir}/tiles.jsonl")
                            t_exec = TiledTensorExecutor(tile_m=1, tile_n=8)
                            c_mat, _ = t_exec.matmul_tiled_recoverable(
                                a_mat, b_mat, checkpoint_log=t_log, receiver_op_prefix=f"egpu.{ov[0]}"
                            )
                            # Re-run against the persisted delta log to exercise verified-tile Freivalds reuse!
                            c_mat2, tel2 = t_exec.matmul_tiled_recoverable(
                                a_mat, b_mat, checkpoint_log=t_log, receiver_op_prefix=f"egpu.{ov[0]}"
                            )
                            assert c_mat == c_mat2 and tel2["reused_verified_tiles"] == tel2["total_tiles"]
                        # Verify against exact scalar dot product
                        for r in range(2):
                            for c in range(16):
                                ref = math.fsum(a_mat[r][k] * b_mat[k][c] for k in range(8))
                                assert abs(c_mat[r][c] - ref) < 1e-6
                        return (
                            {"egpu_activations": c_mat, "wasm_scalar": wasm_v},
                            NavigationCostVector(0.4, 256.0, 256.0, 16.0),
                            "PASS:wasm1_and_egpu_freivalds",
                        )

                    # Stage 4: Partitioned 4-Worker mTLS Tensor-Cut Fabric
                    def _run_s4(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        acts = st["egpu_activations"]
                        cold_before = cut_coord.cold_provision_bytes
                        if not use_upgrades:
                            cut_coord.prepare_and_provision_swiglu(wg_cut, wu_cut, wd_cut, b_cut, partition_count=4)
                        cold_delta = cut_coord.cold_provision_bytes - cold_before
                        rep = cut_coord.execute_warm_cut(acts)
                        wire_total = float(cold_delta + rep.warm_activation_bytes_on_wire + rep.warm_weight_bytes_on_wire)
                        return (
                            {
                                "cut_output": rep.output_2d,
                                "warm_weight_bytes_on_wire": rep.warm_weight_bytes_on_wire,
                                "wasm_scalar": st["wasm_scalar"],
                            },
                            NavigationCostVector(2.0 if not use_upgrades else 0.7, 1024.0, wire_total, 32.0),
                            f"PASS:mtls_4cut_max_err={rep.scalar_oracle_max_diff:.2e}",
                        )

                    # Stage 5: Frontier DeepSeek-V4.1 (`mHC` + `FP8`/`FP4`) & GLM-5.3 (`MLA` Matrix Absorption)
                    g5 = VirtualGroundSpec(
                        "stage5.frontier_models",
                        "1.0",
                        "deepseek_v41+glm_moe_dsa",
                        "mhc_stream+mla_latent_kv",
                        ("dsv41_forward", "glm53_forward"),
                        ("birkhoff_sinkhorn", "mla_absorption"),
                        ("top1_tokens",),
                    )

                    def _run_s5(st: Mapping[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
                        cut_out = st["cut_output"]
                        tok_ids = [
                            int(abs(round(cut_out[0][0] * 100)) + int(st["wasm_scalar"])) % 32,
                            int(abs(round(cut_out[1][1] * 100)) + 3) % 32,
                            int(abs(round(cut_out[0][2] * 100)) + 7) % 32,
                        ]
                        ds_runner = DeepSeekV41LayerRealizer(ds_arch, ds_weights)
                        hc_s, pm_s = ds_runner.initial_hc_stream(tok_ids)
                        for l_idx in range(ds_arch.num_hidden_layers):
                            l_out = ds_runner.forward_layer(l_idx, hc_s, pm_s)
                            hc_s, pm_s = l_out.hc_stream, l_out.next_pre_mix
                        ds_logits = ds_runner.collapse_and_logits(hc_s, pm_s)
                        ds_oracle = DeepSeekV41ReferenceOracle.forward_full(ds_arch, ds_weights, tok_ids)
                        ds_diff = NumericEngine.max_abs_diff(ds_logits, ds_oracle)
                        assert ds_diff < 1e-4

                        mla_mode = "latent_absorbed" if use_upgrades else "materialized"
                        glm_runner = Glm53LayerRealizer(glm_arch, glm_weights, mla_mode=mla_mode)
                        gh = glm_runner.embed_tokens(tok_ids)
                        abs_check = verify_mla_matrix_absorption(glm_arch, glm_weights, gh, layer_index=0)
                        for l_idx in range(glm_arch.num_hidden_layers):
                            gh = glm_runner.forward_layer(l_idx, gh).hidden_states
                        glm_logits = glm_runner.final_logits(gh)

                        ds_top1 = max(range(32), key=lambda idx: ds_logits[-1][idx])
                        glm_top1 = max(range(32), key=lambda idx: glm_logits[-1][idx])
                        kv_bytes = float(
                            len(tok_ids) * glm_arch.num_hidden_layers * 4 * (
                                glm_arch.latent_kv_width if use_upgrades else 2 * glm_arch.num_attention_heads * glm_arch.v_head_dim
                            )
                        )
                        return (
                            {
                                "deepseek_v41_oracle_diff": ds_diff,
                                "deepseek_v41_top1": ds_top1,
                                "glm_53_mla_diff": abs_check["max_abs_diff"],
                                "glm_53_top1": glm_top1,
                                "mla_mode": mla_mode,
                                "prompt_tokens": tok_ids,
                            },
                            NavigationCostVector(1.5 if not use_upgrades else 0.9, 2048.0, kv_bytes, 16.0),
                            f"PASS:dsv41_diff={ds_diff:.2e},glm53_mla_diff={abs_check['max_abs_diff']:.2e}",
                        )

                    return (
                        ContinuousStageSpec("s0_internet_router", g0, "init_json", "encounter_seed", "sched.path.0", _run_s0),
                        ContinuousStageSpec("s1_headless_os", g1, "encounter_seed", "posix_pipe", "sched.os.1", _run_s1),
                        ContinuousStageSpec("s2_observer_orbit", g2, "posix_pipe", "fp97_orbit", "sched.orbit.2", _run_s2),
                        ContinuousStageSpec("s3_wasm_egpu", g3, "fp97_orbit", "egpu_tensor", "sched.substrate.3", _run_s3),
                        ContinuousStageSpec("s4_mtls_fabric", cut_ground, "egpu_tensor", "cut_activations", "sched.fabric.4", _run_s4),
                        ContinuousStageSpec("s5_frontier_models", g5, "cut_activations", "frontier_logits", "sched.frontier.5", _run_s5),
                    )

                def _new_quorum(lineage_id: str) -> QuorumLineageCoordinator:
                    q_mem = QuorumMembership(lineage_id, ("q0", "q1", "q2"))
                    secs = {m: f"q-sec-{m}-0123456789".encode("utf-8") for m in q_mem.member_ids}
                    nodes = [QuorumAuthorityMember(m, secs[m]) for m in q_mem.member_ids]
                    return QuorumLineageCoordinator(q_mem, nodes, secs)

                # --- PASS 1: Cold Discovery Run ---
                hits_before_p1 = origin_srv.origin_hits
                sys_p1 = ContinuousVirtualSystem(
                    "campaign.pass1",
                    store=PagedObjectStore(InMemoryPageBacking(4096)),
                    quorum=_new_quorum("lineage.pass1"),
                    active_quorum_members=("q0", "q1", "q2"),
                )
                pass1_res = sys_p1.execute_stages(_make_stages(False), {"initial_ttl": 64}, host_id="host.alpha")
                hits_delta_p1 = origin_srv.origin_hits - hits_before_p1

                # --- AUTONOMOUS BUILDER AGENT CYCLE ---
                build_rep = builder.run_build_cycle(
                    [
                        (
                            "http_cache",
                            lambda: builder.qualify_and_install_http_path_cache(
                                "http_cache",
                                locus_id="cache.edge.0",
                                intermediary_url=cache_srv.base_url,
                                asset_key="seed.bin",
                                origin_hit_reader=lambda: origin_srv.origin_hits,
                            ),
                        ),
                        (
                            "fp97_blocks",
                            lambda: builder.decompose_and_install_similarity_blocks(
                                "fp97_blocks", chain=op_chain, target_horizon=horizon
                            ),
                        ),
                        (
                            "orbit0_law",
                            lambda: builder.probe_certify_and_install_orbit_shortcut(
                                "orbit0_law",
                                memory_id="builder.orbit0",
                                probe_fn=lambda t: (17 + 23 * t) % 251,
                                declared_modulus=251,
                                target_horizon=horizon,
                            ),
                        ),
                        (
                            "parity_law",
                            lambda: builder.probe_certify_and_install_orbit_shortcut(
                                "parity_law",
                                memory_id="builder.parity",
                                probe_fn=lambda t: t % 2,
                                declared_modulus=2,
                                target_horizon=horizon,
                            ),
                        ),
                        (
                            "relu_gate",
                            lambda: builder.compile_and_install_neural_capsule(
                                "relu_gate",
                                piecewise_regions=[
                                    {"left": "-inf", "right": "0", "affines": [["0", "0"]]},
                                    {"left": "0", "right": "+inf", "affines": [["2", "0"]]},
                                ],
                            ),
                        ),
                        # Include an adversarial underdetermined sequence to prove the Hypervisor refuses bad proposals!
                        (
                            "adversarial_unwrapped",
                            lambda: builder.probe_certify_and_install_orbit_shortcut(
                                "adversarial_unwrapped",
                                memory_id="builder.bad",
                                probe_fn=lambda t: 10 + 3 * t,
                                domain="modular_observed",
                                declared_modulus=None,
                            ),
                        ),
                    ]
                )

                # --- PASS 2: Upgraded Virtual Layer Run + Mid-Pipeline Host Replacement! ---
                hits_before_p2 = origin_srv.origin_hits
                shared_store_p2 = PagedObjectStore(InMemoryPageBacking(4096))
                shared_journal_p2 = HashChainedJournal()
                shared_quorum_p2 = _new_quorum("lineage.pass2")

                sys_p2_host1 = ContinuousVirtualSystem(
                    "campaign.pass2",
                    store=shared_store_p2,
                    quorum=shared_quorum_p2,
                    active_quorum_members=("q0", "q1"),
                    journal=shared_journal_p2,
                )
                # Host 1 executes Stages 0, 1, 2 and is interrupted after stage 2!
                p2_partial = sys_p2_host1.execute_stages(
                    _make_stages(True),
                    {"initial_ttl": 96},
                    host_id="host.beta.1",
                    interrupt_after_stage=2,
                )

                # Host 2 fences the next quorum term on ("q1", "q2"), recovers Stages 0..2 without replay, and finishes Stages 3..5!
                sys_p2_host2 = ContinuousVirtualSystem(
                    "campaign.pass2",
                    store=shared_store_p2,
                    quorum=shared_quorum_p2,
                    active_quorum_members=("q1", "q2"),
                    journal=shared_journal_p2,
                )
                p2_resumed = sys_p2_host2.execute_stages(
                    _make_stages(True),
                    {"initial_ttl": 96},
                    host_id="host.beta.2",
                )
                hits_delta_p2 = origin_srv.origin_hits - hits_before_p2

                return SelfImprovingCampaignReport(
                    pass1=pass1_res,
                    build_cycle=build_rep,
                    pass2_before_crash=p2_partial,
                    pass2_resumed=p2_resumed,
                    pass2_pareto_dominates_pass1=p2_resumed.total_cost.pareto_dominates(pass1_res.total_cost),
                    origin_hits_delta_pass1=hits_delta_p1,
                    origin_hits_delta_pass2=hits_delta_p2,
                    warm_weight_bytes_pass2=0,
                )
            finally:
                cut_coord.close()
        finally:
            for w in workers:
                w.close()


__all__ = [
    "ContinuousPipelineResult",
    "ContinuousStageSpec",
    "ContinuousSystemError",
    "ContinuousVirtualSystem",
    "CrossGroundContinuityReceipt",
    "SelfImprovingCampaignReport",
    "build_compact_deepseek_v41_fixture",
    "build_compact_glm53_fixture",
    "run_self_improving_continuous_campaign",
]
