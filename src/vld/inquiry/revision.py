r"""Append-only Hypothesis Revision Ledger and Observation-to-Ground Active Inquiry Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest
from vld.inquiry.probes import (
    GroundUncertaintyMetrics,
    TransitionProbe,
    evaluate_ground_uncertainty,
    plan_probes_beam,
    plan_probes_greedy,
    plan_probes_minimax,
)
from vld.signal.transitions import GroundTransitionSystem, TransitionOutcome


@dataclass(frozen=True, slots=True)
class HypothesisDefinition:
    hypothesis_id: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.hypothesis_id, str) or not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id must be a non-empty string.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class HypothesisSetRevision:
    revision_id: str
    parent_revision_id: str | None
    hypotheses: tuple[HypothesisDefinition, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class HypothesisLedgerEvent:
    seq: int
    event_type: str
    actor: str
    details: Mapping[str, Any]
    event_digest: str


class HypothesisRevisionLedger:
    """Append-only ledger tracking proposed, validated, and superseded Ground hypotheses."""

    def __init__(
        self,
        initial_hypotheses: Sequence[HypothesisDefinition],
        *,
        author: str = "vld.inquiry",
    ) -> None:
        init = tuple(initial_hypotheses)
        if not init or len({h.hypothesis_id for h in init}) != len(init):
            raise ValueError("initial_hypotheses must be non-empty and uniquely identified.")
        rev = HypothesisSetRevision(
            revision_id="rev-0001",
            parent_revision_id=None,
            hypotheses=init,
            digest=self._digest_set(init),
        )
        self._revisions: list[HypothesisSetRevision] = [rev]
        self._proposed: dict[str, dict[str, Any]] = {}
        self._events: list[HypothesisLedgerEvent] = []
        self._append("revision_created", author, {"digest": rev.digest, "revision_id": rev.revision_id})

    @staticmethod
    def _digest_set(items: Sequence[HypothesisDefinition]) -> str:
        payload = [
            {"description": h.description, "hypothesis_id": h.hypothesis_id}
            for h in sorted(items, key=lambda x: x.hypothesis_id)
        ]
        return domain_digest("vld.hypothesis-set.v1", payload)

    def _append(self, event_type: str, actor: str, details: Mapping[str, Any]) -> HypothesisLedgerEvent:
        seq = len(self._events)
        prev = self._events[-1].event_digest if self._events else ("0" * 64)
        body = {
            "actor": actor,
            "details": dict(details),
            "event_type": event_type,
            "prev_digest": prev,
            "seq": seq,
        }
        ev = HypothesisLedgerEvent(
            seq=seq,
            event_type=event_type,
            actor=actor,
            details=dict(details),
            event_digest=domain_digest("vld.hypothesis-event.v1", body),
        )
        self._events.append(ev)
        return ev

    @property
    def current_revision(self) -> HypothesisSetRevision:
        return self._revisions[-1]

    @property
    def revisions(self) -> tuple[HypothesisSetRevision, ...]:
        return tuple(self._revisions)

    @property
    def events(self) -> tuple[HypothesisLedgerEvent, ...]:
        return tuple(self._events)

    def propose_hypothesis(
        self,
        hypothesis_id: str,
        description: str,
        *,
        evidence_ref: str,
        proposer: str,
    ) -> dict[str, Any]:
        active_ids = {h.hypothesis_id for h in self.current_revision.hypotheses}
        if hypothesis_id in active_ids or hypothesis_id in self._proposed:
            raise ValueError(f"Hypothesis {hypothesis_id!r} already exists.")
        defn = HypothesisDefinition(hypothesis_id, description)
        rec = {
            "based_on_revision": self.current_revision.revision_id,
            "definition": defn,
            "evidence_ref": evidence_ref,
            "status": "proposed",
            "validation_receipt": None,
        }
        self._proposed[hypothesis_id] = rec
        self._append(
            "hypothesis_proposed",
            proposer,
            {
                "based_on_revision": self.current_revision.revision_id,
                "description": description,
                "evidence_ref": evidence_ref,
                "hypothesis_id": hypothesis_id,
            },
        )
        return dict(rec)

    def validate_and_promote(
        self,
        hypothesis_id: str,
        *,
        verdict: str,
        validation_receipt_ref: str,
        validator: str,
        demote_ids: Sequence[str] = (),
    ) -> HypothesisSetRevision | None:
        if verdict not in ("supported", "rejected", "inconclusive"):
            raise ValueError("verdict must be 'supported', 'rejected', or 'inconclusive'.")
        if hypothesis_id not in self._proposed:
            raise ValueError(f"Unknown proposed hypothesis {hypothesis_id!r}.")
        cand = self._proposed[hypothesis_id]
        cand["status"] = verdict
        cand["validation_receipt"] = validation_receipt_ref
        self._append(
            "hypothesis_validated",
            validator,
            {
                "demote_ids": list(demote_ids),
                "hypothesis_id": hypothesis_id,
                "validation_receipt_ref": validation_receipt_ref,
                "verdict": verdict,
            },
        )
        if verdict != "supported":
            return None

        demote_set = set(demote_ids)
        kept = [h for h in self.current_revision.hypotheses if h.hypothesis_id not in demote_set]
        kept.append(cand["definition"])
        new_id = f"rev-{len(self._revisions) + 1:04d}"
        new_rev = HypothesisSetRevision(
            revision_id=new_id,
            parent_revision_id=self.current_revision.revision_id,
            hypotheses=tuple(kept),
            digest=self._digest_set(kept),
        )
        self._revisions.append(new_rev)
        self._append(
            "revision_promoted",
            validator,
            {
                "digest": new_rev.digest,
                "parent_revision_id": new_rev.parent_revision_id,
                "promoted_hypothesis": hypothesis_id,
                "removed_hypotheses": sorted(demote_set),
                "revision_id": new_id,
            },
        )
        _ = canonical_json({"rev": new_id})
        return new_rev


class ActiveGroundInquiry:
    """Versioned observation-to-Ground active perception controller."""

    def __init__(
        self,
        initial_ground: GroundTransitionSystem,
        *,
        policy: str = "minimax",
        horizon: int = 2,
    ) -> None:
        if policy not in ("greedy", "beam", "minimax"):
            raise ValueError("policy must be 'greedy', 'beam', or 'minimax'.")
        self.policy = policy
        self.horizon = horizon
        self._versions: list[GroundTransitionSystem] = [initial_ground]
        self.ledger = HypothesisRevisionLedger(
            [HypothesisDefinition("ground-v1", f"Initial Ground {initial_ground.fingerprint[:12]}")]
        )

    @property
    def current_ground(self) -> GroundTransitionSystem:
        return self._versions[-1]

    @property
    def ground_versions(self) -> tuple[GroundTransitionSystem, ...]:
        return tuple(self._versions)

    def uncertainty(self) -> GroundUncertaintyMetrics:
        return evaluate_ground_uncertainty(self.current_ground, horizon=self.horizon)

    def select_next_probe(
        self,
        candidates: Sequence[TransitionProbe],
        *,
        budget: int | Fraction | None = None,
    ) -> TransitionProbe | None:
        if self.policy == "greedy":
            return plan_probes_greedy(self.current_ground, candidates, horizon=self.horizon, budget=budget)
        if self.policy == "beam":
            seq = plan_probes_beam(self.current_ground, candidates, horizon=self.horizon, budget=budget)
            return seq[0] if seq else None
        plan = plan_probes_minimax(self.current_ground, candidates, horizon=self.horizon, budget=budget)
        return plan.first_probe

    def run_inquiry_loop(
        self,
        candidates: Sequence[TransitionProbe],
        environment_oracle: Callable[[str, str], TransitionOutcome],
        *,
        budget: int | Fraction = Fraction(10),
        max_steps: int = 16,
    ) -> dict[str, Any]:
        """Autonomously probe `environment_oracle` within `budget`, recording versioned Grounds & surprises."""
        rem = Fraction(budget)
        executed_probes: list[dict[str, Any]] = []
        used_cells: set[tuple[str, str]] = set()

        for step_idx in range(max_steps):
            avail = [p for p in candidates if (p.state, p.action) not in used_cells and p.cost <= rem]
            if not avail:
                break
            chosen = self.select_next_probe(avail, budget=rem)
            if chosen is None:
                break

            rem -= chosen.cost
            used_cells.add((chosen.state, chosen.action))
            actual_outcome = environment_oracle(chosen.state, chosen.action)
            declared_outcomes = {s.outcome for s in chosen.scenarios}
            surprise = actual_outcome not in declared_outcomes

            updated = self.current_ground.with_resolved_cell(
                chosen.state,
                chosen.action,
                actual_outcome,
                evidence_tag=f"active_probe_step_{step_idx}",
            )
            self._versions.append(updated)
            hyp_id = f"ground-v{len(self._versions)}"
            desc = (
                f"Surprise branch resolved ({chosen.state}, {chosen.action}) -> {actual_outcome.target}"
                if surprise
                else f"Predicted branch resolved ({chosen.state}, {chosen.action}) -> {actual_outcome.target}"
            )
            self.ledger.propose_hypothesis(
                hyp_id,
                desc,
                evidence_ref=f"probe:{chosen.label}:step:{step_idx}",
                proposer="vld.inquiry.controller",
            )
            prev_hyp_id = self.ledger.current_revision.hypotheses[-1].hypothesis_id
            self.ledger.validate_and_promote(
                hyp_id,
                verdict="supported",
                validation_receipt_ref=updated.fingerprint,
                validator="vld.inquiry.verifier",
                demote_ids=(prev_hyp_id,) if surprise else (),
            )
            executed_probes.append(
                {
                    "action": chosen.action,
                    "cost": str(chosen.cost),
                    "outcome_output": list(actual_outcome.output),
                    "outcome_target": actual_outcome.target,
                    "state": chosen.state,
                    "surprise": surprise,
                }
            )

        final_metrics = self.uncertainty()
        return {
            "complete": self.current_ground.complete,
            "executed_probes": executed_probes,
            "final_ground_fingerprint": self.current_ground.fingerprint,
            "remaining_budget": str(rem),
            "unresolved_cells": final_metrics.unresolved_cells,
            "unresolved_pairs": final_metrics.unresolved_pairs,
            "version_count": len(self._versions),
        }
