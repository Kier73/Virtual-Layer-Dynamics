r"""Phase 7 — AND/OR Proof-Obligation Hypergraph & Untrusted Trace Boundary (`vld.agents.proof`).

Mathematical model:
A `ProofGround` is a finite directed AND/OR hypergraph $\mathcal{H} = (\mathcal{E}, \mathcal{V}, \mathcal{T}, \mathcal{R})$:
  - Each obligation $v \in \mathcal{V}$ is content-addressed over $(\mathcal{E}, \text{goal}, \Gamma, \text{metas}, \text{assumptions})$.
  - Each hyper-transition $e = (u, \text{tactic}, \{w_1, \dots, w_k\}) \in \mathcal{T}$ represents one OR-alternative
    requiring ALL subgoals $w_1 \land \dots \land w_k$ (vacuously discharged when $k = 0$ and kernel-checked).
  - Raw JSON claims imported from external proof traces are stored strictly in `observed_untrusted_kernel_claims`
    and NEVER promoted to `kernel_accepted` without a matching `VerifiedKernelReceipt`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest, parse_strict_json


class ProofGroundError(ValueError):
    """Raised when a proof obligation, hyperedge, or kernel receipt violates structural/scope invariants."""


def _freeze(val: Any) -> Any:
    if isinstance(val, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in val.items()})
    if isinstance(val, (list, tuple)):
        return tuple(_freeze(v) for v in val)
    return val


def _thaw(val: Any) -> Any:
    if isinstance(val, Mapping):
        return {k: _thaw(v) for k, v in val.items()}
    if isinstance(val, tuple):
        return [_thaw(v) for v in val]
    return val


@dataclass(frozen=True, slots=True)
class ProofEnvironment:
    assistant: str
    assistant_version: str
    environment_fingerprint: str
    imports_fingerprint: str
    options_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "assistant",
            "assistant_version",
            "environment_fingerprint",
            "imports_fingerprint",
            "options_fingerprint",
        ):
            val = getattr(self, field_name)
            if not isinstance(val, str) or not val.strip():
                raise ProofGroundError(f"ProofEnvironment.{field_name} must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assistant": self.assistant,
            "assistant_version": self.assistant_version,
            "environment_fingerprint": self.environment_fingerprint,
            "imports_fingerprint": self.imports_fingerprint,
            "options_fingerprint": self.options_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """Scope-bound proof obligation; surface text alone never defines identity."""

    environment: ProofEnvironment
    goal: Mapping[str, Any]
    local_context: tuple[Mapping[str, Any], ...] = ()
    metavariables: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    assumptions: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.environment, ProofEnvironment):
            raise ProofGroundError("ProofEnvironment required.")
        if not isinstance(self.goal, Mapping):
            raise ProofGroundError("goal must be a structured mapping.")
        norm = parse_strict_json(
            canonical_json(
                {
                    "assumptions": list(self.assumptions),
                    "goal": dict(self.goal),
                    "local_context": list(self.local_context),
                    "metavariables": dict(self.metavariables or {}),
                }
            )
        )
        object.__setattr__(self, "goal", _freeze(norm["goal"]))
        object.__setattr__(self, "local_context", _freeze(norm["local_context"]))
        object.__setattr__(self, "metavariables", _freeze(norm["metavariables"]))
        object.__setattr__(self, "assumptions", _freeze(norm["assumptions"]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": _thaw(self.assumptions),
            "environment": self.environment.to_dict(),
            "goal": _thaw(self.goal),
            "local_context": _thaw(self.local_context),
            "metavariables": _thaw(self.metavariables),
        }

    @property
    def fingerprint(self) -> str:
        return domain_digest("vld.proof-obligation.v1", self.to_dict())


@dataclass(frozen=True, slots=True)
class ProofHyperTransition:
    """Single OR-alternative tactic step whose `subgoal_ids` are jointly required (AND)."""

    source_id: str
    tactic: str
    subgoal_ids: tuple[str, ...]
    attempt_status: str = "candidate"
    evidence_ref: str = ""
    transition_id: str = ""

    ALLOWED_STATUSES = frozenset({"candidate", "elaborated", "timeout", "rejected"})

    def __post_init__(self) -> None:
        if not self.source_id or not self.tactic:
            raise ProofGroundError("source_id and tactic must be non-empty.")
        subs = tuple(self.subgoal_ids)
        if len(set(subs)) != len(subs):
            raise ProofGroundError("subgoal_ids cannot contain duplicate obligations.")
        if self.attempt_status not in self.ALLOWED_STATUSES:
            raise ProofGroundError(f"Unsupported attempt_status {self.attempt_status!r}.")
        object.__setattr__(self, "subgoal_ids", subs)
        if not self.transition_id:
            tid = domain_digest(
                "vld.proof-hyperedge.v1",
                {
                    "attempt_status": self.attempt_status,
                    "evidence_ref": self.evidence_ref,
                    "source_id": self.source_id,
                    "subgoal_ids": list(subs),
                    "tactic": self.tactic,
                },
            )
            object.__setattr__(self, "transition_id", tid)


@dataclass(frozen=True, slots=True)
class VerifiedKernelReceipt:
    """Independently verified kernel checker receipt bound to obligation and environment."""

    obligation_id: str
    proof_term_fingerprint: str
    environment_fingerprint: str
    checker_id: str
    checker_version: str
    result: str
    receipt_ref: str

    def __post_init__(self) -> None:
        for name in (
            "obligation_id",
            "proof_term_fingerprint",
            "environment_fingerprint",
            "checker_id",
            "checker_version",
            "receipt_ref",
        ):
            if not getattr(self, name):
                raise ProofGroundError(f"VerifiedKernelReceipt.{name} must be non-empty.")
        if self.result not in ("accepted", "rejected"):
            raise ProofGroundError("result must be 'accepted' or 'rejected'.")


class ProofGround:
    """Finite AND/OR proof-obligation hypergraph with strict kernel-receipt gating."""

    def __init__(
        self,
        environment: ProofEnvironment,
        obligations: Sequence[ProofObligation],
        transitions: Sequence[ProofHyperTransition] = (),
        receipts: Sequence[VerifiedKernelReceipt] = (),
        *,
        observed_untrusted_kernel_claims: Sequence[Any] = (),
    ) -> None:
        self.environment = environment
        self.obligations = tuple(obligations)
        self.transitions = tuple(transitions)
        self.receipts = tuple(receipts)
        self.observed_untrusted_kernel_claims = tuple(_freeze(c) for c in observed_untrusted_kernel_claims)

        by_id = {ob.fingerprint: ob for ob in self.obligations}
        if len(by_id) != len(self.obligations):
            raise ProofGroundError("Duplicate canonical proof obligations.")
        if any(ob.environment != environment for ob in self.obligations):
            raise ProofGroundError("All obligations must share the same ProofEnvironment.")
        for edge in self.transitions:
            if edge.source_id not in by_id or any(sg not in by_id for sg in edge.subgoal_ids):
                raise ProofGroundError("Hyper-transition references undeclared obligation.")

        rcpt_map: dict[str, VerifiedKernelReceipt] = {}
        for rcpt in self.receipts:
            if rcpt.obligation_id not in by_id:
                raise ProofGroundError("Kernel receipt references undeclared obligation.")
            if rcpt.environment_fingerprint != environment.environment_fingerprint:
                raise ProofGroundError("Kernel receipt environment_fingerprint mismatch.")
            prev = rcpt_map.get(rcpt.obligation_id)
            if prev is not None and prev != rcpt:
                raise ProofGroundError("Conflicting kernel receipts for one obligation.")
            rcpt_map[rcpt.obligation_id] = rcpt

        self._by_id = MappingProxyType(by_id)
        self._receipts = MappingProxyType(rcpt_map)

    @property
    def obligation_ids(self) -> tuple[str, ...]:
        return tuple(self._by_id.keys())

    def kernel_status(self, obligation_id: str) -> str:
        if obligation_id not in self._by_id:
            raise ProofGroundError(f"Unknown obligation {obligation_id!r}.")
        rcpt = self._receipts.get(obligation_id)
        if rcpt is None:
            return "unverified"
        return "kernel_accepted" if rcpt.result == "accepted" else "kernel_rejected"

    def provable_obligations(self) -> frozenset[str]:
        """Compute least fixed point of obligations backed by accepted kernel receipts + elaborated AND/OR reductions."""
        proved: set[str] = {
            oid for oid in self._by_id if self.kernel_status(oid) == "kernel_accepted"
        }
        changed = True
        while changed:
            changed = False
            for edge in self.transitions:
                if (
                    edge.attempt_status == "elaborated"
                    and edge.subgoal_ids  # empty subgoal leaf still requires kernel_accepted receipt!
                    and edge.source_id not in proved
                    and self.kernel_status(edge.source_id) != "kernel_rejected"
                    and all(sg in proved for sg in edge.subgoal_ids)
                ):
                    proved.add(edge.source_id)
                    changed = True
        return frozenset(proved)

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_attempts": sum(e.attempt_status == "candidate" for e in self.transitions),
            "environment": self.environment.to_dict(),
            "hyper_transition_count": len(self.transitions),
            "kernel_accepted_obligations": sum(self.kernel_status(oid) == "kernel_accepted" for oid in self._by_id),
            "kernel_rejected_obligations": sum(self.kernel_status(oid) == "kernel_rejected" for oid in self._by_id),
            "obligation_count": len(self.obligations),
            "observed_untrusted_kernel_claims": [_thaw(c) for c in self.observed_untrusted_kernel_claims],
            "provable_via_elaborated_reduction": len(self.provable_obligations()),
            "unverified_obligations": sum(self.kernel_status(oid) == "unverified" for oid in self._by_id),
        }


class ProofGroundAdapter:
    """Import normalized proof traces while quarantining untrusted JSON kernel claims."""

    FORMAT = "vld.proof-trace.v1"

    @classmethod
    def from_trace(
        cls,
        payload: Mapping[str, Any],
        *,
        verified_receipts: Sequence[VerifiedKernelReceipt] = (),
    ) -> ProofGround:
        if not isinstance(payload, Mapping) or payload.get("format") not in (cls.FORMAT, "VGI-LeanProofTrace-1"):
            raise ProofGroundError("Unsupported proof trace format.")
        env = ProofEnvironment(**payload["environment"])
        key_to_id: dict[str, str] = {}
        obligations: list[ProofObligation] = []
        for row in payload["obligations"]:
            ob = ProofObligation(
                environment=env,
                goal=row["goal"],
                local_context=tuple(row.get("local_context", ())),
                metavariables=dict(row.get("metavariables", {})),
                assumptions=tuple(row.get("assumptions", ())),
            )
            key = str(row["key"])
            if not key or key in key_to_id:
                raise ProofGroundError("Duplicate or empty obligation key in trace.")
            key_to_id[key] = ob.fingerprint
            obligations.append(ob)

        edges: list[ProofHyperTransition] = []
        for tr in payload.get("transitions", ()):
            src_key = str(tr["source"])
            sub_keys = tuple(str(k) for k in tr.get("subgoals", ()))
            if src_key not in key_to_id or any(k not in key_to_id for k in sub_keys):
                raise ProofGroundError("Transition references undeclared obligation key.")
            edges.append(
                ProofHyperTransition(
                    source_id=key_to_id[src_key],
                    tactic=str(tr["tactic"]),
                    subgoal_ids=tuple(key_to_id[k] for k in sub_keys),
                    attempt_status=str(tr.get("attempt_status", "candidate")),
                    evidence_ref=str(tr.get("evidence_ref", "")),
                )
            )
        raw_claims = payload.get("kernel_claims", payload.get("kernel_accepted", ()))
        return ProofGround(
            environment=env,
            obligations=obligations,
            transitions=edges,
            receipts=verified_receipts,
            observed_untrusted_kernel_claims=tuple(raw_claims),
        )


__all__ = [
    "ProofEnvironment",
    "ProofGround",
    "ProofGroundAdapter",
    "ProofGroundError",
    "ProofHyperTransition",
    "ProofObligation",
    "VerifiedKernelReceipt",
]
