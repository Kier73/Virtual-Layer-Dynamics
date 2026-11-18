r"""Phase 3 — Strict-Majority ($2q > N$) Quorum Fencing & High-Water Lineage (`vld.authority.quorum`).

Governing equations:
1. **Strict Majority Quorum Intersection:**
   Over fixed authority membership $\mathcal{M}$ of size $N = |\mathcal{M}| \ge 1$, the minimum quorum size is
     $$q_{\min} = \left\lfloor \frac{N}{2} \right\rfloor + 1 \implies 2 q_{\min} > N \implies \forall Q_1, Q_2 \subseteq \mathcal{M},\; |Q_1| \ge q_{\min} \land |Q_2| \ge q_{\min} \implies Q_1 \cap Q_2 \neq \emptyset.$$
2. **Per-Member Atomic Term Ownership, Single-Vote & Cross-Term Value Locking (Paxos Phase 1b/2a):**
   Each authority member $a \in \mathcal{M}$ persists `(highest_term[lineage], term_owner[(lineage, term)], votes[(lineage, slot, term)])`
   inside an atomic cross-process `HashChainedJournal` under `exclusive_guard()`:
   - Member $a$ refuses any proposal or claim with `term < highest_term[lineage]` (`StaleTermError`), and refuses
     competing `owner_id` claims in an already-claimed `(lineage, term)`.
   - Member $a$ refuses any conflicting proposal in an already-voted `(lineage, slot, term)` (`QuorumEquivocationError`).
   - Across terms, `fence_term` / `propose_and_seal` inspect the highest-term prior accepted proposal $(t^*, P^*)$
     reported by any member in the active majority $Q$; by quorum intersection $Q_1 \cap Q_2 \neq \emptyset$, any
     conflicting manifest at an already-quorum-voted slot is deterministically refused (`QuorumEquivocationError`).
3. **Rollback Detection:**
   Any valid historical checkpoint at slot $s < s_{\text{high-water}}$ is deterministically rejected by
   `verify_fresh_checkpoint` (`RollbackDetectedError`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_bytes, domain_digest
from vld.journal import GENESIS_PREV_DIGEST, HashChainedJournal

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class QuorumError(ValueError):
    """Base error for quorum membership, proposal validation, or lineage framing."""


class StaleTermError(QuorumError):
    """Raised when a proposal carries a fencing term/epoch below the promised high-water term or owned by another host."""


class QuorumEquivocationError(QuorumError):
    """Raised when a proposer or cloned credential attempts conflicting proposals for a slot."""


class QuorumInsufficientError(QuorumError):
    """Raised when valid authority votes fail to reach strict majority $2q > N$."""


class RollbackDetectedError(QuorumError):
    """Raised when a presented checkpoint/manifest lags behind or diverges from the quorum high-water mark."""


@dataclass(frozen=True, slots=True)
class QuorumMembership:
    r"""Fixed authority membership $\mathcal{M}$ with strict-majority threshold $q_{\min} = \lfloor N/2 \rfloor + 1$."""

    lineage_id: str
    member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or not self.lineage_id.strip():
            raise QuorumError("lineage_id must be a non-empty string.")
        if isinstance(self.member_ids, (str, bytes)):
            raise QuorumError("member_ids must be a sequence of member identifier strings.")
        cleaned = tuple(sorted(self.member_ids))
        if not cleaned or any(not isinstance(m, str) or not m.strip() for m in cleaned):
            raise QuorumError("QuorumMembership requires at least one non-empty member_id.")
        if len(set(cleaned)) != len(cleaned):
            raise QuorumError("QuorumMembership member_ids must be unique.")
        object.__setattr__(self, "member_ids", cleaned)

    @property
    def size(self) -> int:
        return len(self.member_ids)

    @property
    def quorum_size(self) -> int:
        return (len(self.member_ids) // 2) + 1

    @property
    def membership_digest(self) -> str:
        return domain_digest(
            "vld.quorum-membership.v1",
            {"lineage_id": self.lineage_id, "member_ids": list(self.member_ids)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "member_ids": list(self.member_ids),
            "membership_digest": self.membership_digest,
            "quorum_size": self.quorum_size,
        }


@dataclass(frozen=True, slots=True)
class QuorumProposal:
    """Candidate lineage slot anchor binding predecessor state digest to successor M1 state manifest."""

    lineage_id: str
    membership_digest: str
    slot: int
    term: int
    predecessor_manifest_digest: str
    state_manifest_digest: str
    proposer_id: str
    operation_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.lineage_id, str) or not self.lineage_id.strip():
            raise QuorumError("lineage_id must be non-empty.")
        for name in ("membership_digest", "predecessor_manifest_digest", "state_manifest_digest"):
            val = getattr(self, name)
            if not isinstance(val, str) or _SHA256_RE.fullmatch(val) is None:
                raise QuorumError(f"{name} must be a 64-char lowercase SHA-256 hex digest.")
        if isinstance(self.slot, bool) or not isinstance(self.slot, int) or self.slot <= 0:
            raise QuorumError("slot must be a positive integer (>= 1).")
        if isinstance(self.term, bool) or not isinstance(self.term, int) or self.term <= 0:
            raise QuorumError("term must be a positive integer (>= 1).")
        if not isinstance(self.proposer_id, str) or not self.proposer_id.strip():
            raise QuorumError("proposer_id must be non-empty.")
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise QuorumError("operation_id must be non-empty.")

    def body_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "membership_digest": self.membership_digest,
            "operation_id": self.operation_id,
            "predecessor_manifest_digest": self.predecessor_manifest_digest,
            "proposer_id": self.proposer_id,
            "schema": "vld.quorum-proposal.v1",
            "slot": self.slot,
            "state_manifest_digest": self.state_manifest_digest,
            "term": self.term,
        }

    @property
    def proposal_digest(self) -> str:
        return domain_digest("vld.quorum-proposal.v1", self.body_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.body_dict(), "proposal_digest": self.proposal_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuorumProposal:
        req = {
            "lineage_id",
            "membership_digest",
            "operation_id",
            "predecessor_manifest_digest",
            "proposal_digest",
            "proposer_id",
            "schema",
            "slot",
            "state_manifest_digest",
            "term",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.quorum-proposal.v1":
            raise QuorumError("Invalid QuorumProposal schema.")
        prop = cls(
            lineage_id=data["lineage_id"],
            membership_digest=data["membership_digest"],
            slot=data["slot"],
            term=data["term"],
            predecessor_manifest_digest=data["predecessor_manifest_digest"],
            state_manifest_digest=data["state_manifest_digest"],
            proposer_id=data["proposer_id"],
            operation_id=data["operation_id"],
        )
        if prop.proposal_digest != data["proposal_digest"]:
            raise QuorumError("QuorumProposal digest mismatch.")
        return prop


@dataclass(frozen=True, slots=True)
class QuorumVote:
    """Cryptographically authenticated member vote over one `QuorumProposal`."""

    proposal_digest: str
    lineage_id: str
    slot: int
    term: int
    voter_id: str
    signature_hmac: str

    def __post_init__(self) -> None:
        if not isinstance(self.proposal_digest, str) or _SHA256_RE.fullmatch(self.proposal_digest) is None:
            raise QuorumError("proposal_digest must be a 64-char SHA-256 hex digest.")
        if not isinstance(self.signature_hmac, str) or _SHA256_RE.fullmatch(self.signature_hmac) is None:
            raise QuorumError("signature_hmac must be a 64-char HMAC-SHA256 hex digest.")
        if not isinstance(self.voter_id, str) or not self.voter_id.strip():
            raise QuorumError("voter_id must be non-empty.")

    @staticmethod
    def signing_bytes(proposal_digest: str, lineage_id: str, slot: int, term: int, voter_id: str) -> bytes:
        return canonical_bytes(
            {
                "domain": "vld.quorum-vote.v1",
                "lineage_id": lineage_id,
                "proposal_digest": proposal_digest,
                "slot": slot,
                "term": term,
                "voter_id": voter_id,
            }
        )

    def verify_signature(self, secret: bytes) -> bool:
        expected = hmac.new(
            secret,
            self.signing_bytes(self.proposal_digest, self.lineage_id, self.slot, self.term, self.voter_id),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(self.signature_hmac, expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "proposal_digest": self.proposal_digest,
            "signature_hmac": self.signature_hmac,
            "slot": self.slot,
            "term": self.term,
            "voter_id": self.voter_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuorumVote:
        req = {"lineage_id", "proposal_digest", "signature_hmac", "slot", "term", "voter_id"}
        if not isinstance(data, Mapping) or set(data.keys()) != req:
            raise QuorumError("Invalid QuorumVote fields.")
        return cls(
            proposal_digest=data["proposal_digest"],
            lineage_id=data["lineage_id"],
            slot=data["slot"],
            term=data["term"],
            voter_id=data["voter_id"],
            signature_hmac=data["signature_hmac"],
        )


@dataclass(frozen=True, slots=True)
class AcceptedSlotRecord:
    """Highest-term accepted vote metadata retained by an authority member for Paxos Phase 1b/2a."""

    term: int
    proposal_digest: str
    state_manifest_digest: str
    predecessor_manifest_digest: str
    operation_id: str


class QuorumAuthorityMember:
    """Durable quorum voter enforcing atomic term promises, single-owner epochs, and single-vote per `(slot, term)`."""

    def __init__(
        self,
        member_id: str,
        secret: bytes,
        *,
        journal_path: str | Path | None = None,
    ) -> None:
        if not isinstance(member_id, str) or not member_id.strip():
            raise QuorumError("member_id must be non-empty.")
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise QuorumError("Authority member secret must be bytes of length >= 16.")
        self.member_id = member_id
        self._secret = secret
        self.journal = HashChainedJournal(journal_path)
        self._highest_term: dict[str, int] = {}
        self._term_owners: dict[tuple[str, int], str] = {}
        self._votes: dict[tuple[str, int, int], str] = {}
        self._highest_accepted: dict[tuple[str, int], AcceptedSlotRecord] = {}
        with self.journal.exclusive_guard():
            self._replay_journal_unlocked()

    def _replay_journal_unlocked(self) -> None:
        self.journal.verify()
        for rec in self.journal.events:
            p = rec.payload
            if rec.operation == "quorum.promise_term":
                lin = str(p["lineage_id"])
                term = int(p["term"])
                self._highest_term[lin] = max(self._highest_term.get(lin, 0), term)
                owner = p.get("owner_id")
                if isinstance(owner, str) and owner:
                    self._term_owners.setdefault((lin, term), owner)
            elif rec.operation == "quorum.cast_vote":
                lin = str(p["lineage_id"])
                slot = int(p["slot"])
                term = int(p["term"])
                pdigest = str(p["proposal_digest"])
                self._highest_term[lin] = max(self._highest_term.get(lin, 0), term)
                self._votes[(lin, slot, term)] = pdigest
                s_manifest = str(p.get("state_manifest_digest", ""))
                pred_manifest = str(p.get("predecessor_manifest_digest", ""))
                op_id = str(p.get("operation_id", ""))
                if s_manifest:
                    prev_acc = self._highest_accepted.get((lin, slot))
                    if prev_acc is None or term >= prev_acc.term:
                        self._highest_accepted[(lin, slot)] = AcceptedSlotRecord(
                            term=term,
                            proposal_digest=pdigest,
                            state_manifest_digest=s_manifest,
                            predecessor_manifest_digest=pred_manifest,
                            operation_id=op_id,
                        )

    def highest_term(self, lineage_id: str) -> int:
        with self.journal.exclusive_guard():
            self._replay_journal_unlocked()
            return self._highest_term.get(lineage_id, 0)

    def accepted_slots(self, lineage_id: str) -> dict[int, AcceptedSlotRecord]:
        with self.journal.exclusive_guard():
            self._replay_journal_unlocked()
            return {slot: rec for (lin, slot), rec in self._highest_accepted.items() if lin == lineage_id}

    def promise_term_report(
        self,
        lineage_id: str,
        new_term: int,
        *,
        owner_id: str | None = None,
    ) -> tuple[int, dict[int, AcceptedSlotRecord]]:
        """Atomically fence `new_term` (optionally binding `owner_id`) and return Phase 1b accepted slot records."""
        if isinstance(new_term, bool) or not isinstance(new_term, int) or new_term <= 0:
            raise QuorumError("new_term must be a positive integer.")
        if owner_id is not None and (not isinstance(owner_id, str) or not owner_id.strip()):
            raise QuorumError("owner_id must be a non-empty string when supplied.")

        with self.journal.exclusive_guard():
            self._replay_journal_unlocked()
            cur = self._highest_term.get(lineage_id, 0)
            if new_term < cur:
                raise StaleTermError(
                    f"Member {self.member_id!r} refuses stale term {new_term} < promised {cur}."
                )
            existing_owner = self._term_owners.get((lineage_id, new_term))
            if owner_id is not None and existing_owner is not None and existing_owner != owner_id:
                raise StaleTermError(
                    f"Member {self.member_id!r} already promised term {new_term} to owner {existing_owner!r} (refusing {owner_id!r})."
                )

            need_record = (new_term > cur) or (owner_id is not None and existing_owner is None)
            if need_record:
                req_suffix = f":{owner_id}" if owner_id is not None else ""
                payload: dict[str, Any] = {"lineage_id": lineage_id, "term": new_term}
                if owner_id is not None:
                    payload["owner_id"] = owner_id
                self.journal.record(
                    request_id=f"promise:{lineage_id}:{new_term}{req_suffix}",
                    ground_fingerprint=domain_digest("vld.quorum-lineage-id.v1", lineage_id),
                    operation="quorum.promise_term",
                    payload=payload,
                    result_fn=lambda: {"promised_term": new_term},
                )
                self._highest_term[lineage_id] = max(cur, new_term)
                if owner_id is not None:
                    self._term_owners[(lineage_id, new_term)] = owner_id

            accepted = {slot: rec for (lin, slot), rec in self._highest_accepted.items() if lin == lineage_id}
            return new_term, accepted

    def promise_term(self, lineage_id: str, new_term: int, *, owner_id: str | None = None) -> int:
        """Fence off older epochs by advancing `highest_term[lineage_id]` monotonically."""
        promised, _ = self.promise_term_report(lineage_id, new_term, owner_id=owner_id)
        return promised

    def vote(self, proposal: QuorumProposal) -> QuorumVote:
        """Validate term, epoch ownership & equivocation invariants atomically, persist vote, and return signed `QuorumVote`."""
        with self.journal.exclusive_guard():
            self._replay_journal_unlocked()
            cur_term = self._highest_term.get(proposal.lineage_id, 0)
            if proposal.term < cur_term:
                raise StaleTermError(
                    f"Member {self.member_id!r} refuses proposal in stale term {proposal.term} < {cur_term}."
                )

            key = (proposal.lineage_id, proposal.slot, proposal.term)
            prior = self._votes.get(key)
            if prior is not None and prior != proposal.proposal_digest:
                raise QuorumEquivocationError(
                    f"Member {self.member_id!r} detected equivocation at {key}: already voted for {prior}, "
                    f"refusing conflicting {proposal.proposal_digest}."
                )

            existing_owner = self._term_owners.get((proposal.lineage_id, proposal.term))
            if existing_owner is not None and existing_owner != proposal.proposer_id:
                raise StaleTermError(
                    f"Member {self.member_id!r} promised term {proposal.term} to {existing_owner!r}; "
                    f"refusing proposal from {proposal.proposer_id!r}."
                )

            if prior is None:
                self.journal.record(
                    request_id=f"vote:{proposal.lineage_id}:{proposal.slot}:{proposal.term}",
                    ground_fingerprint=proposal.membership_digest,
                    operation="quorum.cast_vote",
                    payload={
                        "lineage_id": proposal.lineage_id,
                        "operation_id": proposal.operation_id,
                        "predecessor_manifest_digest": proposal.predecessor_manifest_digest,
                        "proposal_digest": proposal.proposal_digest,
                        "proposer_id": proposal.proposer_id,
                        "slot": proposal.slot,
                        "state_manifest_digest": proposal.state_manifest_digest,
                        "term": proposal.term,
                    },
                    result_fn=lambda: {"voted": proposal.proposal_digest},
                )
                self._highest_term[proposal.lineage_id] = max(cur_term, proposal.term)
                self._votes[key] = proposal.proposal_digest
                prev_acc = self._highest_accepted.get((proposal.lineage_id, proposal.slot))
                if prev_acc is None or proposal.term >= prev_acc.term:
                    self._highest_accepted[(proposal.lineage_id, proposal.slot)] = AcceptedSlotRecord(
                        term=proposal.term,
                        proposal_digest=proposal.proposal_digest,
                        state_manifest_digest=proposal.state_manifest_digest,
                        predecessor_manifest_digest=proposal.predecessor_manifest_digest,
                        operation_id=proposal.operation_id,
                    )

        sig = hmac.new(
            self._secret,
            QuorumVote.signing_bytes(
                proposal.proposal_digest,
                proposal.lineage_id,
                proposal.slot,
                proposal.term,
                self.member_id,
            ),
            hashlib.sha256,
        ).hexdigest()
        return QuorumVote(
            proposal_digest=proposal.proposal_digest,
            lineage_id=proposal.lineage_id,
            slot=proposal.slot,
            term=proposal.term,
            voter_id=self.member_id,
            signature_hmac=sig,
        )


@dataclass(frozen=True, slots=True)
class QuorumCommitCertificate:
    """Strict-majority ($2q > N$) sealed proof for one lineage slot transition."""

    proposal: QuorumProposal
    membership: QuorumMembership
    votes: tuple[QuorumVote, ...]
    certificate_digest: str = ""

    def __post_init__(self) -> None:
        if self.proposal.lineage_id != self.membership.lineage_id:
            raise QuorumError("Proposal lineage_id disagrees with QuorumMembership.")
        if self.proposal.membership_digest != self.membership.membership_digest:
            raise QuorumError("Proposal membership_digest disagrees with QuorumMembership.")
        sorted_votes = tuple(sorted(self.votes, key=lambda v: v.voter_id))
        voter_ids = [v.voter_id for v in sorted_votes]
        if len(set(voter_ids)) != len(voter_ids):
            raise QuorumError("Duplicate voter_id in QuorumCommitCertificate.")
        allowed = set(self.membership.member_ids)
        for v in sorted_votes:
            if v.voter_id not in allowed:
                raise QuorumError(f"Outsider voter {v.voter_id!r} not in QuorumMembership.")
            if (
                v.proposal_digest != self.proposal.proposal_digest
                or v.lineage_id != self.proposal.lineage_id
                or v.slot != self.proposal.slot
                or v.term != self.proposal.term
            ):
                raise QuorumError("Vote coordinates disagree with QuorumProposal.")
        if len(sorted_votes) < self.membership.quorum_size:
            raise QuorumInsufficientError(
                f"Insufficient votes ({len(sorted_votes)} < quorum_size {self.membership.quorum_size} for N={self.membership.size})."
            )
        object.__setattr__(self, "votes", sorted_votes)
        computed = domain_digest("vld.quorum-certificate.v1", self._body())
        if self.certificate_digest and self.certificate_digest != computed:
            raise QuorumError("QuorumCommitCertificate digest mismatch.")
        object.__setattr__(self, "certificate_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "membership": self.membership.to_dict(),
            "proposal": self.proposal.to_dict(),
            "schema": "vld.quorum-certificate.v1",
            "votes": [v.to_dict() for v in self.votes],
        }

    def verify_signatures(self, member_secrets: Mapping[str, bytes]) -> bool:
        valid = 0
        for v in self.votes:
            sec = member_secrets.get(v.voter_id)
            if sec is not None and v.verify_signature(sec):
                valid += 1
        return valid >= self.membership.quorum_size

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "certificate_digest": self.certificate_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> QuorumCommitCertificate:
        req = {"certificate_digest", "membership", "proposal", "schema", "votes"}
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.quorum-certificate.v1":
            raise QuorumError("Invalid QuorumCommitCertificate payload.")
        m_raw = data["membership"]
        membership = QuorumMembership(lineage_id=m_raw["lineage_id"], member_ids=tuple(m_raw["member_ids"]))
        proposal = QuorumProposal.from_dict(data["proposal"])
        votes = tuple(QuorumVote.from_dict(v) for v in data["votes"])
        return cls(
            proposal=proposal,
            membership=membership,
            votes=votes,
            certificate_digest=data["certificate_digest"],
        )


class QuorumLineageCoordinator:
    """Coordinates $2q > N$ slot commits and maintains the durable high-water lineage chain."""

    def __init__(
        self,
        membership: QuorumMembership,
        members: Sequence[QuorumAuthorityMember],
        member_secrets: Mapping[str, bytes],
        *,
        ledger_path: str | Path | None = None,
    ) -> None:
        self.membership = membership
        self._members = {m.member_id: m for m in members}
        self._secrets = dict(member_secrets)
        self.ledger = HashChainedJournal(ledger_path)
        self._sealed_by_slot: dict[int, QuorumCommitCertificate] = {}
        self._locked_by_slot: dict[int, AcceptedSlotRecord] = {}
        with self.ledger.exclusive_guard():
            self._sync_from_ledger_unlocked()

    def _unique_targets(self, active_member_ids: Sequence[str] | None) -> tuple[str, ...]:
        if active_member_ids is None:
            return tuple(self.membership.member_ids)
        if isinstance(active_member_ids, (str, bytes)):
            raise QuorumError("active_member_ids must be a sequence of member ID strings.")
        allowed = set(self.membership.member_ids)
        seen: list[str] = []
        seen_set: set[str] = set()
        for mid in active_member_ids:
            if mid in allowed and mid not in seen_set:
                seen_set.add(mid)
                seen.append(mid)
        return tuple(seen)

    def _sync_from_ledger_unlocked(self) -> None:
        self.ledger.verify()
        for rec in self.ledger.events:
            if rec.operation == "quorum.seal_slot":
                cert = QuorumCommitCertificate.from_dict(rec.result["certificate"])
                if not cert.verify_signatures(self._secrets):
                    raise QuorumError("Persisted quorum certificate failed signature verification.")
                self._sealed_by_slot[cert.proposal.slot] = cert

    @property
    def high_water_slot(self) -> int:
        with self.ledger.exclusive_guard():
            self._sync_from_ledger_unlocked()
            return max(self._sealed_by_slot.keys(), default=0)

    @property
    def high_water_certificate(self) -> QuorumCommitCertificate | None:
        with self.ledger.exclusive_guard():
            self._sync_from_ledger_unlocked()
            hw = max(self._sealed_by_slot.keys(), default=0)
            return self._sealed_by_slot.get(hw) if hw > 0 else None

    def certificate_at(self, slot: int) -> QuorumCommitCertificate | None:
        with self.ledger.exclusive_guard():
            self._sync_from_ledger_unlocked()
            return self._sealed_by_slot.get(slot)

    def verify_certificate(self, cert: QuorumCommitCertificate) -> bool:
        """Verify membership identity and strict-majority voter HMAC signatures on `cert`."""
        return (
            cert.membership.lineage_id == self.membership.lineage_id
            and cert.membership.membership_digest == self.membership.membership_digest
            and cert.verify_signatures(self._secrets)
        )

    def max_observed_term(self, *, active_member_ids: Sequence[str] | None = None) -> int:
        """Query the highest promised/voted term across a strict majority $2q > N$ of reachable members."""
        targets = self._unique_targets(active_member_ids)
        hw_cert = self.high_water_certificate
        max_t = hw_cert.proposal.term if hw_cert is not None else 0
        responded = 0
        for mid in targets:
            member = self._members.get(mid)
            if member is None:
                continue
            max_t = max(max_t, member.highest_term(self.membership.lineage_id))
            responded += 1
        if responded < self.membership.quorum_size:
            raise QuorumInsufficientError(
                f"Cannot observe high-water term: only {responded}/{self.membership.size} reachable "
                f"(need {self.membership.quorum_size})."
            )
        return max_t

    def locked_slot_record(self, slot: int) -> AcceptedSlotRecord | None:
        return self._locked_by_slot.get(slot)

    def fence_term(
        self,
        new_term: int,
        *,
        active_member_ids: Sequence[str] | None = None,
        owner_id: str | None = None,
    ) -> int:
        """Broadcast term promise across unique reachable members, require $2q > N$, and record Phase 1b locks."""
        targets = self._unique_targets(active_member_ids)
        promised = 0
        stale_err: StaleTermError | None = None
        last_err: Exception | None = None
        merged_locks: dict[int, AcceptedSlotRecord] = {}

        for mid in targets:
            member = self._members.get(mid)
            if member is None:
                continue
            try:
                _, acc_map = member.promise_term_report(
                    self.membership.lineage_id,
                    new_term,
                    owner_id=owner_id,
                )
                promised += 1
                for s_idx, acc in acc_map.items():
                    prev = merged_locks.get(s_idx)
                    if prev is None or acc.term > prev.term:
                        merged_locks[s_idx] = acc
            except StaleTermError as exc:
                stale_err = exc
                last_err = exc
            except QuorumError as exc:
                last_err = exc

        if promised < self.membership.quorum_size:
            if stale_err is not None:
                raise stale_err
            raise QuorumInsufficientError(
                f"Failed to fence term {new_term}: only {promised}/{self.membership.size} unique members promised "
                f"(need {self.membership.quorum_size}). Last error: {last_err}"
            )
        for s_idx, acc in merged_locks.items():
            prev = self._locked_by_slot.get(s_idx)
            if prev is None or acc.term >= prev.term:
                self._locked_by_slot[s_idx] = acc
        return new_term

    def propose_and_seal(
        self,
        *,
        slot: int,
        term: int,
        predecessor_manifest_digest: str,
        state_manifest_digest: str,
        proposer_id: str,
        operation_id: str,
        active_member_ids: Sequence[str] | None = None,
    ) -> QuorumCommitCertificate:
        """Collect $2q > N$ votes for `slot`, enforce Paxos Phase 2a value-locking & predecessor continuity, and seal."""
        proposal = QuorumProposal(
            lineage_id=self.membership.lineage_id,
            membership_digest=self.membership.membership_digest,
            slot=slot,
            term=term,
            predecessor_manifest_digest=predecessor_manifest_digest,
            state_manifest_digest=state_manifest_digest,
            proposer_id=proposer_id,
            operation_id=operation_id,
        )
        targets = self._unique_targets(active_member_ids)
        if len(targets) < self.membership.quorum_size:
            raise QuorumInsufficientError(
                f"Insufficient unique target members ({len(targets)} < quorum_size {self.membership.quorum_size})."
            )

        with self.ledger.exclusive_guard():
            self._sync_from_ledger_unlocked()
            existing = self._sealed_by_slot.get(slot)
            if existing is not None:
                if (
                    existing.proposal.state_manifest_digest == state_manifest_digest
                    and existing.proposal.operation_id == operation_id
                    and existing.proposal.predecessor_manifest_digest == predecessor_manifest_digest
                ):
                    return existing
                raise QuorumEquivocationError(
                    f"Slot {slot} already quorum-sealed with manifest {existing.proposal.state_manifest_digest}."
                )

            hw = max(self._sealed_by_slot.keys(), default=0)
            if slot != hw + 1:
                raise QuorumError(f"Lineage slot must advance contiguously: expected {hw + 1}, got {slot}.")
            expected_pred = (
                GENESIS_PREV_DIGEST
                if hw == 0
                else self._sealed_by_slot[hw].proposal.state_manifest_digest
            )
            if predecessor_manifest_digest != expected_pred:
                raise RollbackDetectedError(
                    f"Predecessor manifest digest {predecessor_manifest_digest} != high-water {expected_pred}."
                )

            # Paxos Phase 1b/2a check across target majority: ensure no conflicting prior-term vote exists at `slot`!
            prior_lock = self._locked_by_slot.get(slot)
            inspected = 0
            for mid in targets:
                member = self._members.get(mid)
                if member is None:
                    continue
                inspected += 1
                acc_slot = member.accepted_slots(self.membership.lineage_id).get(slot)
                if acc_slot is not None and (prior_lock is None or acc_slot.term > prior_lock.term):
                    prior_lock = acc_slot
            if inspected < self.membership.quorum_size:
                raise QuorumInsufficientError(
                    f"Only {inspected}/{self.membership.size} configured members available."
                )
            if prior_lock is not None:
                if prior_lock.term > term:
                    raise StaleTermError(
                        f"Slot {slot} already carries vote from higher term {prior_lock.term} > {term}."
                    )
                if (
                    prior_lock.state_manifest_digest != state_manifest_digest
                    or prior_lock.predecessor_manifest_digest != predecessor_manifest_digest
                ):
                    raise QuorumEquivocationError(
                        f"Cross-term equivocation refused at slot {slot}: quorum member(s) already voted for manifest "
                        f"{prior_lock.state_manifest_digest} in term {prior_lock.term}; refusing conflicting {state_manifest_digest}."
                    )

            votes: list[QuorumVote] = []
            stale_err: StaleTermError | None = None
            equiv_err: QuorumEquivocationError | None = None
            last_err: Exception | None = None

            for mid in targets:
                member = self._members.get(mid)
                if member is None:
                    continue
                try:
                    v = member.vote(proposal)
                except QuorumEquivocationError as exc:
                    equiv_err = exc
                    last_err = exc
                    continue
                except StaleTermError as exc:
                    stale_err = exc
                    last_err = exc
                    continue
                sec = self._secrets.get(mid)
                if sec is None or not v.verify_signature(sec):
                    raise QuorumError(f"Invalid signature returned by member {mid!r}.")
                votes.append(v)

            if len(votes) < self.membership.quorum_size:
                if equiv_err is not None:
                    raise equiv_err
                if stale_err is not None:
                    raise stale_err
                raise QuorumInsufficientError(
                    f"Insufficient votes ({len(votes)} < {self.membership.quorum_size}). Last error: {last_err}"
                )

            cert = QuorumCommitCertificate(proposal=proposal, membership=self.membership, votes=tuple(votes))
            if not cert.verify_signatures(self._secrets):
                raise QuorumError("Sealed QuorumCommitCertificate failed independent signature verification.")

            self.ledger.record(
                request_id=f"seal:{self.membership.lineage_id}:{slot}",
                ground_fingerprint=self.membership.membership_digest,
                operation="quorum.seal_slot",
                payload={"proposal_digest": proposal.proposal_digest, "slot": slot, "term": term},
                result_fn=lambda: {"certificate": cert.to_dict()},
            )
            self._sealed_by_slot[slot] = cert
            return cert

    def verify_fresh_checkpoint(self, *, slot: int, state_manifest_digest: str) -> QuorumCommitCertificate:
        """Verify that `(slot, state_manifest_digest)` matches the latest quorum high-water mark (defeating rollback)."""
        hw_cert = self.high_water_certificate
        if hw_cert is None:
            raise RollbackDetectedError("No quorum lineage has been initialized yet.")
        if slot < hw_cert.proposal.slot:
            raise RollbackDetectedError(
                f"Rollback detected: checkpoint at slot {slot} is stale behind high-water slot {hw_cert.proposal.slot}."
            )
        if slot > hw_cert.proposal.slot:
            raise RollbackDetectedError(
                f"Unanchored future checkpoint at slot {slot} > high-water slot {hw_cert.proposal.slot}."
            )
        if state_manifest_digest != hw_cert.proposal.state_manifest_digest:
            raise RollbackDetectedError("Checkpoint state_manifest_digest disagrees with quorum high-water mark.")
        return hw_cert


__all__ = [
    "AcceptedSlotRecord",
    "QuorumAuthorityMember",
    "QuorumCommitCertificate",
    "QuorumEquivocationError",
    "QuorumError",
    "QuorumInsufficientError",
    "QuorumLineageCoordinator",
    "QuorumMembership",
    "QuorumProposal",
    "QuorumVote",
    "RollbackDetectedError",
    "StaleTermError",
]
