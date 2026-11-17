r"""Phase 4 — Direct Terminal-to-Terminal M1/Tensor Forwarding Retaining Origin Commit Authority (`vld.fabric.forwarding`, `G11`).

Mathematical & architectural contract:
Given a coordinator $C$ owning `NavigableComputationDAG` $\mathcal{D}$ and an edge $u \to v$ assigned to
distinct worker terminals $H_1$ (producer of $M_u$) and $H_2$ (consumer of $M_u$):
  1. **Zero Control-Plane Bulk Bloat:** $H_1$ materializes $M_u$ into its local `PagedObjectStore` and
     returns only a compact (< 1 KiB) capability-signed `M1ForwardingTicket` to $C$.
  2. **Direct Terminal-to-Terminal Transfer ($H_1 \to H_2$):** $H_2$ pulls $M_u$ directly from $H_1$'s
     mutually authenticated `MTLSPeerServer` (`fetch_m1_object_resumable`), verifying per-chunk and
     whole-object SHA-256 digests end-to-end.
  3. **Origin-Retained Commit Authority (`G11`):** Neither $H_1$ nor $H_2$ can commit results directly to
     $\mathcal{D}$. Every state commit requires $C$'s assignment MAC verification, expected `node_digest`
     match, and `HashChainedJournal` audit commit at $C$.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import Any, Callable, Mapping

from vld.canonical import canonical_bytes, domain_digest
from vld.fabric.mtls import MTLSPeerClient, MTLSPeerServer
from vld.journal import HashChainedJournal
from vld.memory import PagedObjectStore
from vld.navigation.dag import CommittedNodeResult, ComputationNode, NavigableComputationDAG

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ForwardingError(ValueError):
    """Raised when a forwarding ticket, worker assignment, or commit receipt fails validation."""


class UnauthorizedCommitError(PermissionError):
    """Raised when a worker or outsider attempts to commit without valid coordinator assignment authority."""


@dataclass(frozen=True, slots=True)
class M1ForwardingTicket:
    """Compact control-plane reference allowing consumer $H_2$ to pull bulk M1 bytes directly from producer $H_1$."""

    node_id: str
    manifest_digest: str
    content_digest: str
    size_bytes: int
    producer_dns: str
    producer_host: str
    producer_port: int

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.producer_dns.strip() or not self.producer_host.strip():
            raise ForwardingError("Ticket identifiers must be non-empty.")
        if _SHA256_RE.fullmatch(self.manifest_digest) is None or _SHA256_RE.fullmatch(self.content_digest) is None:
            raise ForwardingError("manifest_digest and content_digest must be 64-char SHA-256 hex digests.")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ForwardingError("size_bytes must be a non-negative integer.")
        if isinstance(self.producer_port, bool) or not isinstance(self.producer_port, int) or self.producer_port <= 0:
            raise ForwardingError("producer_port must be a positive TCP port.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_digest": self.content_digest,
            "manifest_digest": self.manifest_digest,
            "node_id": self.node_id,
            "producer_dns": self.producer_dns,
            "producer_host": self.producer_host,
            "producer_port": self.producer_port,
            "schema": "vld.m1-forwarding-ticket.v1",
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> M1ForwardingTicket:
        req = {
            "content_digest",
            "manifest_digest",
            "node_id",
            "producer_dns",
            "producer_host",
            "producer_port",
            "schema",
            "size_bytes",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.m1-forwarding-ticket.v1":
            raise ForwardingError("Invalid M1ForwardingTicket payload.")
        return cls(
            node_id=data["node_id"],
            manifest_digest=data["manifest_digest"],
            content_digest=data["content_digest"],
            size_bytes=data["size_bytes"],
            producer_dns=data["producer_dns"],
            producer_host=data["producer_host"],
            producer_port=data["producer_port"],
        )


@dataclass(frozen=True, slots=True)
class WorkerExecutionAssignment:
    """Coordinator-signed execution grant scoping one DAG node to one authorized worker peer."""

    dag_id: str
    node: ComputationNode
    assigned_worker_dns: str
    upstream_tickets: tuple[M1ForwardingTicket, ...]
    assignment_mac: str

    @staticmethod
    def compute_mac(
        secret: bytes,
        *,
        dag_id: str,
        node_digest: str,
        assigned_worker_dns: str,
        ticket_digests: tuple[str, ...],
    ) -> str:
        msg = canonical_bytes(
            {
                "assigned_worker_dns": assigned_worker_dns,
                "dag_id": dag_id,
                "domain": "vld.worker-assignment.v1",
                "node_digest": node_digest,
                "ticket_digests": list(ticket_digests),
            }
        )
        return hmac.new(secret, msg, hashlib.sha256).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "assigned_worker_dns": self.assigned_worker_dns,
            "assignment_mac": self.assignment_mac,
            "dag_id": self.dag_id,
            "node": self.node.to_dict(),
            "schema": "vld.worker-assignment.v1",
            "upstream_tickets": [t.to_dict() for t in self.upstream_tickets],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkerExecutionAssignment:
        req = {"assigned_worker_dns", "assignment_mac", "dag_id", "node", "schema", "upstream_tickets"}
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.worker-assignment.v1":
            raise ForwardingError("Invalid WorkerExecutionAssignment payload.")
        return cls(
            dag_id=data["dag_id"],
            node=ComputationNode.from_dict(data["node"]),
            assigned_worker_dns=data["assigned_worker_dns"],
            upstream_tickets=tuple(M1ForwardingTicket.from_dict(t) for t in data["upstream_tickets"]),
            assignment_mac=data["assignment_mac"],
        )


@dataclass(frozen=True, slots=True)
class WorkerCompletionReceipt:
    """Compact worker execution receipt returned to Origin Coordinator $C$ for authoritative commit."""

    dag_id: str
    node_id: str
    node_digest: str
    worker_dns: str
    assignment_mac: str
    summary_value: Any
    output_ticket: M1ForwardingTicket | None
    direct_reconnects: int
    worker_signature_mac: str

    @staticmethod
    def compute_worker_mac(
        worker_secret: bytes,
        *,
        dag_id: str,
        node_id: str,
        node_digest: str,
        worker_dns: str,
        assignment_mac: str,
        summary_value: Any,
        output_manifest_digest: str | None,
    ) -> str:
        msg = canonical_bytes(
            {
                "assignment_mac": assignment_mac,
                "dag_id": dag_id,
                "domain": "vld.worker-completion.v1",
                "node_digest": node_digest,
                "node_id": node_id,
                "output_manifest_digest": output_manifest_digest,
                "summary_value": summary_value,
                "worker_dns": worker_dns,
            }
        )
        return hmac.new(worker_secret, msg, hashlib.sha256).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment_mac": self.assignment_mac,
            "dag_id": self.dag_id,
            "direct_reconnects": self.direct_reconnects,
            "node_digest": self.node_digest,
            "node_id": self.node_id,
            "output_ticket": None if self.output_ticket is None else self.output_ticket.to_dict(),
            "schema": "vld.worker-completion.v1",
            "summary_value": self.summary_value,
            "worker_dns": self.worker_dns,
            "worker_signature_mac": self.worker_signature_mac,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkerCompletionReceipt:
        req = {
            "assignment_mac",
            "dag_id",
            "direct_reconnects",
            "node_digest",
            "node_id",
            "output_ticket",
            "schema",
            "summary_value",
            "worker_dns",
            "worker_signature_mac",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req or data["schema"] != "vld.worker-completion.v1":
            raise ForwardingError("Invalid WorkerCompletionReceipt payload.")
        ot = None if data["output_ticket"] is None else M1ForwardingTicket.from_dict(data["output_ticket"])
        return cls(
            dag_id=data["dag_id"],
            node_id=data["node_id"],
            node_digest=data["node_digest"],
            worker_dns=data["worker_dns"],
            assignment_mac=data["assignment_mac"],
            summary_value=data["summary_value"],
            output_ticket=ot,
            direct_reconnects=int(data["direct_reconnects"]),
            worker_signature_mac=data["worker_signature_mac"],
        )


class ForwardingWorkerNode:
    """Terminal worker node hosting mTLS M1 page transfers and executing coordinator-assigned DAG ops."""

    def __init__(
        self,
        *,
        worker_dns: str,
        worker_secret: bytes,
        store: PagedObjectStore,
        server_ssl_context: Any,
        client_ssl_context: Any,
        allowed_peers: frozenset[str],
        executor_fn: Callable[[ComputationNode, Mapping[str, bytes]], tuple[Any, bytes | None]],
        drop_stream_after_bytes: int | None = None,
    ) -> None:
        self.worker_dns = worker_dns
        self._worker_secret = worker_secret
        self.store = store
        self._client_ssl = client_ssl_context
        self._executor_fn = executor_fn
        self.server = MTLSPeerServer(
            peer_id=worker_dns,
            ssl_context=server_ssl_context,
            allowed_peer_dns=allowed_peers,
            store=store,
            control_handler=self._handle_control,
            drop_stream_after_bytes=drop_stream_after_bytes,
        )

    def start(self) -> tuple[str, int]:
        return self.server.start_background()

    def close(self) -> None:
        self.server.close()

    def _handle_control(self, caller_dns: str, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("op") != "execute_assignment":
            raise ForwardingError(f"Unsupported worker control op: {body.get('op')!r}")
        assignment = WorkerExecutionAssignment.from_dict(body["assignment"])
        if assignment.assigned_worker_dns != self.worker_dns:
            raise UnauthorizedCommitError("Assignment target DNS does not match this worker.")

        # Pull upstream M1 objects DIRECTLY from predecessor worker hosts over mTLS!
        upstream_bytes: dict[str, bytes] = {}
        total_reconnects = 0
        for ticket in assignment.upstream_tickets:
            peer_client = MTLSPeerClient(
                host=ticket.producer_host,
                port=ticket.producer_port,
                expected_server_dns=ticket.producer_dns,
                ssl_context=self._client_ssl,
            )
            local_manifest, recon = peer_client.fetch_m1_object_resumable(ticket.manifest_digest, self.store)
            total_reconnects += recon
            raw = self.store.read_bytes(local_manifest)
            if hashlib.sha256(raw).hexdigest() != ticket.content_digest:
                raise ForwardingError("Direct-forwarded M1 payload digest mismatch.")
            upstream_bytes[ticket.node_id] = raw

        summary_val, produced_bulk = self._executor_fn(assignment.node, upstream_bytes)
        out_ticket: M1ForwardingTicket | None = None
        out_mdigest: str | None = None
        if produced_bulk is not None:
            m = self.store.put_bytes(produced_bulk, media_type="application/octet-stream")
            host, port = self.server.endpoint
            out_ticket = M1ForwardingTicket(
                node_id=assignment.node.node_id,
                manifest_digest=m.manifest_digest,
                content_digest=m.content_digest,
                size_bytes=m.size_bytes,
                producer_dns=self.worker_dns,
                producer_host=host,
                producer_port=port,
            )
            out_mdigest = m.manifest_digest

        sig = WorkerCompletionReceipt.compute_worker_mac(
            self._worker_secret,
            dag_id=assignment.dag_id,
            node_id=assignment.node.node_id,
            node_digest=assignment.node.digest,
            worker_dns=self.worker_dns,
            assignment_mac=assignment.assignment_mac,
            summary_value=summary_val,
            output_manifest_digest=out_mdigest,
        )
        receipt = WorkerCompletionReceipt(
            dag_id=assignment.dag_id,
            node_id=assignment.node.node_id,
            node_digest=assignment.node.digest,
            worker_dns=self.worker_dns,
            assignment_mac=assignment.assignment_mac,
            summary_value=summary_val,
            output_ticket=out_ticket,
            direct_reconnects=total_reconnects,
            worker_signature_mac=sig,
        )
        return receipt.to_dict()


class ForwardingCoordinator:
    """Origin coordinator (`G11`) retaining sole DAG commit authority while keeping bulk bytes off control plane."""

    def __init__(
        self,
        dag: NavigableComputationDAG,
        *,
        coordinator_secret: bytes,
        worker_secrets: Mapping[str, bytes],
        journal: HashChainedJournal | None = None,
    ) -> None:
        if len(coordinator_secret) < 16:
            raise ForwardingError("coordinator_secret must be >= 16 bytes.")
        self.dag = dag
        self._coord_secret = coordinator_secret
        self._worker_secrets = dict(worker_secrets)
        self.journal = journal if journal is not None else HashChainedJournal()
        self._tickets_by_node: dict[str, M1ForwardingTicket] = {}
        self.coordinator_control_bytes = 0

    def issue_assignment(self, node_id: str, worker_dns: str) -> WorkerExecutionAssignment:
        if worker_dns not in self._worker_secrets:
            raise UnauthorizedCommitError(f"Unknown worker_dns {worker_dns!r}.")
        ready_ids = {n.node_id for n in self.dag.ready_frontier()}
        if node_id not in ready_ids:
            raise ForwardingError(f"Node {node_id!r} is not in the ready DAG frontier.")
        node = self.dag.nodes[node_id]
        tickets = tuple(self._tickets_by_node[d] for d in node.dependencies if d in self._tickets_by_node)
        mac = WorkerExecutionAssignment.compute_mac(
            self._coord_secret,
            dag_id=self.dag.dag_id,
            node_digest=node.digest,
            assigned_worker_dns=worker_dns,
            ticket_digests=tuple(t.manifest_digest for t in tickets),
        )
        return WorkerExecutionAssignment(
            dag_id=self.dag.dag_id,
            node=node,
            assigned_worker_dns=worker_dns,
            upstream_tickets=tickets,
            assignment_mac=mac,
        )

    def dispatch_and_commit(
        self,
        assignment: WorkerExecutionAssignment,
        worker_client: MTLSPeerClient,
    ) -> tuple[CommittedNodeResult, WorkerCompletionReceipt]:
        req_dict = {"assignment": assignment.to_dict(), "op": "execute_assignment"}
        self.coordinator_control_bytes += len(canonical_bytes(req_dict))
        resp_dict = worker_client.invoke_control(req_dict)
        self.coordinator_control_bytes += len(canonical_bytes(resp_dict))
        receipt = WorkerCompletionReceipt.from_dict(resp_dict)
        committed = self.verify_and_commit_receipt(receipt)
        return committed, receipt

    def verify_and_commit_receipt(self, receipt: WorkerCompletionReceipt) -> CommittedNodeResult:
        """Authoritatively verify assignment MAC + worker MAC before committing node into `NavigableComputationDAG`."""
        if receipt.dag_id != self.dag.dag_id:
            raise UnauthorizedCommitError("Receipt dag_id mismatch.")
        node = self.dag.nodes.get(receipt.node_id)
        if node is None or node.digest != receipt.node_digest:
            raise UnauthorizedCommitError("Receipt node_id or node_digest mismatch.")
        w_sec = self._worker_secrets.get(receipt.worker_dns)
        if w_sec is None:
            raise UnauthorizedCommitError(f"Unregistered worker {receipt.worker_dns!r}.")

        tickets = tuple(self._tickets_by_node[d] for d in node.dependencies if d in self._tickets_by_node)
        expected_assign_mac = WorkerExecutionAssignment.compute_mac(
            self._coord_secret,
            dag_id=self.dag.dag_id,
            node_digest=node.digest,
            assigned_worker_dns=receipt.worker_dns,
            ticket_digests=tuple(t.manifest_digest for t in tickets),
        )
        if not hmac.compare_digest(receipt.assignment_mac, expected_assign_mac):
            raise UnauthorizedCommitError("Forged or mismatched coordinator assignment MAC.")

        out_mdigest = None if receipt.output_ticket is None else receipt.output_ticket.manifest_digest
        expected_worker_mac = WorkerCompletionReceipt.compute_worker_mac(
            w_sec,
            dag_id=receipt.dag_id,
            node_id=receipt.node_id,
            node_digest=receipt.node_digest,
            worker_dns=receipt.worker_dns,
            assignment_mac=receipt.assignment_mac,
            summary_value=receipt.summary_value,
            output_manifest_digest=out_mdigest,
        )
        if not hmac.compare_digest(receipt.worker_signature_mac, expected_worker_mac):
            raise UnauthorizedCommitError("Invalid worker completion MAC.")

        if receipt.output_ticket is not None:
            self._tickets_by_node[node.node_id] = receipt.output_ticket

        res = self.dag.commit_result(
            node.node_id,
            receipt.summary_value,
            m1_manifest_digest=out_mdigest,
        )
        self.journal.record(
            request_id=f"dag-commit:{self.dag.dag_id}:{node.node_id}",
            ground_fingerprint=domain_digest("vld.dag-id.v1", self.dag.dag_id),
            operation="coordinator.commit_node",
            payload={"node_digest": node.digest, "node_id": node.node_id, "worker_dns": receipt.worker_dns},
            result_fn=lambda: {"m1_manifest_digest": out_mdigest, "result_digest": res.result_digest},
        )
        return res


__all__ = [
    "ForwardingCoordinator",
    "ForwardingError",
    "ForwardingWorkerNode",
    "M1ForwardingTicket",
    "UnauthorizedCommitError",
    "WorkerCompletionReceipt",
    "WorkerExecutionAssignment",
]
