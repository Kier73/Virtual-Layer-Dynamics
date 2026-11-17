r"""Phase 4 — Hierarchical Lazy NavigableComputationDAG (`vld.navigation.dag`).

Mathematical contract:
1. **Unbounded Hierarchical Frontier (Removing the 256-Node Flat Ceiling):**
   A `NavigableComputationDAG` consists of content-addressed atomic `ComputationNode`s and lazy
   `SubgraphThunk`s. A thunk $T_k$ materializes its child subgraph $G_k$ *only* when all upstream
   dependencies $\operatorname{deps}(T_k) \subseteq \operatorname{dom}(C)$ have committed, keeping active
   frontiers compact even for 10,000+ logical operator graphs (e.g., 128-expert MoE layers).
2. **Static & Dynamic Kleene $\mathcal{K}_3$ Readiness:**
     $$F_{\text{ready}}(C) = \{ v \in V \setminus (\operatorname{dom}(C) \cup P) \mid \operatorname{deps}(v) \subseteq \operatorname{dom}(C) \land g_v(C) = \text{TRUE} \}$$
   - $g_v(C) = \text{FALSE}$ deterministically prunes $v$ (and downstream nodes that strictly require $v$).
   - $g_v(C) = \text{INDETERMINATE}$ raises `IndeterminateGuardTrap` atomically before mutating $C$.
3. **Content-Addressed Checkpoint / Restore:**
   `export_checkpoint()` and `from_checkpoint()` preserve and re-verify every node digest, committed
   result digest, expanded thunk, and pruned set across process restarts and host migrations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
from typing import Any, Callable, Iterable, Mapping

from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.triad import IndeterminateGuardTrap, TruthDisposition

_NODE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ComputationDAGError(ValueError):
    """Raised when a computation DAG is malformed, cyclic, tampered, or violates readiness invariants."""


@dataclass(frozen=True, slots=True)
class ComputationNode:
    """Immutable content-addressed operation node in a `NavigableComputationDAG`."""

    node_id: str
    revision: int
    operation: str
    resource: str
    context_digest: str
    dependencies: tuple[str, ...]
    payload: Mapping[str, Any]
    guard_key: str | None = None
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or _NODE_ID_RE.fullmatch(self.node_id) is None:
            raise ComputationDAGError(f"Invalid node_id: {self.node_id!r}.")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ComputationDAGError("revision must be a positive integer.")
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ComputationDAGError("operation must be a non-empty string.")
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ComputationDAGError("resource must be a non-empty string.")
        if not isinstance(self.context_digest, str) or _SHA256_RE.fullmatch(self.context_digest) is None:
            raise ComputationDAGError("context_digest must be a 64-char SHA-256 hex digest.")
        if isinstance(self.dependencies, (str, bytes)):
            raise ComputationDAGError("dependencies must be a sequence of node_id strings.")
        deps = tuple(self.dependencies)
        if any(not isinstance(d, str) or _NODE_ID_RE.fullmatch(d) is None for d in deps):
            raise ComputationDAGError("Invalid dependency node_id.")
        if len(set(deps)) != len(deps) or self.node_id in deps:
            raise ComputationDAGError("dependencies must be unique and exclude self.")
        if self.guard_key is not None and (not isinstance(self.guard_key, str) or not self.guard_key.strip()):
            raise ComputationDAGError("guard_key must be non-empty when supplied.")
        clean_payload = parse_strict_json(canonical_json(dict(self.payload)))
        if not isinstance(clean_payload, dict):
            raise ComputationDAGError("payload must be a JSON object.")
        object.__setattr__(self, "dependencies", deps)
        object.__setattr__(self, "payload", clean_payload)
        computed = domain_digest("vld.dag-node.v1", self._body())
        if self.digest and self.digest != computed:
            raise ComputationDAGError(f"ComputationNode {self.node_id!r} digest mismatch.")
        object.__setattr__(self, "digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "context_digest": self.context_digest,
            "dependencies": list(self.dependencies),
            "guard_key": self.guard_key,
            "node_id": self.node_id,
            "operation": self.operation,
            "payload": dict(self.payload),
            "resource": self.resource,
            "revision": self.revision,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "digest": self.digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComputationNode:
        req = {
            "context_digest",
            "dependencies",
            "digest",
            "guard_key",
            "node_id",
            "operation",
            "payload",
            "resource",
            "revision",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != req:
            raise ComputationDAGError("Invalid ComputationNode fields.")
        return cls(
            node_id=data["node_id"],
            revision=data["revision"],
            operation=data["operation"],
            resource=data["resource"],
            context_digest=data["context_digest"],
            dependencies=tuple(data["dependencies"]),
            payload=data["payload"],
            guard_key=data["guard_key"],
            digest=data["digest"],
        )


@dataclass(frozen=True, slots=True)
class SubgraphThunk:
    """Lazy hierarchical subgraph descriptor expanded only when `dependencies` have committed."""

    thunk_id: str
    dependencies: tuple[str, ...]
    spec_digest: str
    expander: Callable[[Mapping[str, CommittedNodeResult]], Iterable[ComputationNode]]

    def __post_init__(self) -> None:
        if not isinstance(self.thunk_id, str) or _NODE_ID_RE.fullmatch(self.thunk_id) is None:
            raise ComputationDAGError(f"Invalid thunk_id: {self.thunk_id!r}.")
        deps = tuple(self.dependencies)
        if any(not isinstance(d, str) or _NODE_ID_RE.fullmatch(d) is None for d in deps) or len(set(deps)) != len(deps):
            raise ComputationDAGError("SubgraphThunk dependencies must be unique valid node IDs.")
        if not isinstance(self.spec_digest, str) or _SHA256_RE.fullmatch(self.spec_digest) is None:
            raise ComputationDAGError("spec_digest must be a 64-char SHA-256 hex digest.")
        object.__setattr__(self, "dependencies", deps)


@dataclass(frozen=True, slots=True)
class CommittedNodeResult:
    """Immutable verified output receipt for a completed `ComputationNode`."""

    node_id: str
    node_digest: str
    value: Any
    m1_manifest_digest: str | None = None
    result_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or _NODE_ID_RE.fullmatch(self.node_id) is None:
            raise ComputationDAGError("Invalid node_id in CommittedNodeResult.")
        if not isinstance(self.node_digest, str) or _SHA256_RE.fullmatch(self.node_digest) is None:
            raise ComputationDAGError("Invalid node_digest in CommittedNodeResult.")
        if self.m1_manifest_digest is not None and (
            not isinstance(self.m1_manifest_digest, str) or _SHA256_RE.fullmatch(self.m1_manifest_digest) is None
        ):
            raise ComputationDAGError("m1_manifest_digest must be a 64-char SHA-256 hex digest when supplied.")
        clean_val = parse_strict_json(canonical_json(self.value))
        object.__setattr__(self, "value", clean_val)
        computed = domain_digest("vld.dag-result.v1", self._body())
        if self.result_digest and self.result_digest != computed:
            raise ComputationDAGError(f"CommittedNodeResult {self.node_id!r} digest mismatch.")
        object.__setattr__(self, "result_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "m1_manifest_digest": self.m1_manifest_digest,
            "node_digest": self.node_digest,
            "node_id": self.node_id,
            "value": self.value,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "result_digest": self.result_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CommittedNodeResult:
        req = {"m1_manifest_digest", "node_digest", "node_id", "result_digest", "value"}
        if not isinstance(data, Mapping) or set(data.keys()) != req:
            raise ComputationDAGError("Invalid CommittedNodeResult fields.")
        return cls(
            node_id=data["node_id"],
            node_digest=data["node_digest"],
            value=data["value"],
            m1_manifest_digest=data["m1_manifest_digest"],
            result_digest=data["result_digest"],
        )


class NavigableComputationDAG:
    r"""Hierarchical, content-addressed, checkpointable dependency DAG with Kleene $\mathcal{K}_3$ dynamic guards."""

    def __init__(
        self,
        dag_id: str = "vld.dag.default",
        *,
        max_nodes: int = 100_000,
    ) -> None:
        if not isinstance(dag_id, str) or not dag_id.strip():
            raise ComputationDAGError("dag_id must be a non-empty string.")
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
            raise ComputationDAGError("max_nodes must be a positive integer.")
        self.dag_id = dag_id
        self.max_nodes = max_nodes
        self._nodes: dict[str, ComputationNode] = {}
        self._thunks: dict[str, SubgraphThunk] = {}
        self._expanded_thunks: set[str] = set()
        self._committed: dict[str, CommittedNodeResult] = {}
        self._pruned: set[str] = set()
        self._approved_guards: set[str] = set()

    @property
    def nodes(self) -> dict[str, ComputationNode]:
        return dict(self._nodes)

    @property
    def committed(self) -> dict[str, CommittedNodeResult]:
        return dict(self._committed)

    @property
    def pruned(self) -> frozenset[str]:
        return frozenset(self._pruned)

    @property
    def is_complete(self) -> bool:
        """True iff all pending thunks have either expanded or been pruned and every node is committed or pruned."""
        for tid, thunk in self._thunks.items():
            if tid not in self._expanded_thunks and not any(d in self._pruned for d in thunk.dependencies):
                return False
        return all((nid in self._committed or nid in self._pruned) for nid in self._nodes)

    def add_node(self, node: ComputationNode) -> None:
        if not isinstance(node, ComputationNode):
            raise ComputationDAGError("Expected ComputationNode.")
        existing = self._nodes.get(node.node_id)
        if existing is not None:
            if existing.digest == node.digest:
                return
            raise ComputationDAGError(f"Conflicting definition for node_id {node.node_id!r}.")
        if len(self._nodes) + 1 > self.max_nodes:
            raise ComputationDAGError(f"DAG node budget ({self.max_nodes}) exceeded.")
        self._nodes[node.node_id] = node
        try:
            self._verify_acyclic()
        except ComputationDAGError:
            del self._nodes[node.node_id]
            raise

    def add_subgraph_thunk(self, thunk: SubgraphThunk) -> None:
        if not isinstance(thunk, SubgraphThunk):
            raise ComputationDAGError("Expected SubgraphThunk.")
        if thunk.thunk_id in self._thunks:
            raise ComputationDAGError(f"Duplicate thunk_id {thunk.thunk_id!r}.")
        self._thunks[thunk.thunk_id] = thunk

    def _verify_acyclic(self) -> list[str]:
        """Kahn topological sort over currently materialized nodes; raises on any cycle or missing non-thunk dep."""
        in_deg: dict[str, int] = {nid: 0 for nid in self._nodes}
        children: dict[str, list[str]] = {nid: [] for nid in self._nodes}
        for nid, node in self._nodes.items():
            for dep in node.dependencies:
                if dep in self._nodes:
                    in_deg[nid] += 1
                    children[dep].append(nid)
                else:
                    raise ComputationDAGError(f"Node {nid!r} depends on unknown upstream node {dep!r}.")
        queue = deque(sorted(nid for nid, deg in in_deg.items() if deg == 0))
        order: list[str] = []
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in sorted(children[u]):
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)
        if len(order) != len(self._nodes):
            raise ComputationDAGError("Cycle detected in NavigableComputationDAG.")
        return order

    def topological_order(self) -> tuple[str, ...]:
        return tuple(self._verify_acyclic())

    def expand_ready_thunks(self) -> tuple[str, ...]:
        """Materialize all `SubgraphThunk`s whose upstream dependencies have committed."""
        newly_expanded: list[str] = []
        changed = True
        while changed:
            changed = False
            for tid in sorted(self._thunks.keys()):
                if tid in self._expanded_thunks:
                    continue
                thunk = self._thunks[tid]
                if any(d in self._pruned for d in thunk.dependencies):
                    continue
                if all(d in self._committed for d in thunk.dependencies):
                    sub_nodes = list(thunk.expander(self._committed))
                    snapshot = dict(self._nodes)
                    try:
                        for sn in sub_nodes:
                            self.add_node(sn)
                    except Exception:
                        self._nodes = snapshot
                        raise
                    self._expanded_thunks.add(tid)
                    newly_expanded.append(tid)
                    changed = True
        return tuple(newly_expanded)

    def _propagate_pruning(self) -> None:
        changed = True
        while changed:
            changed = False
            for nid, node in self._nodes.items():
                if nid in self._committed or nid in self._pruned:
                    continue
                if any(d in self._pruned for d in node.dependencies):
                    self._pruned.add(nid)
                    changed = True

    def ready_frontier(
        self,
        guard_evaluator: Callable[[ComputationNode, Mapping[str, CommittedNodeResult]], TruthDisposition] | None = None,
    ) -> tuple[ComputationNode, ...]:
        r"""Expand ready thunks, evaluate dynamic Kleene $\mathcal{K}_3$ guards, and return ready frontier."""
        self.expand_ready_thunks()
        self._propagate_pruning()

        pending_prunes: set[str] = set()
        pending_approved: set[str] = set()
        ready: list[ComputationNode] = []
        for nid in sorted(self._nodes.keys()):
            if nid in self._committed or nid in self._pruned:
                continue
            node = self._nodes[nid]
            if not all(d in self._committed for d in node.dependencies):
                continue
            if node.guard_key is not None and nid not in self._approved_guards:
                if guard_evaluator is None:
                    raise IndeterminateGuardTrap(
                        f"Node {nid!r} declares guard_key={node.guard_key!r} but no guard_evaluator was supplied."
                    )
                disp = guard_evaluator(node, self._committed)
                if disp is TruthDisposition.INDETERMINATE:
                    # Atomic trap: mutate neither _committed, _pruned, nor _approved_guards!
                    raise IndeterminateGuardTrap(f"Dynamic DAG guard {node.guard_key!r} on {nid!r} is INDETERMINATE.")
                if disp is TruthDisposition.FALSE:
                    pending_prunes.add(nid)
                    continue
                pending_approved.add(nid)
            ready.append(node)

        if pending_approved:
            self._approved_guards.update(pending_approved)
        if pending_prunes:
            self._pruned.update(pending_prunes)
            self._propagate_pruning()
        return tuple(ready)

    def commit_result(
        self,
        node_id: str,
        value: Any,
        *,
        m1_manifest_digest: str | None = None,
        guard_evaluator: Callable[[ComputationNode, Mapping[str, CommittedNodeResult]], TruthDisposition] | None = None,
    ) -> CommittedNodeResult:
        """Commit verified output for a ready `ComputationNode` (idempotent on identical value)."""
        node = self._nodes.get(node_id)
        if node is None:
            raise ComputationDAGError(f"Unknown node_id {node_id!r}.")
        if node_id in self._pruned:
            raise ComputationDAGError(f"Cannot commit pruned node {node_id!r}.")
        if not all(d in self._committed for d in node.dependencies):
            raise ComputationDAGError(f"Cannot commit {node_id!r}: upstream dependencies incomplete.")
        if node.guard_key is not None and node_id not in self._approved_guards and node_id not in self._committed:
            if guard_evaluator is None:
                raise IndeterminateGuardTrap(
                    f"Cannot commit guarded node {node_id!r} before dynamic guard {node.guard_key!r} evaluates TRUE."
                )
            disp = guard_evaluator(node, self._committed)
            if disp is TruthDisposition.INDETERMINATE:
                raise IndeterminateGuardTrap(f"Dynamic DAG guard {node.guard_key!r} on {node_id!r} is INDETERMINATE.")
            if disp is TruthDisposition.FALSE:
                self._pruned.add(node_id)
                self._propagate_pruning()
                raise ComputationDAGError(f"Cannot commit {node_id!r}: dynamic guard evaluated FALSE and pruned node.")
            self._approved_guards.add(node_id)

        candidate = CommittedNodeResult(
            node_id=node_id,
            node_digest=node.digest,
            value=value,
            m1_manifest_digest=m1_manifest_digest,
        )
        prior = self._committed.get(node_id)
        if prior is not None:
            if prior.result_digest == candidate.result_digest:
                return prior
            raise ComputationDAGError(f"Conflicting commit for already-committed node {node_id!r}.")
        self._committed[node_id] = candidate
        return candidate

    @property
    def state_digest(self) -> str:
        return domain_digest("vld.dag-checkpoint.v1", self._checkpoint_body())

    def _checkpoint_body(self) -> dict[str, Any]:
        return {
            "approved_guards": sorted(self._approved_guards),
            "committed": {nid: res.to_dict() for nid, res in sorted(self._committed.items())},
            "dag_id": self.dag_id,
            "expanded_thunks": sorted(self._expanded_thunks),
            "max_nodes": self.max_nodes,
            "nodes": {nid: node.to_dict() for nid, node in sorted(self._nodes.items())},
            "pruned": sorted(self._pruned),
            "schema": "vld.dag-checkpoint.v1",
        }

    def export_checkpoint(self) -> dict[str, Any]:
        body = self._checkpoint_body()
        return {**body, "checkpoint_digest": domain_digest("vld.dag-checkpoint.v1", body)}

    @classmethod
    def from_checkpoint(
        cls,
        data: Mapping[str, Any],
        *,
        thunks: Iterable[SubgraphThunk] = (),
    ) -> NavigableComputationDAG:
        base_req = {
            "checkpoint_digest",
            "committed",
            "dag_id",
            "expanded_thunks",
            "max_nodes",
            "nodes",
            "pruned",
            "schema",
        }
        if not isinstance(data, Mapping) or not base_req.issubset(data.keys()) or (set(data.keys()) - base_req - {"approved_guards"}):
            raise ComputationDAGError("Invalid DAG checkpoint payload.")
        if data["schema"] != "vld.dag-checkpoint.v1":
            raise ComputationDAGError("Invalid DAG checkpoint schema.")
        body = {k: v for k, v in data.items() if k != "checkpoint_digest"}
        if domain_digest("vld.dag-checkpoint.v1", body) != data["checkpoint_digest"]:
            raise ComputationDAGError("DAG checkpoint digest mismatch.")

        dag = cls(dag_id=data["dag_id"], max_nodes=data["max_nodes"])
        if not isinstance(data["nodes"], Mapping) or len(data["nodes"]) > dag.max_nodes:
            raise ComputationDAGError("Checkpoint nodes invalid or exceed max_nodes budget.")
        raw_nodes: dict[str, ComputationNode] = {}
        for nid, nd in data["nodes"].items():
            parsed_node = ComputationNode.from_dict(nd)
            if parsed_node.node_id != nid:
                raise ComputationDAGError(f"Checkpoint node key {nid!r} != node_id {parsed_node.node_id!r}.")
            raw_nodes[nid] = parsed_node
        dag._nodes = raw_nodes
        topo = dag._verify_acyclic()

        for thunk in thunks:
            dag.add_subgraph_thunk(thunk)
        dag._expanded_thunks = set(data["expanded_thunks"])

        pruned_set = set(data["pruned"])
        if not pruned_set.issubset(raw_nodes.keys()):
            raise ComputationDAGError("Checkpoint pruned set references unknown node_id.")
        dag._pruned = set(pruned_set)
        dag._propagate_pruning()
        if dag._pruned != pruned_set:
            raise ComputationDAGError("Checkpoint pruned set is not closed under downstream propagation.")

        if not isinstance(data["committed"], Mapping) or not set(data["committed"].keys()).issubset(raw_nodes.keys()):
            raise ComputationDAGError("Checkpoint committed map references unknown node_id.")

        approved_set = set(data.get("approved_guards", ()))
        if not approved_set.issubset(raw_nodes.keys()):
            raise ComputationDAGError("Checkpoint approved_guards references unknown node_id.")
        dag._approved_guards = approved_set

        for nid in topo:
            if nid in data["committed"]:
                res = CommittedNodeResult.from_dict(data["committed"][nid])
                if res.node_id != nid or res.node_digest != raw_nodes[nid].digest or nid in dag._pruned:
                    raise ComputationDAGError(f"Corrupted committed node {nid!r} in DAG checkpoint.")
                if not all(d in dag._committed for d in raw_nodes[nid].dependencies):
                    raise ComputationDAGError(f"Committed node {nid!r} lacks committed dependency in checkpoint.")
                if raw_nodes[nid].guard_key is not None:
                    dag._approved_guards.add(nid)
                dag._committed[nid] = res
        return dag


__all__ = [
    "CommittedNodeResult",
    "ComputationDAGError",
    "ComputationNode",
    "NavigableComputationDAG",
    "SubgraphThunk",
]
