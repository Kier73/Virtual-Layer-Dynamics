r"""Federated Computational-Fabric Routing, Epoch-Fenced Execution Claims & Warm Partitioned Tensor-Cut Dispatch (`vld.fabric.routing`, `G19`, `TCR-002`).

Mathematical & architectural contracts:
1. **Multi-Objective Lexicographic Fabric Routing (`ComputationalFabricRouter`):**
   Remote `HostAdvertisement`s are lease-bounded, MAC-attributed claims (`observed_at <= now <= expires_at`),
   never local authority. Over directed peer graph $G = (V, E)$, route weights add componentwise in the
   declared metric basis $\mathbf{m}(\pi) = \sum_{e \in \pi} \mathbf{m}(e) + \mathbf{m}(h_{\text{target}})$,
   pruning ceilings $\mathbf{m}(\pi) \le \mathbf{C}_{\max}$ and returning exact `FabricBlockerCode`s on failure.

2. **Exact Partitioned Tensor-Cut Decomposition (`TCR-002` / `G19`):**
   For disjoint hidden-unit partition $\bigsqcup_{k=0}^{P-1} I_k = \{0, \dots, d_{\text{ff}}-1\}$:
     - *SwiGLU cut:* $Y = b_{\text{out}} + \sum_{k=0}^{P-1} \big(\operatorname{SiLU}(X W_{g[I_k,:]}^\top) \odot (X W_{u[I_k,:]}^\top)\big) W_{d[:,I_k]}^\top$
     - *Elementwise MLP cut:* $Y = b_{\text{out}} + \sum_{k=0}^{P-1} \phi(X W_{1[I_k,:]}^\top + b_{1[I_k]}) W_{2[:,I_k]}^\top$
   Cold provisioning installs shard $k$ in worker $H_k$'s M1 store once; **warm calls transmit only
   typed input/output activations** (`warm_weight_bytes_on_wire == 0`), while `PortableExecutionClaim`
   epoch fencing ($e \mapsto e + 1$) safely reassigns failed partitions and refuses stale zombie receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import heapq
import hmac
import math
import re
from typing import Any, Mapping, Sequence

from vld.accel import NumericEngine, decode_f32_le, encode_f32_le
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.fabric.mtls import MTLSPeerClient, MTLSPeerServer
from vld.journal import HashChainedJournal
from vld.memory import PagedByteManifest, PagedObjectStore
from vld.navigation import ComputationNode, NavigableComputationDAG

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class FabricRoutingError(ValueError):
    """Raised on malformed fabric directory entries, incompatible metrics, or invalid tensor-cut verification."""


class StaleClaimEpochError(PermissionError):
    """Raised when a fenced or superseded worker attempts to complete a stale `assignment_epoch`."""


class FabricResolutionStatus(str, Enum):
    FOUND = "found"
    BLOCKED = "blocked"
    UNREACHABLE = "unreachable"


class FabricBlockerCode(str, Enum):
    NO_HOST = "no_host"
    LEASE_EXPIRED = "lease_expired"
    GROUND_MISMATCH = "ground_mismatch"
    CONTRACT_MISMATCH = "contract_mismatch"
    SUBSTRATE_MISMATCH = "substrate_mismatch"
    RESOURCE_LIMIT = "resource_limit"
    HOP_LIMIT = "hop_limit"
    UNREACHABLE_GRAPH = "unreachable_graph"


# ============================================================================
# 1. Lease-Bounded Host Directory & Multi-Metric Lexicographic Fabric Router
# ============================================================================


@dataclass(frozen=True, slots=True)
class RouteMetric:
    """Single named non-negative integer cost dimension with explicit unit."""

    name: str
    unit: str
    amount: int

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.name) or not _ID_RE.fullmatch(self.unit):
            raise FabricRoutingError("RouteMetric name and unit must be valid identifiers.")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount < 0:
            raise FabricRoutingError("RouteMetric amount must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        return {"amount": self.amount, "name": self.name, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class HostAdvertisement:
    """Portable, lease-bounded remote host capability claim (never treated as local authority)."""

    peer_id: str
    host_id: str
    tls_dns: str
    endpoint_host: str
    port: int
    ground_fingerprint: str
    operator_contract: str
    substrate_kind: str
    local_metrics: tuple[RouteMetric, ...]
    observed_at: int
    expires_at: int
    advertisement_mac: str = ""

    def __post_init__(self) -> None:
        for nm, val in (("peer_id", self.peer_id), ("host_id", self.host_id), ("operator_contract", self.operator_contract), ("substrate_kind", self.substrate_kind)):
            if not _ID_RE.fullmatch(val):
                raise FabricRoutingError(f"Invalid identifier for {nm}: {val!r}")
        if not _SHA256_RE.fullmatch(self.ground_fingerprint):
            raise FabricRoutingError("ground_fingerprint must be 64-char hex.")
        if isinstance(self.port, bool) or not (1 <= self.port <= 65535):
            raise FabricRoutingError("port must lie in [1, 65535].")
        if isinstance(self.observed_at, bool) or isinstance(self.expires_at, bool) or self.observed_at < 0 or self.expires_at < self.observed_at:
            raise FabricRoutingError("Invalid monotonic lease bounds [observed_at, expires_at].")
        sorted_m = tuple(sorted(self.local_metrics, key=lambda m: m.name))
        if len({m.name for m in sorted_m}) != len(sorted_m):
            raise FabricRoutingError("Duplicate metric names in HostAdvertisement.")
        object.__setattr__(self, "local_metrics", sorted_m)

    def body_without_mac(self) -> dict[str, Any]:
        return {
            "domain": "vld.host-advertisement.v1",
            "endpoint_host": self.endpoint_host,
            "expires_at": self.expires_at,
            "ground_fingerprint": self.ground_fingerprint,
            "host_id": self.host_id,
            "local_metrics": [m.to_dict() for m in self.local_metrics],
            "observed_at": self.observed_at,
            "operator_contract": self.operator_contract,
            "peer_id": self.peer_id,
            "port": self.port,
            "substrate_kind": self.substrate_kind,
            "tls_dns": self.tls_dns,
        }

    @classmethod
    def signed(cls, secret: bytes, **kwargs: Any) -> HostAdvertisement:
        draft = cls(**kwargs, advertisement_mac="")
        mac = hmac.new(secret, canonical_bytes(draft.body_without_mac()), hashlib.sha256).hexdigest()
        return cls(**kwargs, advertisement_mac=mac)


@dataclass(frozen=True, slots=True)
class FabricLink:
    """Directed network link `source_peer -> target_peer` with additive `RouteMetric` vector."""

    source_peer: str
    target_peer: str
    metrics: tuple[RouteMetric, ...]

    def __post_init__(self) -> None:
        if not _ID_RE.fullmatch(self.source_peer) or not _ID_RE.fullmatch(self.target_peer):
            raise FabricRoutingError("Invalid peer IDs in FabricLink.")
        sorted_m = tuple(sorted(self.metrics, key=lambda m: m.name))
        if len({m.name for m in sorted_m}) != len(sorted_m):
            raise FabricRoutingError("Duplicate metric names in FabricLink.")
        object.__setattr__(self, "metrics", sorted_m)


@dataclass(frozen=True, slots=True)
class FabricRouteWitness:
    """Ordered lexicographically optimal route from `origin_peer` to `target_host`."""

    peer_path: tuple[str, ...]
    target_host: HostAdvertisement
    total_metrics: tuple[RouteMetric, ...]
    witness_digest: str


@dataclass(frozen=True, slots=True)
class FabricResolutionResult:
    status: FabricResolutionStatus
    routes: tuple[FabricRouteWitness, ...]
    blockers: tuple[FabricBlockerCode, ...]


def _extract_metric_tuple(metrics: Sequence[RouteMetric], order: Sequence[tuple[str, str]]) -> tuple[int, ...]:
    by_name = {m.name: m for m in metrics}
    vec: list[int] = []
    for m_name, m_unit in order:
        found = by_name.get(m_name)
        if found is None or found.unit != m_unit:
            raise FabricRoutingError(f"Missing or incompatible metric {m_name!r} ({m_unit!r}).")
        vec.append(found.amount)
    return tuple(vec)


class ComputationalFabricRouter:
    """Authenticated host directory and multi-objective lexicographic Dijkstra route planner."""

    def __init__(self, origin_peer_id: str, peer_secrets: Mapping[str, bytes] | None = None) -> None:
        if not _ID_RE.fullmatch(origin_peer_id):
            raise FabricRoutingError("Invalid origin_peer_id.")
        self.origin_peer_id = origin_peer_id
        self._peer_secrets = dict(peer_secrets or {})
        self._hosts: dict[tuple[str, str], HostAdvertisement] = {}
        self._adj: dict[str, list[FabricLink]] = {}

    def register_advertisement(self, adv: HostAdvertisement) -> None:
        if adv.peer_id in self._peer_secrets:
            expected = hmac.new(
                self._peer_secrets[adv.peer_id], canonical_bytes(adv.body_without_mac()), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(adv.advertisement_mac, expected):
                raise FabricRoutingError(f"Forged HostAdvertisement MAC from peer {adv.peer_id!r}.")
        self._hosts[(adv.peer_id, adv.host_id)] = adv

    def add_link(self, link: FabricLink) -> None:
        self._adj.setdefault(link.source_peer, []).append(link)

    def resolve_routes(
        self,
        *,
        ground_fingerprint: str,
        operator_contract: str,
        now: int,
        metric_order: Sequence[tuple[str, str]],
        substrate_kind: str | None = None,
        ceilings: Mapping[str, int] | None = None,
        excluded_peers: frozenset[str] = frozenset(),
        max_hops: int = 8,
    ) -> FabricResolutionResult:
        if not metric_order:
            raise FabricRoutingError("metric_order must be non-empty.")
        if not self._hosts:
            return FabricResolutionResult(FabricResolutionStatus.BLOCKED, (), (FabricBlockerCode.NO_HOST,))

        eligible_by_peer: dict[str, list[HostAdvertisement]] = {}
        blockers: set[FabricBlockerCode] = set()
        for adv in self._hosts.values():
            if adv.peer_id in excluded_peers:
                continue
            if not (adv.observed_at <= now <= adv.expires_at):
                blockers.add(FabricBlockerCode.LEASE_EXPIRED)
                continue
            if adv.ground_fingerprint != ground_fingerprint:
                blockers.add(FabricBlockerCode.GROUND_MISMATCH)
                continue
            if adv.operator_contract != operator_contract:
                blockers.add(FabricBlockerCode.CONTRACT_MISMATCH)
                continue
            if substrate_kind is not None and adv.substrate_kind != substrate_kind:
                blockers.add(FabricBlockerCode.SUBSTRATE_MISMATCH)
                continue
            eligible_by_peer.setdefault(adv.peer_id, []).append(adv)

        if not eligible_by_peer:
            return FabricResolutionResult(
                FabricResolutionStatus.BLOCKED,
                (),
                tuple(sorted(blockers or {FabricBlockerCode.NO_HOST}, key=lambda b: b.value)),
            )

        # Multi-metric constrained label-setting search from origin_peer_id
        zero_vec = tuple(0 for _ in metric_order)
        ceil_map = dict(ceilings or {})
        ceil_indices = tuple(idx for idx, (m_name, _u) in enumerate(metric_order) if m_name in ceil_map)

        def _dominates(v1: tuple[int, ...], h1: int, v2: tuple[int, ...], h2: int) -> bool:
            return v1 <= v2 and h1 <= h2 and all(v1[i] <= v2[i] for i in ceil_indices)

        pq: list[tuple[tuple[int, ...], int, tuple[str, ...]]] = [(zero_vec, 0, (self.origin_peer_id,))]
        peer_labels: dict[str, list[tuple[tuple[int, ...], int]]] = {self.origin_peer_id: [(zero_vec, 0)]}
        best_witness_by_host: dict[tuple[str, str], tuple[tuple[int, ...], FabricRouteWitness]] = {}
        saw_hop_limit = False
        saw_ceiling_limit = False

        while pq:
            path_vec, hops, peer_path = heapq.heappop(pq)
            cur_peer = peer_path[-1]
            if any(
                (lv, lh) != (path_vec, hops) and _dominates(lv, lh, path_vec, hops)
                for lv, lh in peer_labels.get(cur_peer, ())
            ):
                continue

            for adv in eligible_by_peer.get(cur_peer, ()):
                host_vec = _extract_metric_tuple(adv.local_metrics, metric_order)
                tot_vec = tuple(a + b for a, b in zip(path_vec, host_vec))
                exceeds = any(tot_vec[idx] > ceil_map[metric_order[idx][0]] for idx in ceil_indices)
                if exceeds:
                    saw_ceiling_limit = True
                    continue
                tot_metrics = tuple(
                    RouteMetric(m_name, m_unit, tot_vec[idx])
                    for idx, (m_name, m_unit) in enumerate(metric_order)
                )
                wdigest = domain_digest(
                    "vld.fabric-route-witness.v1",
                    {
                        "host_id": adv.host_id,
                        "metrics": [m.to_dict() for m in tot_metrics],
                        "peer_path": list(peer_path),
                        "target_peer": adv.peer_id,
                    },
                )
                witness = FabricRouteWitness(
                    peer_path=peer_path,
                    target_host=adv,
                    total_metrics=tot_metrics,
                    witness_digest=wdigest,
                )
                h_key = (adv.peer_id, adv.host_id)
                prev_w = best_witness_by_host.get(h_key)
                if prev_w is None or (tot_vec, peer_path) < (prev_w[0], prev_w[1].peer_path):
                    best_witness_by_host[h_key] = (tot_vec, witness)

            if hops >= max_hops:
                if self._adj.get(cur_peer):
                    saw_hop_limit = True
                continue

            for link in self._adj.get(cur_peer, ()):
                nxt = link.target_peer
                if nxt in excluded_peers or nxt in peer_path:
                    continue
                edge_vec = _extract_metric_tuple(link.metrics, metric_order)
                nxt_vec = tuple(a + b for a, b in zip(path_vec, edge_vec))
                if any(nxt_vec[idx] > ceil_map[metric_order[idx][0]] for idx in ceil_indices):
                    saw_ceiling_limit = True
                    continue
                nxt_hops = hops + 1
                existing = peer_labels.get(nxt, [])
                if any(_dominates(lv, lh, nxt_vec, nxt_hops) for lv, lh in existing):
                    continue
                peer_labels[nxt] = [
                    (lv, lh) for lv, lh in existing if not _dominates(nxt_vec, nxt_hops, lv, lh)
                ] + [(nxt_vec, nxt_hops)]
                heapq.heappush(pq, (nxt_vec, nxt_hops, (*peer_path, nxt)))

        if best_witness_by_host:
            ordered = sorted(
                best_witness_by_host.values(),
                key=lambda it: (it[0], it[1].peer_path, it[1].target_host.host_id),
            )
            return FabricResolutionResult(
                FabricResolutionStatus.FOUND,
                tuple(it[1] for it in ordered),
                (),
            )

        if saw_ceiling_limit:
            blockers.add(FabricBlockerCode.RESOURCE_LIMIT)
        if saw_hop_limit:
            blockers.add(FabricBlockerCode.HOP_LIMIT)
        if not blockers:
            blockers.add(FabricBlockerCode.UNREACHABLE_GRAPH)
        return FabricResolutionResult(
            FabricResolutionStatus.UNREACHABLE,
            (),
            tuple(sorted(blockers, key=lambda b: b.value)),
        )


# ============================================================================
# 2. Epoch-Fenced Execution Claims & Warm Partitioned Tensor-Cut Dispatch (`G19`)
# ============================================================================


@dataclass(frozen=True, slots=True)
class PortableExecutionClaim:
    """Epoch-fenced execution ownership ticket binding one partition cut to one routed host."""

    claim_id: str
    cut_id: str
    partition_index: int
    assignment_epoch: int
    assigned_peer_id: str
    assigned_host_id: str
    route_witness_digest: str
    shard_manifest_digest: str
    claim_mac: str

    @staticmethod
    def mint(
        coordinator_secret: bytes,
        *,
        claim_id: str,
        cut_id: str,
        partition_index: int,
        assignment_epoch: int,
        assigned_peer_id: str,
        assigned_host_id: str,
        route_witness_digest: str,
        shard_manifest_digest: str,
    ) -> PortableExecutionClaim:
        body = {
            "assigned_host_id": assigned_host_id,
            "assigned_peer_id": assigned_peer_id,
            "assignment_epoch": assignment_epoch,
            "claim_id": claim_id,
            "cut_id": cut_id,
            "domain": "vld.portable-execution-claim.v1",
            "partition_index": partition_index,
            "route_witness_digest": route_witness_digest,
            "shard_manifest_digest": shard_manifest_digest,
        }
        mac = hmac.new(coordinator_secret, canonical_bytes(body), hashlib.sha256).hexdigest()
        return PortableExecutionClaim(
            claim_id=claim_id,
            cut_id=cut_id,
            partition_index=partition_index,
            assignment_epoch=assignment_epoch,
            assigned_peer_id=assigned_peer_id,
            assigned_host_id=assigned_host_id,
            route_witness_digest=route_witness_digest,
            shard_manifest_digest=shard_manifest_digest,
            claim_mac=mac,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assigned_host_id": self.assigned_host_id,
            "assigned_peer_id": self.assigned_peer_id,
            "assignment_epoch": self.assignment_epoch,
            "claim_id": self.claim_id,
            "claim_mac": self.claim_mac,
            "cut_id": self.cut_id,
            "partition_index": self.partition_index,
            "route_witness_digest": self.route_witness_digest,
            "shard_manifest_digest": self.shard_manifest_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PortableExecutionClaim:
        return cls(
            claim_id=str(data["claim_id"]),
            cut_id=str(data["cut_id"]),
            partition_index=int(data["partition_index"]),
            assignment_epoch=int(data["assignment_epoch"]),
            assigned_peer_id=str(data["assigned_peer_id"]),
            assigned_host_id=str(data["assigned_host_id"]),
            route_witness_digest=str(data["route_witness_digest"]),
            shard_manifest_digest=str(data["shard_manifest_digest"]),
            claim_mac=str(data["claim_mac"]),
        )


def _pack_swiglu_shard(
    w_gate_slice: Sequence[Sequence[float]],  # [part_ff, d_in]
    w_up_slice: Sequence[Sequence[float]],  # [part_ff, d_in]
    w_down_slice: Sequence[Sequence[float]],  # [d_out, part_ff]
) -> bytes:
    part_ff = len(w_gate_slice)
    d_in = len(w_gate_slice[0])
    d_out = len(w_down_slice)
    header = canonical_json({"d_in": d_in, "d_out": d_out, "part_ff": part_ff}).encode("utf-8")
    flat = [
        *(float(v) for row in w_gate_slice for v in row),
        *(float(v) for row in w_up_slice for v in row),
        *(float(v) for row in w_down_slice for v in row),
    ]
    payload = encode_f32_le(flat)
    return len(header).to_bytes(4, "little") + header + payload


def _unpack_swiglu_shard(raw: bytes) -> tuple[int, int, int, list[list[float]], list[list[float]], list[list[float]]]:
    hlen = int.from_bytes(raw[:4], "little")
    meta = parse_strict_json(raw[4 : 4 + hlen])
    d_in, d_out, part_ff = int(meta["d_in"]), int(meta["d_out"]), int(meta["part_ff"])
    vals = decode_f32_le(raw[4 + hlen :])
    n_g = part_ff * d_in
    g_flat = vals[:n_g]
    u_flat = vals[n_g : 2 * n_g]
    d_flat = vals[2 * n_g :]
    wg = [g_flat[r * d_in : (r + 1) * d_in] for r in range(part_ff)]
    wu = [u_flat[r * d_in : (r + 1) * d_in] for r in range(part_ff)]
    wd = [d_flat[r * part_ff : (r + 1) * part_ff] for r in range(d_out)]
    return d_in, d_out, part_ff, wg, wu, wd


class TensorCutWorkerNode:
    """mTLS-authenticated worker caching provisioned SwiGLU parameter shards in M1 for warm zero-weight-wire execution."""

    def __init__(
        self,
        *,
        peer_id: str,
        host_id: str,
        worker_dns: str,
        worker_secret: bytes,
        coordinator_secret: bytes,
        allowed_peers: frozenset[str],
        server_ssl_context: Any,
        client_ssl_context: Any,
        store: PagedObjectStore,
        corrupt_output: bool = False,
    ) -> None:
        self.peer_id = peer_id
        self.host_id = host_id
        self.worker_dns = worker_dns
        self._worker_secret = worker_secret
        self._coordinator_secret = coordinator_secret
        self._client_ssl = client_ssl_context
        self.store = store
        self.corrupt_output = corrupt_output
        self._cached_shards: dict[str, PagedByteManifest] = {}
        self._engine = NumericEngine("auto")
        self.cold_bytes_pulled = 0
        self.warm_invocations = 0
        self.server = MTLSPeerServer(
            peer_id=worker_dns,
            ssl_context=server_ssl_context,
            allowed_peer_dns=allowed_peers,
            store=store,
            control_handler=self._handle_control,
        )

    def start(self) -> tuple[str, int]:
        return self.server.start_background()

    def close(self) -> None:
        self.server.close()

    def _handle_control(self, _caller_dns: str, body: dict[str, Any]) -> dict[str, Any]:
        op = body.get("op")
        if op == "provision_shard":
            src_host = str(body["origin_host"])
            src_port = int(body["origin_port"])
            src_dns = str(body["origin_dns"])
            m_digest = str(body["shard_manifest_digest"])
            if m_digest not in self._cached_shards:
                client = MTLSPeerClient(
                    host=src_host,
                    port=src_port,
                    expected_server_dns=src_dns,
                    ssl_context=self._client_ssl,
                )
                stored, _ = client.fetch_m1_object_resumable(m_digest, self.store)
                self._cached_shards[m_digest] = stored
                self.cold_bytes_pulled += stored.size_bytes
            return {"cached": True, "shard_manifest_digest": m_digest}

        if op == "execute_warm_cut":
            claim = PortableExecutionClaim.from_dict(body["claim"])
            expected = PortableExecutionClaim.mint(
                self._coordinator_secret,
                claim_id=claim.claim_id,
                cut_id=claim.cut_id,
                partition_index=claim.partition_index,
                assignment_epoch=claim.assignment_epoch,
                assigned_peer_id=claim.assigned_peer_id,
                assigned_host_id=claim.assigned_host_id,
                route_witness_digest=claim.route_witness_digest,
                shard_manifest_digest=claim.shard_manifest_digest,
            )
            if not hmac.compare_digest(claim.claim_mac, expected.claim_mac):
                raise PermissionError("Worker rejected forged PortableExecutionClaim MAC.")
            if claim.assigned_peer_id != self.peer_id or claim.assigned_host_id != self.host_id:
                raise PermissionError("Claim target host mismatch.")
            shard_m = self._cached_shards.get(claim.shard_manifest_digest)
            if shard_m is None:
                raise FabricRoutingError("Warm execution requested before shard provisioning.")

            seq_len = int(body["seq_len"])
            d_in_req = int(body["d_in"])
            act_bytes = bytes.fromhex(str(body["activations_f32_hex"]))
            act_dg = hashlib.sha256(act_bytes).hexdigest()
            x_flat = decode_f32_le(act_bytes)
            if len(x_flat) != seq_len * d_in_req:
                raise FabricRoutingError("Activation element count disagrees with seq_len * d_in.")
            x_2d = [x_flat[t * d_in_req : (t + 1) * d_in_req] for t in range(seq_len)]

            d_in, d_out, _pff, wg, wu, wd = _unpack_swiglu_shard(self.store.read_bytes(shard_m))
            if d_in != d_in_req:
                raise FabricRoutingError("Activation width disagrees with cached shard d_in.")

            gate = self._engine.linear_x_wt(x_2d, wg)
            up = self._engine.linear_x_wt(x_2d, wu)
            swi = self._engine.swiglu_2d(gate, up)
            partial_2d = self._engine.linear_x_wt(swi, wd)
            if self.corrupt_output:
                partial_2d[0][0] += 1.25  # adversarial rehashed false arithmetic test!

            out_hex = encode_f32_le([v for row in partial_2d for v in row]).hex()
            out_dg = hashlib.sha256(bytes.fromhex(out_hex)).hexdigest()
            rcpt_body = {
                "activation_sha256": act_dg,
                "assignment_epoch": claim.assignment_epoch,
                "claim_id": claim.claim_id,
                "cut_id": claim.cut_id,
                "d_out": d_out,
                "domain": "vld.tensor-cut-receipt.v1",
                "host_id": self.host_id,
                "partial_sha256": out_dg,
                "partition_index": claim.partition_index,
                "peer_id": self.peer_id,
                "route_witness_digest": claim.route_witness_digest,
                "shard_manifest_digest": claim.shard_manifest_digest,
            }
            w_mac = hmac.new(self._worker_secret, canonical_bytes(rcpt_body), hashlib.sha256).hexdigest()
            self.warm_invocations += 1
            return {
                "d_out": d_out,
                "partial_f32_hex": out_hex,
                "partial_sha256": out_dg,
                "receipt": {**rcpt_body, "worker_mac": w_mac},
            }

        raise FabricRoutingError(f"Unsupported TensorCutWorkerNode op: {op!r}")


@dataclass(frozen=True, slots=True)
class PartitionedCutReport:
    """Complete verified report for a multi-host partitioned SwiGLU tensor-cut execution (`G19`)."""

    cut_id: str
    partition_count: int
    output_2d: list[list[float]]
    cold_provision_bytes: int
    warm_activation_bytes_on_wire: int
    warm_weight_bytes_on_wire: int
    reassignments_performed: int
    scalar_oracle_max_diff: float
    dag_checkpoint_digest: str
    claims: tuple[PortableExecutionClaim, ...]


class PartitionedTensorCutCoordinator:
    """Plans, provisions, executes, reassigns, and scalar-oracle-verifies partitioned SwiGLU cuts over mTLS."""

    def __init__(
        self,
        *,
        cut_id: str,
        ground_fingerprint: str,
        coordinator_dns: str,
        coordinator_secret: bytes,
        worker_secrets: Mapping[str, bytes],
        origin_store: PagedObjectStore,
        origin_server_ssl: Any,
        client_ssl: Any,
        router: ComputationalFabricRouter,
        journal: HashChainedJournal | None = None,
    ) -> None:
        self.cut_id = cut_id
        self.ground_fingerprint = ground_fingerprint
        self._coord_secret = coordinator_secret
        self._worker_secrets = dict(worker_secrets)
        self.origin_store = origin_store
        self._client_ssl = client_ssl
        self.router = router
        self.journal = journal if journal is not None else HashChainedJournal()
        self.origin_server = MTLSPeerServer(
            peer_id=coordinator_dns,
            ssl_context=origin_server_ssl,
            allowed_peer_dns=frozenset({coordinator_dns, *worker_secrets.keys()}),
            store=origin_store,
        )
        self._origin_addr: tuple[str, int] | None = None
        self._coordinator_dns = coordinator_dns
        self._shard_manifests: list[PagedByteManifest] = []
        self._active_epochs: dict[int, int] = {}
        self._active_routes: dict[int, FabricRouteWitness] = {}
        self._w_gate: list[list[float]] = []
        self._w_up: list[list[float]] = []
        self._w_down: list[list[float]] = []
        self._b_out: list[float] = []
        self.cold_provision_bytes = 0

    def start(self) -> tuple[str, int]:
        self._origin_addr = self.origin_server.start_background()
        return self._origin_addr

    def close(self) -> None:
        self.origin_server.close()

    def prepare_and_provision_swiglu(
        self,
        w_gate: Sequence[Sequence[float]],
        w_up: Sequence[Sequence[float]],
        w_down: Sequence[Sequence[float]],
        b_out: Sequence[float],
        *,
        partition_count: int = 4,
        now: int = 10,
        metric_order: Sequence[tuple[str, str]] = (("latency_ms", "ms"), ("bytes", "B")),
    ) -> tuple[PortableExecutionClaim, ...]:
        """Slice `[d_ff, d_in]` / `[d_out, d_ff]` into `partition_count` disjoint shards and provision over mTLS."""
        if self._origin_addr is None:
            self.start()
        assert self._origin_addr is not None
        d_ff = len(w_gate)
        if partition_count < 1 or d_ff % partition_count != 0:
            raise FabricRoutingError("d_ff must be divisible by partition_count >= 1.")
        self._w_gate = [[float(v) for v in r] for r in w_gate]
        self._w_up = [[float(v) for v in r] for r in w_up]
        self._w_down = [[float(v) for v in r] for r in w_down]
        self._b_out = [float(v) for v in b_out]

        res = self.router.resolve_routes(
            ground_fingerprint=self.ground_fingerprint,
            operator_contract="tensor_cut.swiglu.v1",
            now=now,
            metric_order=metric_order,
        )
        if res.status is not FabricResolutionStatus.FOUND or not res.routes:
            raise FabricRoutingError(f"No eligible tensor-cut fabric hosts found: {res.blockers}")

        step = d_ff // partition_count
        claims: list[PortableExecutionClaim] = []
        self._shard_manifests.clear()

        for k in range(partition_count):
            lo, hi = k * step, (k + 1) * step
            raw_shard = _pack_swiglu_shard(
                self._w_gate[lo:hi],
                self._w_up[lo:hi],
                [row[lo:hi] for row in self._w_down],
            )
            m = self.origin_store.put_bytes(raw_shard)
            self._shard_manifests.append(m)
            route = res.routes[k % len(res.routes)]
            adv = route.target_host
            w_client = MTLSPeerClient(
                host=adv.endpoint_host,
                port=adv.port,
                expected_server_dns=adv.tls_dns,
                ssl_context=self._client_ssl,
            )
            w_client.invoke_control(
                {
                    "op": "provision_shard",
                    "origin_dns": self._coordinator_dns,
                    "origin_host": self._origin_addr[0],
                    "origin_port": self._origin_addr[1],
                    "shard_manifest_digest": m.manifest_digest,
                }
            )
            self.cold_provision_bytes += len(raw_shard)
            self._active_epochs[k] = 1
            self._active_routes[k] = route
            claims.append(
                PortableExecutionClaim.mint(
                    self._coord_secret,
                    claim_id=f"{self.cut_id}.part.{k}",
                    cut_id=self.cut_id,
                    partition_index=k,
                    assignment_epoch=1,
                    assigned_peer_id=adv.peer_id,
                    assigned_host_id=adv.host_id,
                    route_witness_digest=route.witness_digest,
                    shard_manifest_digest=m.manifest_digest,
                )
            )
        return tuple(claims)

    def verify_receipt_epoch(self, receipt: Mapping[str, Any]) -> None:
        p_idx = int(receipt["partition_index"])
        ep = int(receipt["assignment_epoch"])
        cur_ep = self._active_epochs.get(p_idx, 0)
        if ep != cur_ep:
            raise StaleClaimEpochError(
                f"Stale execution claim epoch {ep} for partition {p_idx} (current fenced epoch is {cur_ep})."
            )

    def execute_warm_cut(
        self,
        inputs_2d: Sequence[Sequence[float]],
        *,
        now: int = 12,
        metric_order: Sequence[tuple[str, str]] = (("latency_ms", "ms"), ("bytes", "B")),
        atol: float = 1e-4,
    ) -> PartitionedCutReport:
        """Execute warm activation-only partitioned SwiGLU cut with automatic crash reassignment & scalar oracle check."""
        assert self._origin_addr is not None
        seq_len = len(inputs_2d)
        d_in = len(inputs_2d[0])
        d_out = len(self._b_out)
        p_count = len(self._shard_manifests)
        act_bytes = encode_f32_le([float(v) for row in inputs_2d for v in row])
        act_sha256 = hashlib.sha256(act_bytes).hexdigest()
        act_hex = act_bytes.hex()

        partials: list[list[list[float]]] = []
        used_claims: list[PortableExecutionClaim] = []
        warm_wire_bytes = 0
        reassignments = 0

        for k in range(p_count):
            shard_m = self._shard_manifests[k]
            route = self._active_routes[k]
            epoch = self._active_epochs[k]

            def _try_invoke(rt: FabricRouteWitness, ep: int) -> tuple[PortableExecutionClaim, dict[str, Any]]:
                adv = rt.target_host
                clm = PortableExecutionClaim.mint(
                    self._coord_secret,
                    claim_id=f"{self.cut_id}.part.{k}",
                    cut_id=self.cut_id,
                    partition_index=k,
                    assignment_epoch=ep,
                    assigned_peer_id=adv.peer_id,
                    assigned_host_id=adv.host_id,
                    route_witness_digest=rt.witness_digest,
                    shard_manifest_digest=shard_m.manifest_digest,
                )
                w_client = MTLSPeerClient(
                    host=adv.endpoint_host,
                    port=adv.port,
                    expected_server_dns=adv.tls_dns,
                    ssl_context=self._client_ssl,
                )
                resp = w_client.invoke_control(
                    {
                        "activations_f32_hex": act_hex,
                        "claim": clm.to_dict(),
                        "d_in": d_in,
                        "op": "execute_warm_cut",
                        "seq_len": seq_len,
                    }
                )
                return clm, resp

            try:
                claim_used, resp = _try_invoke(route, epoch)
            except Exception:
                # Durable pre-reassignment intent + epoch increment fencing old host!
                failed_peer = route.target_host.peer_id
                new_epoch = epoch + 1
                self._active_epochs[k] = new_epoch
                self.journal.record(
                    request_id=f"reassign-{self.cut_id}-{k}-ep{new_epoch}",
                    ground_fingerprint=self.ground_fingerprint,
                    operation="fabric.reassign_partition",
                    payload={"failed_peer": failed_peer, "new_epoch": new_epoch, "partition": k},
                    result_fn=lambda: {"fenced_epoch": epoch, "new_epoch": new_epoch},
                )
                backup_res = self.router.resolve_routes(
                    ground_fingerprint=self.ground_fingerprint,
                    operator_contract="tensor_cut.swiglu.v1",
                    now=now,
                    metric_order=metric_order,
                    excluded_peers=frozenset({failed_peer}),
                )
                if backup_res.status is not FabricResolutionStatus.FOUND or not backup_res.routes:
                    raise FabricRoutingError(f"Partition {k} reassignment failed; no backup host available.")
                new_route = backup_res.routes[0]
                self._active_routes[k] = new_route
                adv_b = new_route.target_host
                b_client = MTLSPeerClient(
                    host=adv_b.endpoint_host,
                    port=adv_b.port,
                    expected_server_dns=adv_b.tls_dns,
                    ssl_context=self._client_ssl,
                )
                b_client.invoke_control(
                    {
                        "op": "provision_shard",
                        "origin_dns": self._coordinator_dns,
                        "origin_host": self._origin_addr[0],
                        "origin_port": self._origin_addr[1],
                        "shard_manifest_digest": shard_m.manifest_digest,
                    }
                )
                reassignments += 1
                claim_used, resp = _try_invoke(new_route, new_epoch)

            rcpt = resp["receipt"]
            self.verify_receipt_epoch(rcpt)
            expected_adv = self._active_routes[k].target_host
            w_dns = expected_adv.tls_dns
            w_sec = self._worker_secrets[w_dns]
            rcpt_body = {key: val for key, val in rcpt.items() if key != "worker_mac"}
            exp_mac = hmac.new(w_sec, canonical_bytes(rcpt_body), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(str(rcpt["worker_mac"]), exp_mac):
                raise FabricRoutingError(f"Forged worker receipt MAC on partition {k}.")
            if (
                str(rcpt.get("cut_id")) != self.cut_id
                or int(rcpt.get("partition_index", -1)) != k
                or str(rcpt.get("claim_id")) != claim_used.claim_id
                or str(rcpt.get("peer_id")) != expected_adv.peer_id
                or str(rcpt.get("host_id")) != expected_adv.host_id
                or str(rcpt.get("shard_manifest_digest")) != shard_m.manifest_digest
                or str(rcpt.get("activation_sha256")) != act_sha256
            ):
                raise FabricRoutingError(f"Worker receipt identity/input mismatch on partition {k}.")

            p_raw = bytes.fromhex(str(resp["partial_f32_hex"]))
            actual_p_sha = hashlib.sha256(p_raw).hexdigest()
            if actual_p_sha != str(resp["partial_sha256"]) or actual_p_sha != str(rcpt.get("partial_sha256")):
                raise FabricRoutingError(f"Corrupted or unauthenticated partial bytes on partition {k}.")
            warm_wire_bytes += len(act_bytes) + len(p_raw)
            p_vals = decode_f32_le(p_raw)
            partials.append([p_vals[t * d_out : (t + 1) * d_out] for t in range(seq_len)])
            used_claims.append(claim_used)

        # Commit partition leaves and exact reduction node in NavigableComputationDAG
        dag = NavigableComputationDAG(f"dag.{self.cut_id}")
        ctx = self.ground_fingerprint
        leaf_ids: list[str] = []
        for k in range(p_count):
            nid = f"part_{k}"
            leaf_ids.append(nid)
            dag.add_node(ComputationNode(nid, 1, "tensor_cut_part", "/f32", ctx, (), {"partition": k}))
            dag.commit_result(nid, {"claim_id": used_claims[k].claim_id, "epoch": used_claims[k].assignment_epoch})

        dag.add_node(ComputationNode("reduce_sum_bias", 1, "tensor_cut_reduce", "/f32", ctx, tuple(leaf_ids), {}))
        reduced: list[list[float]] = []
        for t in range(seq_len):
            row_sum: list[float] = []
            for o_i in range(d_out):
                # Apply b_out EXACTLY ONCE at coordinator reduction!
                row_sum.append(float(self._b_out[o_i]) + math.fsum(partials[k][t][o_i] for k in range(p_count)))
            reduced.append(row_sum)
        dag.commit_result("reduce_sum_bias", {"rows": seq_len, "cols": d_out})

        # Independent `math.fsum` scalar FP64 oracle check across output coordinates (catches rehashed false arithmetic!)
        max_err = 0.0
        d_ff = len(self._w_gate)
        for t in range(seq_len):
            x_vec = [float(v) for v in inputs_2d[t]]
            act_ff: list[float] = []
            for j in range(d_ff):
                gv = math.fsum(self._w_gate[j][i] * x_vec[i] for i in range(d_in))
                uv = math.fsum(self._w_up[j][i] * x_vec[i] for i in range(d_in))
                sig = 1.0 / (1.0 + math.exp(-gv)) if gv >= 0.0 else math.exp(gv) / (1.0 + math.exp(gv))
                act_ff.append((gv * sig) * uv)
            for o_i in range(d_out):
                expected_val = self._b_out[o_i] + math.fsum(self._w_down[o_i][j] * act_ff[j] for j in range(d_ff))
                max_err = max(max_err, abs(reduced[t][o_i] - expected_val))

        if max_err > atol:
            raise FabricRoutingError(
                f"Partitioned tensor-cut failed independent scalar oracle check (max_err={max_err:.3e} > {atol}); "
                "worker returned false arithmetic."
            )

        ckpt = dag.export_checkpoint()
        return PartitionedCutReport(
            cut_id=self.cut_id,
            partition_count=p_count,
            output_2d=reduced,
            cold_provision_bytes=self.cold_provision_bytes,
            warm_activation_bytes_on_wire=warm_wire_bytes,
            warm_weight_bytes_on_wire=0,
            reassignments_performed=reassignments,
            scalar_oracle_max_diff=max_err,
            dag_checkpoint_digest=str(ckpt["checkpoint_digest"]),
            claims=tuple(used_claims),
        )


__all__ = [
    "ComputationalFabricRouter",
    "FabricBlockerCode",
    "FabricLink",
    "FabricResolutionResult",
    "FabricResolutionStatus",
    "FabricRouteWitness",
    "FabricRoutingError",
    "HostAdvertisement",
    "PartitionedCutReport",
    "PartitionedTensorCutCoordinator",
    "PortableExecutionClaim",
    "RouteMetric",
    "StaleClaimEpochError",
    "TensorCutWorkerNode",
]
