r"""Specialized Internet-Path & Kernel IPv4 Router-Hop Grounds (`vld.substrates.internet_paths`, `G24`, `PATH-GROUND-004`, `INTERNET-004`, `INTERNET-005`).

Consolidates three network/Internet observation-to-Ground experiments with strict epistemic & negative-control boundaries:
1. **Kernel IPv4 Forwarding Hop Ground (`PATH-GROUND-004`):**
   Reconstructs raw Ethernet/IPv4/UDP packet headers (`RFC 791`, `RFC 1071` ones'-complement header checksum)
   across sender, intermediate network interfaces, and receiver `IP_RECVTTL`.
   - *Mandatory Negative Control:* An L2 Ethernet bridge (`route_id="bridge"`) physically forwards every frame
     with $\Delta \text{TTL} = 0$, proving intermediate physical transit $\neq$ L3 TTL decrement!
   - *Admitted Hop Law:* Only routes where every intermediate router capture witnesses $\text{TTL}_{\text{out}} = \text{TTL}_{\text{in}} - 1$
     with valid recomputed IPv4 checksums (`0xFFFF`) qualify as $h$-hop affine TTL Grounds $T(\text{ttl}) = (\text{ttl} - h) \bmod 256$.
2. **Bounded Redirect-Refusing Hypertext Link-Graph Observation (`INTERNET-004`):**
   Host-pinned, redirect-refusing BFS over canonical RFC-style hypertext documents (`https://<host>/rfc/rfc<N>.html`),
   emitting a detached observation capsule (`scope="observed-https-content-and-links-not-external-attestation"`)
   that an offline verifier reconstructs from raw bytes, SHA-256 digests, and HTML `<a href>` parsing without
   ever promoting retrieved text to executable code.
3. **Evidence-Gated Internet Observation Router (`INTERNET-005`):**
   Routes desired representation transitions (`source_rep -> target_rep`) over receiver-verified
   `InternetObservationCase`s (`git`, `http`, `dns`, `rfc`, `law`) subject to monotonic logical leases
   (`observed_at <= now <= expires_at`) and lexicographic `RouteMetric` ceilings.
"""

from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass
import hashlib
from html.parser import HTMLParser
import ipaddress
import re
import socket
import struct
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urldefrag, urljoin, urlsplit

from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.fabric.routing import RouteMetric
from vld.ground import AffineOrbitLaw, VirtualGroundSpec

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_RFC_PATH_RE = re.compile(r"/rfc/rfc([1-9][0-9]{0,4})\.html\Z")
_RFC_REF_RE = re.compile(r"/rfc/rfc([1-9][0-9]{0,4})(?:\.html)?\Z")
_PROVIDER_RE = re.compile(r"[a-z][a-z0-9_.-]{0,47}\Z")

TTL_PROBES: tuple[int, ...] = (4, 7, 32, 33, 64, 127)
ROUTER_MAGIC = b"VLD-PG4:"
ROUTER_TOPOLOGIES: dict[str, tuple[str, ...]] = {
    "direct": (),
    "bridge": ("bridge.r1",),
    "one": ("one.r1",),
    "two": ("two.r1", "two.r2"),
}
ROUTER_L3_HOPS: dict[str, int] = {"direct": 0, "bridge": 0, "one": 1, "two": 2}
LOCUS_TTL_DELTA: dict[str, int] = {"bridge.r1": 0, "one.r1": -1, "two.r1": -1, "two.r2": -1}

ALLOWED_INTERNET_KIND_OPS: dict[str, frozenset[str]] = {
    "git": frozenset({"observe.graph", "observe.object"}),
    "http": frozenset({"observe.content", "observe.conditional", "observe.range"}),
    "dns": frozenset({"observe.rrset"}),
    "rfc": frozenset({"navigate.references"}),
    "law": frozenset({"navigate.affine"}),
}


class InternetPathRefused(ValueError):
    """Raised when an Internet/router path observation, negative control, or capsule check fails."""


# ============================================================================
# 1. Kernel IPv4 Router-Hop Ground & L2 Bridge Negative Control (`PATH-GROUND-004`)
# ============================================================================


def _ones_complement_ipv4_checksum(header20: bytes) -> int:
    words = struct.unpack("!10H", header20)
    acc = sum(words)
    while acc >> 16:
        acc = (acc & 0xFFFF) + (acc >> 16)
    return (~acc) & 0xFFFF


def build_probe_payload(sequence: int) -> bytes:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not (0 <= sequence < len(TTL_PROBES)):
        raise InternetPathRefused("probe sequence out of bounds.")
    body = ROUTER_MAGIC + sequence.to_bytes(2, "big")
    return body + hashlib.sha256(body).digest()[:4]


def build_ethernet_ipv4_udp_frame(
    *,
    sequence: int,
    ttl: int,
    source_ip: str = "10.10.1.2",
    destination_ip: str = "10.10.3.2",
    source_port: int = 41000,
    destination_port: int = 42000,
    ip_id: int = 0x1A2B,
) -> bytes:
    """Construct a valid Ethernet II + IPv4 (RFC 791/1071) + UDP frame for controlled router-hop testing."""
    if not (1 <= ttl <= 255):
        raise InternetPathRefused("IPv4 TTL must lie in [1, 255].")
    src_addr = ipaddress.IPv4Address(source_ip)
    dst_addr = ipaddress.IPv4Address(destination_ip)
    if not (src_addr.is_private and dst_addr.is_private) or src_addr.is_loopback or dst_addr.is_loopback or src_addr == dst_addr:
        raise InternetPathRefused("Controlled router Ground requires distinct private non-loopback IPv4 endpoints.")

    payload = build_probe_payload(sequence)
    udp_len = 8 + len(payload)
    udp_hdr = struct.pack("!HHHH", source_port, destination_port, udp_len, 0)
    tot_len = 20 + udp_len
    ip_no_csum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        tot_len,
        ip_id & 0xFFFF,
        0x4000,  # DF bit, unfragmented
        ttl,
        socket.IPPROTO_UDP,
        0,
        src_addr.packed,
        dst_addr.packed,
    )
    csum = _ones_complement_ipv4_checksum(ip_no_csum)
    ip_hdr = ip_no_csum[:10] + struct.pack("!H", csum) + ip_no_csum[12:]
    eth_hdr = b"\x02\x00\x00\x00\x00\x02\x02\x00\x00\x00\x00\x01\x08\x00"
    return eth_hdr + ip_hdr + udp_hdr + payload


def parse_ethernet_ipv4_udp_frame(frame: bytes) -> dict[str, Any]:
    """Independently decode Ethernet/IPv4/UDP bytes and verify RFC 1071 ones'-complement IPv4 checksum."""
    if not isinstance(frame, bytes) or not (42 <= len(frame) <= 2048) or frame[12:14] != b"\x08\x00":
        raise InternetPathRefused("Frame is not bounded Ethernet II IPv4.")
    ip = frame[14:]
    ver_ihl = ip[0]
    ihl = (ver_ihl & 0x0F) * 4
    if (ver_ihl >> 4) != 4 or ihl != 20 or len(ip) < 28:
        raise InternetPathRefused("Unsupported or malformed IPv4 header.")
    tot_len = int.from_bytes(ip[2:4], "big")
    if tot_len != len(ip):
        raise InternetPathRefused("IPv4 total_length disagrees with frame byte length.")
    if ip[9] != socket.IPPROTO_UDP or (int.from_bytes(ip[6:8], "big") & 0x3FFF) != 0:
        raise InternetPathRefused("Only unfragmented UDP/IPv4 probe frames are permitted.")
    words = struct.unpack("!10H", ip[:20])
    acc = sum(words)
    while acc >> 16:
        acc = (acc & 0xFFFF) + (acc >> 16)
    if acc != 0xFFFF:
        raise InternetPathRefused("IPv4 header failed RFC 1071 checksum verification.")

    udp = ip[20:]
    if int.from_bytes(udp[4:6], "big") != len(udp):
        raise InternetPathRefused("UDP length header mismatch.")
    payload = udp[8:]
    if not payload.startswith(ROUTER_MAGIC) or len(payload) != len(ROUTER_MAGIC) + 6:
        raise InternetPathRefused("Unexpected probe payload magic or length.")
    seq = int.from_bytes(payload[len(ROUTER_MAGIC) : len(ROUTER_MAGIC) + 2], "big")
    if payload != build_probe_payload(seq):
        raise InternetPathRefused("Corrupted probe payload digest.")
    return {
        "destination_ip": socket.inet_ntoa(ip[16:20]),
        "destination_port": int.from_bytes(udp[2:4], "big"),
        "ip_id": int.from_bytes(ip[4:6], "big"),
        "payload_hex": payload.hex(),
        "seq": seq,
        "source_ip": socket.inet_ntoa(ip[12:16]),
        "source_port": int.from_bytes(udp[0:2], "big"),
        "ttl": ip[8],
    }


@dataclass(frozen=True, slots=True)
class VerifiedRouterHopGround:
    """Verified kernel IPv4 hop observation distinguishing L3 TTL decrements from L2 bridge forwarding."""

    route_id: str
    l3_hops: int
    source_ttls: tuple[int, ...]
    received_ttls: tuple[int, ...]
    performs_in_path_ttl_transform: bool
    certified_ttl_law: AffineOrbitLaw | None
    evidence_digest: str


def verify_ipv4_router_hop_ground(
    route_id: str,
    *,
    sender_frames: Sequence[bytes],
    hop_captures: Sequence[tuple[str, Sequence[bytes], Sequence[bytes]]],  # (locus_id, ingress_frames, egress_frames)
    receiver_ttls: Sequence[int],
    require_in_path_transform: bool = False,
) -> VerifiedRouterHopGround:
    """Verify ingress/egress frames across every expected hop and enforce the L2 bridge negative control."""
    if route_id not in ROUTER_TOPOLOGIES:
        raise InternetPathRefused(f"Unknown route_id {route_id!r}.")
    expected_loci = ROUTER_TOPOLOGIES[route_id]
    if len(sender_frames) != len(TTL_PROBES) or len(receiver_ttls) != len(TTL_PROBES):
        raise InternetPathRefused("Probe count must match TTL_PROBES.")
    if len(hop_captures) != len(expected_loci):
        raise InternetPathRefused("Captured intermediate hop count disagrees with topology.")

    cur_ttls = list(TTL_PROBES)
    parsed_src: list[dict[str, Any]] = []
    for idx, sf in enumerate(sender_frames):
        p = parse_ethernet_ipv4_udp_frame(sf)
        if p["seq"] != idx or p["ttl"] != TTL_PROBES[idx]:
            raise InternetPathRefused("Sender frame sequence or initial TTL mismatch.")
        parsed_src.append(p)

    for (exp_locus, (locus_id, in_frames, out_frames)) in zip(expected_loci, hop_captures):
        if locus_id != exp_locus or len(in_frames) != len(TTL_PROBES) or len(out_frames) != len(TTL_PROBES):
            raise InternetPathRefused(f"Malformed capture at locus {locus_id!r}.")
        delta = LOCUS_TTL_DELTA[locus_id]
        next_ttls: list[int] = []
        for idx in range(len(TTL_PROBES)):
            pin = parse_ethernet_ipv4_udp_frame(in_frames[idx])
            pout = parse_ethernet_ipv4_udp_frame(out_frames[idx])
            if pin["seq"] != idx or pout["seq"] != idx:
                raise InternetPathRefused("Hop sequence mismatch.")
            if pin["payload_hex"] != parsed_src[idx]["payload_hex"] or pout["payload_hex"] != parsed_src[idx]["payload_hex"]:
                raise InternetPathRefused("Hop mutated UDP payload bytes.")
            if pin["ttl"] != cur_ttls[idx]:
                raise InternetPathRefused(f"Locus {locus_id!r} ingress TTL {pin['ttl']} != expected {cur_ttls[idx]}.")
            if pout["ttl"] != pin["ttl"] + delta:
                raise InternetPathRefused(
                    f"Locus {locus_id!r} egress TTL {pout['ttl']} != {pin['ttl'] + delta}."
                )
            next_ttls.append(pout["ttl"])
        cur_ttls = next_ttls

    recv_tup = tuple(int(t) for t in receiver_ttls)
    if recv_tup != tuple(cur_ttls):
        raise InternetPathRefused("Receiver IP_RECVTTL disagrees with verified hop egress TTLs.")

    hops = ROUTER_L3_HOPS[route_id]
    transforms = hops > 0
    if require_in_path_transform and not transforms:
        raise InternetPathRefused(
            f"Negative control triggered: route {route_id!r} (e.g. L2 Ethernet bridge) performs zero L3 TTL transformation."
        )

    law = AffineOrbitLaw(multiplier=1, increment=(-hops) % 256, modulus=256) if transforms else None
    ev_digest = domain_digest(
        "vld.ipv4-router-ground.v1",
        {"hops": hops, "received_ttls": list(recv_tup), "route_id": route_id, "source_ttls": list(TTL_PROBES)},
    )
    return VerifiedRouterHopGround(
        route_id=route_id,
        l3_hops=hops,
        source_ttls=TTL_PROBES,
        received_ttls=recv_tup,
        performs_in_path_ttl_transform=transforms,
        certified_ttl_law=law,
        evidence_digest=ev_digest,
    )


# ============================================================================
# 2. Bounded Redirect-Refusing Hypertext Link-Graph Observation (`INTERNET-004`)
# ============================================================================


def canonical_rfc_url(number: int, *, allowed_host: str = "www.rfc-editor.org") -> str:
    if isinstance(number, bool) or not isinstance(number, int) or not (1 <= number <= 99999):
        raise InternetPathRefused("RFC number must lie in [1, 99999].")
    return f"https://{allowed_host}/rfc/rfc{number}.html"


def normalize_permitted_rfc_url(url: str, *, allowed_host: str = "www.rfc-editor.org") -> str:
    if not isinstance(url, str) or len(url) > 256:
        raise InternetPathRefused("URL exceeds bounded text length.")
    parsed = urlsplit(url)
    m = _RFC_PATH_RE.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != allowed_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or m is None
    ):
        raise InternetPathRefused(f"URL {url!r} outside explicitly permitted RFC Ground.")
    canonical = canonical_rfc_url(int(m.group(1)), allowed_host=allowed_host)
    if canonical != parsed._replace(fragment="").geturl():
        raise InternetPathRefused("Non-canonical RFC URL rejected.")
    return canonical


class _BoundedRFCLinkParser(HTMLParser):
    def __init__(self, source_url: str, allowed_host: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.allowed_host = allowed_host
        self.links: set[str] = set()
        self._in_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if isinstance(href, str):
                cand, _frag = urldefrag(urljoin(self.source_url, href))
                parsed = urlsplit(cand)
                m = _RFC_REF_RE.fullmatch(parsed.path)
                if parsed.scheme == "https" and parsed.netloc == self.allowed_host and not parsed.query and m is not None:
                    self.links.add(canonical_rfc_url(int(m.group(1)), allowed_host=self.allowed_host))

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and sum(len(x) for x in self._title_parts) < 200:
            self._title_parts.append(data[:200])


def extract_rfc_title_and_links(
    source_url: str, raw_html: bytes, *, allowed_host: str = "www.rfc-editor.org"
) -> tuple[str, tuple[str, ...]]:
    norm_src = normalize_permitted_rfc_url(source_url, allowed_host=allowed_host)
    try:
        text = raw_html.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InternetPathRefused("HTML observation is not valid UTF-8.") from exc
    parser = _BoundedRFCLinkParser(norm_src, allowed_host)
    parser.feed(text)
    title = " ".join(" ".join(parser._title_parts).split())[:160]
    return title, tuple(sorted(parser.links))


@dataclass(frozen=True, slots=True)
class InternetNavigationPolicy:
    seeds: tuple[int, ...]
    target: int
    allowed_host: str = "www.rfc-editor.org"
    max_pages: int = 6
    max_depth: int = 2
    page_budget: int = 500_000
    total_budget: int = 1_500_000

    def __post_init__(self) -> None:
        if not (1 <= len(self.seeds) <= 4) or len(set(self.seeds)) != len(self.seeds):
            raise InternetPathRefused("1..4 distinct RFC seeds required.")
        for s in (*self.seeds, self.target):
            canonical_rfc_url(s, allowed_host=self.allowed_host)
        if not (1 <= self.max_pages <= 12) or not (0 <= self.max_depth <= 3) or self.total_budget < self.page_budget:
            raise InternetPathRefused("Invalid InternetNavigationPolicy bounds.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_host": self.allowed_host,
            "max_depth": self.max_depth,
            "max_pages": self.max_pages,
            "page_budget": self.page_budget,
            "redirects_allowed": False,
            "seeds": list(self.seeds),
            "target": self.target,
            "total_budget": self.total_budget,
        }


def navigate_bounded_hypertext_ground(
    policy: InternetNavigationPolicy,
    fetcher: Callable[[str], tuple[int, str, bytes]],  # returns (status, content_type, body)
) -> dict[str, Any]:
    """Execute bounded BFS over a pinned RFC hypertext Ground, refusing redirects & non-HTML responses."""
    target_url = canonical_rfc_url(policy.target, allowed_host=policy.allowed_host)
    queue: deque[tuple[str, str | None, int]] = deque(
        (canonical_rfc_url(s, allowed_host=policy.allowed_host), None, 0) for s in policy.seeds
    )
    visited: set[str] = set()
    parent_of: dict[str, str | None] = {}
    visits: list[dict[str, Any]] = []
    bytes_used = 0
    found = False

    while queue and len(visits) < policy.max_pages:
        url, parent, depth = queue.popleft()
        if url in visited or depth > policy.max_depth:
            continue
        visited.add(url)
        status, ctype, raw = fetcher(url)
        if status in (301, 302, 303, 307, 308):
            raise InternetPathRefused(f"HTTP redirect ({status}) refused by InternetNavigationPolicy.")
        if status != 200 or ctype.split(";", 1)[0].strip().lower() not in ("text/html", "application/xhtml+xml"):
            raise InternetPathRefused(f"Invalid HTTP status {status} or content-type {ctype!r}.")
        if len(raw) > policy.page_budget or bytes_used + len(raw) > policy.total_budget:
            raise InternetPathRefused("HTML response exceeded byte budget.")
        bytes_used += len(raw)
        title, links = extract_rfc_title_and_links(url, raw, allowed_host=policy.allowed_host)
        parent_of[url] = parent
        visits.append(
            {
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "depth": depth,
                "html_b64": base64.b64encode(raw).decode("ascii"),
                "links": list(links),
                "media_type": ctype.split(";", 1)[0].strip().lower(),
                "parent": parent,
                "size_bytes": len(raw),
                "status": status,
                "title": title,
                "url": url,
            }
        )
        if url == target_url or target_url in links:
            if url != target_url:
                parent_of[target_url] = url
            found = True
            break
        if depth + 1 <= policy.max_depth:
            for lk in links:
                if lk not in visited:
                    queue.append((lk, url, depth + 1))

    path: list[str] = []
    if found:
        cur: str | None = target_url
        while cur is not None:
            path.append(cur)
            cur = parent_of.get(cur)
        path.reverse()

    body = {
        "path": path,
        "policy": policy.to_dict(),
        "schema": "vld.public-internet-navigation.v1",
        "scope": "observed-https-content-and-links-not-external-attestation",
        "status": "reached" if found else "exhausted",
        "visits": visits,
    }
    capsule_digest = domain_digest("vld.internet-capsule.v1", body)
    return {**body, "capsule_digest": capsule_digest}


def verify_internet_navigation_capsule(
    capsule: Mapping[str, Any],
    policy: InternetNavigationPolicy,
    *,
    pinned_sha256_by_url: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Independently reconstruct bytes, SHA-256 hashes, extracted links, and BFS lineage offline."""
    req = {"capsule_digest", "path", "policy", "schema", "scope", "status", "visits"}
    if set(capsule.keys()) != req or capsule["schema"] != "vld.public-internet-navigation.v1":
        raise InternetPathRefused("Invalid Internet navigation capsule schema.")
    if capsule["policy"] != policy.to_dict() or capsule["scope"] != "observed-https-content-and-links-not-external-attestation":
        raise InternetPathRefused("Capsule policy or epistemic scope mismatch.")
    body = {k: v for k, v in capsule.items() if k != "capsule_digest"}
    if domain_digest("vld.internet-capsule.v1", body) != capsule["capsule_digest"]:
        raise InternetPathRefused("Capsule digest tampering detected.")

    visits = capsule["visits"]
    if not isinstance(visits, list) or not (1 <= len(visits) <= policy.max_pages):
        raise InternetPathRefused("Capsule visit count out of bounds.")

    by_url: dict[str, dict[str, Any]] = {}
    tot_bytes = 0
    for v in visits:
        url = normalize_permitted_rfc_url(str(v["url"]), allowed_host=policy.allowed_host)
        raw = base64.b64decode(str(v["html_b64"]), validate=True)
        tot_bytes += len(raw)
        if len(raw) != int(v["size_bytes"]) or len(raw) > policy.page_budget or tot_bytes > policy.total_budget:
            raise InternetPathRefused("Retained HTML size mismatch or budget overrun.")
        sha = hashlib.sha256(raw).hexdigest()
        if sha != str(v["content_sha256"]):
            raise InternetPathRefused("Retained HTML SHA-256 mismatch.")
        if pinned_sha256_by_url and url in pinned_sha256_by_url and sha != pinned_sha256_by_url[url]:
            raise InternetPathRefused(f"Pinned content hash mismatch on {url!r}.")
        title, links = extract_rfc_title_and_links(url, raw, allowed_host=policy.allowed_host)
        if title != v["title"] or list(links) != v["links"]:
            raise InternetPathRefused("Capsule title or extracted links disagree with raw HTML bytes.")
        if v["parent"] is not None:
            p_url = str(v["parent"])
            if p_url not in by_url or url not in by_url[p_url]["links"]:
                raise InternetPathRefused("Visit parent was not witnessed earlier or lacked hyperlink.")
        by_url[url] = v

    if capsule["status"] == "reached":
        path = list(capsule["path"])
        target_url = canonical_rfc_url(policy.target, allowed_host=policy.allowed_host)
        seed_urls = {canonical_rfc_url(s, allowed_host=policy.allowed_host) for s in policy.seeds}
        if not path or path[0] not in seed_urls or path[-1] != target_url:
            raise InternetPathRefused("Invalid path endpoints in reached capsule.")
        for a, b in zip(path, path[1:]):
            if a not in by_url or b not in by_url[a]["links"]:
                raise InternetPathRefused("Claimed path edge was not witnessed in HTML links.")

    return {"capsule_digest": capsule["capsule_digest"], "status": capsule["status"], "verified_pages": len(visits)}


# ============================================================================
# 3. Evidence-Gated Internet Observation Router (`INTERNET-005`)
# ============================================================================


@dataclass(frozen=True, slots=True)
class InternetObservationCase:
    """One receiver-verified, lease-bounded Internet observation transition `source_rep -> target_rep`."""

    provider_id: str
    kind: str
    operation: str
    source_rep: str
    target_rep: str
    ground_fingerprint: str
    observed_at: int
    expires_at: int
    expected_projection: Any
    observed_projection: Any
    costs: tuple[RouteMetric, ...]

    def __post_init__(self) -> None:
        if not _PROVIDER_RE.fullmatch(self.provider_id):
            raise InternetPathRefused("Invalid provider_id.")
        if self.kind not in ALLOWED_INTERNET_KIND_OPS or self.operation not in ALLOWED_INTERNET_KIND_OPS[self.kind]:
            raise InternetPathRefused(f"Unsupported Internet observation ({self.kind!r}, {self.operation!r}).")
        if not self.source_rep or not self.target_rep or self.source_rep == self.target_rep:
            raise InternetPathRefused("Observation case must advance between distinct representations.")
        if not _SHA256_RE.fullmatch(self.ground_fingerprint):
            raise InternetPathRefused("ground_fingerprint must be 64-char hex.")
        if self.observed_at < 0 or self.expires_at < self.observed_at:
            raise InternetPathRefused("Invalid monotonic lease bounds.")
        object.__setattr__(self, "expected_projection", parse_strict_json(canonical_json(self.expected_projection)))
        object.__setattr__(self, "observed_projection", parse_strict_json(canonical_json(self.observed_projection)))


class InternetObservationRouter:
    """Selects optimal multi-hop observation paths using only receiver-verified, non-expired cases (`INTERNET-005`)."""

    def __init__(self, verifiers_by_kind: Mapping[str, Callable[[InternetObservationCase], bool]]) -> None:
        self._verifiers = dict(verifiers_by_kind)
        self._cases: list[InternetObservationCase] = []

    def admit_case(self, case: InternetObservationCase) -> None:
        verifier = self._verifiers.get(case.kind)
        if verifier is None:
            raise InternetPathRefused(f"No receiver verifier installed for kind {case.kind!r}.")
        if canonical_bytes(case.expected_projection) != canonical_bytes(case.observed_projection):
            raise InternetPathRefused("Observed projection disagrees with receiver expected_projection.")
        if not verifier(case):
            raise InternetPathRefused(f"Receiver verifier rejected InternetObservationCase {case.provider_id!r}.")
        self._cases.append(case)

    def route_observation(
        self,
        *,
        source_rep: str,
        target_rep: str,
        ground_fingerprint: str,
        now: int,
        metric_order: Sequence[tuple[str, str]],
        max_hops: int = 5,
    ) -> tuple[InternetObservationCase, ...]:
        valid = [
            c
            for c in self._cases
            if c.ground_fingerprint == ground_fingerprint and (c.observed_at <= now <= c.expires_at)
        ]
        # Dijkstra over representation graph
        import heapq

        zero = tuple(0 for _ in metric_order)
        pq: list[tuple[tuple[int, ...], int, str, tuple[InternetObservationCase, ...]]] = [(zero, 0, source_rep, ())]
        best: dict[str, tuple[int, ...]] = {source_rep: zero}

        while pq:
            cost_vec, hops, cur_rep, chain = heapq.heappop(pq)
            if cur_rep == target_rep and chain:
                # Re-verify every selected case before returning!
                for c in chain:
                    if not self._verifiers[c.kind](c):
                        raise InternetPathRefused("Post-selection re-verification failed.")
                return chain
            if hops >= max_hops:
                continue
            for c in valid:
                if c.source_rep != cur_rep:
                    continue
                by_n = {m.name: m for m in c.costs}
                step_vec = tuple(by_n[m_n].amount for m_n, _u in metric_order)
                nxt_vec = tuple(a + b for a, b in zip(cost_vec, step_vec))
                prev = best.get(c.target_rep)
                if prev is None or nxt_vec < prev:
                    best[c.target_rep] = nxt_vec
                    heapq.heappush(pq, (nxt_vec, hops + 1, c.target_rep, (*chain, c)))

        raise InternetPathRefused(f"No verified non-expired observation route from {source_rep!r} to {target_rep!r}.")


__all__ = [
    "ALLOWED_INTERNET_KIND_OPS",
    "InternetNavigationPolicy",
    "InternetObservationCase",
    "InternetObservationRouter",
    "InternetPathRefused",
    "LOCUS_TTL_DELTA",
    "ROUTER_L3_HOPS",
    "ROUTER_TOPOLOGIES",
    "TTL_PROBES",
    "VerifiedRouterHopGround",
    "build_ethernet_ipv4_udp_frame",
    "build_probe_payload",
    "canonical_rfc_url",
    "extract_rfc_title_and_links",
    "navigate_bounded_hypertext_ground",
    "normalize_permitted_rfc_url",
    "parse_ethernet_ipv4_udp_frame",
    "verify_internet_navigation_capsule",
    "verify_ipv4_router_hop_ground",
]
