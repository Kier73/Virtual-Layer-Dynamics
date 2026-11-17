r"""Phase 5 — Active Host Formation & Internet / Path-Native Grounds (`vld.substrates.path_grounds`, `G26`–`G29`).

Mathematical contract:
An external network intermediary, HTTP conditional/Range cache, queue, or transit relay becomes an
admitted **Observer-Host** inside the Virtual Layer **only** after passing the six-stage pipeline:
  $$\text{observe} \longrightarrow \text{qualify} \longrightarrow \text{construct} \longrightarrow \text{admit} \longrightarrow \text{navigate} \longrightarrow \text{verify}$$

Mandatory Negative Controls (Defeating False Attribution of Endpoint Compute to the Path):
1. **Stateful In-Path Cache Qualification (`http_conditional_cache`):**
   - *Positive criterion:* Repeat `If-None-Match` (`304 Not Modified`) and `Range` (`206 Partial Content`)
     queries against the intermediary must match SHA-256 ground truth **while upstream origin hit count
     stays invariant** ($\Delta N_{\text{origin}} = 0$).
   - *Negative Control 1 (Pass-Through Origin Echo):* A stateless pass-through forwarder where
     $\Delta N_{\text{origin}} > 0$ performs zero in-path state retention and is **refused** (`PathQualificationRefused`).
   - *Negative Control 2 (Nondeterministic / Mutating Noise):* An intermediary returning inconsistent
     bytes or forged `ETag`s is **refused** (`PathQualificationRefused`).
2. **In-Transit Functional Projection (`transit_affine_hop`):**
   - *Positive criterion:* Authenticated transit hop $h$ applies declared deterministic byte law
     $T_h(x)_k = (a_h x_k + b_h) \bmod 256$ with verified witness MAC; multi-hop path $\pi = (h_0, \dots, h_m)$
     composes $F_\pi = T_{h_m} \circ \dots \circ T_{h_0}$ into a single $O(1)$ collapsed affine map!
   - *Negative Control (Identity Pass-Through / Forged Witness):* An endpoint-only compute box pretending
     the transit hop transformed bytes (when transit hop was identity or lacked valid witness MAC) is **refused**.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from vld.canonical import domain_digest
from vld.ground import AffineOrbitLaw, VirtualGroundSpec
from vld.journal import HashChainedJournal


class PathGroundError(ValueError):
    """Base error for path observation, qualification, or navigation."""


class PathQualificationRefused(PathGroundError):
    """Raised when a candidate vantage fails mandatory negative controls or repeatability qualification."""


# ============================================================================
# 1. Live HTTP Origin + In-Path Cache vs. Pass-Through Proxy Harnesses
# ============================================================================


class CountingHTTPOriginServer:
    r"""Authoritative HTTP origin tracking exact upstream fetch count $N_{\text{origin}}$."""

    def __init__(self, assets: Mapping[str, bytes], host: str = "127.0.0.1", port: int = 0) -> None:
        self.assets = dict(assets)
        self.origin_hits = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        addr, p = self._httpd.server_address[:2]
        return f"http://{addr}:{p}"

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            self._thread.start()
        return self.base_url

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> CountingHTTPOriginServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class _OriginHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                with parent._lock:
                    parent.origin_hits += 1
                key = self.path.lstrip("/")
                raw = parent.assets.get(key)
                if raw is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                etag = f"\"{hashlib.sha256(raw).hexdigest()[:32]}\""
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("ETag", etag)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return _OriginHandler


class HTTPPathIntermediaryServer:
    """Configurable HTTP intermediary: `mode='cache'` (genuine in-path state), `'passthrough'` (negative control), or `'noisy'`."""

    def __init__(
        self,
        origin_url: str,
        *,
        mode: str = "cache",
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        if mode not in ("cache", "passthrough", "noisy"):
            raise ValueError(f"Unsupported intermediary mode: {mode!r}")
        self.origin_url = origin_url.rstrip("/")
        self.mode = mode
        self._cache: dict[str, tuple[str, bytes]] = {}
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread: threading.Thread | None = None
        self._noise_counter = 0

    @property
    def base_url(self) -> str:
        addr, p = self._httpd.server_address[:2]
        return f"http://{addr}:{p}"

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            self._thread.start()
        return self.base_url

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> HTTPPathIntermediaryServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class _IntermediaryHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                key = self.path.lstrip("/")
                etag: str | None = None
                raw: bytes | None = None

                if parent.mode == "cache":
                    with parent._lock:
                        cached = parent._cache.get(key)
                    if cached is not None:
                        etag, raw = cached

                if raw is None:
                    try:
                        with urlopen(f"{parent.origin_url}/{key}", timeout=5.0) as resp:
                            raw = resp.read()
                            etag = resp.headers.get("ETag", f"\"{hashlib.sha256(raw).hexdigest()[:32]}\"")
                    except HTTPError as exc:
                        self.send_response(exc.code)
                        self.end_headers()
                        return
                    if parent.mode == "cache":
                        with parent._lock:
                            parent._cache[key] = (etag, raw)

                if parent.mode == "noisy":
                    with parent._lock:
                        parent._noise_counter += 1
                        c_val = parent._noise_counter
                    raw = bytes([(raw[0] ^ (c_val & 0xFF))]) + raw[1:] if raw else b"\x01"

                inm = self.headers.get("If-None-Match")
                if parent.mode == "cache" and inm is not None and inm == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag or "")
                    self.end_headers()
                    return

                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes=") and "-" in rng:
                    s_str, e_str = rng[6:].split("-", 1)
                    s_i, e_i = int(s_str), int(e_str)
                    if 0 <= s_i <= e_i < len(raw):
                        sl = raw[s_i : e_i + 1]
                        self.send_response(206)
                        self.send_header("ETag", etag or "")
                        self.send_header("Content-Range", f"bytes {s_i}-{e_i}/{len(raw)}")
                        self.send_header("Content-Length", str(len(sl)))
                        self.end_headers()
                        self.wfile.write(sl)
                        return

                self.send_response(200)
                self.send_header("ETag", etag or "")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return _IntermediaryHandler


# ============================================================================
# 2. Six-Stage Active Host Formation Lifecycle:
#    observe -> qualify -> construct -> admit -> navigate -> verify
# ============================================================================


@dataclass(frozen=True, slots=True)
class PathObservationEvidence:
    """Stage 1 (`observe`) telemetry captured from a candidate HTTP or transit path locus."""

    locus_id: str
    kind: str  # "http_conditional_cache" or "transit_affine_chain"
    observations: Mapping[str, Any]
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class QualifiedPathCertificate:
    """Stage 2 (`qualify`) certificate proving candidate passed both positive invariants and negative controls."""

    locus_id: str
    kind: str
    evidence_digest: str
    negative_controls_passed: tuple[str, ...]
    parameters: Mapping[str, Any]
    qualification_digest: str


@dataclass(frozen=True, slots=True)
class AdmittedPathHost:
    """Stages 3–4 (`construct` + `admit`) Virtual Layer host backed by a verified network/path locus."""

    locus_id: str
    ground_spec: VirtualGroundSpec
    certificate: QualifiedPathCertificate
    navigator: Callable[[Any], Any]


class ActiveHostFormationPipeline:
    r"""Executes `observe -> qualify -> construct -> admit -> navigate -> verify` with mandatory negative controls."""

    def __init__(self, journal: HashChainedJournal | None = None) -> None:
        self.journal = journal if journal is not None else HashChainedJournal()
        self._admitted: dict[str, AdmittedPathHost] = {}

    # --- HTTP Conditional / Range Cache Path ---

    def observe_http_cache_locus(
        self,
        locus_id: str,
        intermediary_url: str,
        asset_key: str,
        origin_hit_reader: Callable[[], int],
    ) -> PathObservationEvidence:
        """Stage 1 (`observe`): Probe cold fetch, conditional `If-None-Match`, and `Range` slice against origin hit counter."""
        url = f"{intermediary_url.rstrip('/')}/{asset_key}"
        hits_t0 = origin_hit_reader()
        with urlopen(Request(url, method="GET"), timeout=5.0) as r1:
            body_1 = r1.read()
            etag_1 = r1.headers.get("ETag", "")
            status_1 = r1.status
        hits_t1 = origin_hit_reader()

        # Conditional If-None-Match probe
        cond_status = 0
        try:
            with urlopen(Request(url, method="GET", headers={"If-None-Match": etag_1}), timeout=5.0) as r_cond:
                cond_status = r_cond.status
        except HTTPError as exc:
            cond_status = exc.code
        hits_t2 = origin_hit_reader()

        # Repeat slice Range probe + full repeat probe
        slice_len = min(8, len(body_1))
        range_body = b""
        range_status = 0
        if slice_len > 0:
            with urlopen(
                Request(url, method="GET", headers={"Range": f"bytes=0-{slice_len - 1}"}),
                timeout=5.0,
            ) as r_rng:
                range_status = r_rng.status
                range_body = r_rng.read()
        with urlopen(Request(url, method="GET"), timeout=5.0) as r2:
            body_2 = r2.read()
        hits_t3 = origin_hit_reader()

        obs = {
            "asset_key": asset_key,
            "body_1_sha256": hashlib.sha256(body_1).hexdigest(),
            "body_2_sha256": hashlib.sha256(body_2).hexdigest(),
            "cold_origin_delta": hits_t1 - hits_t0,
            "conditional_status": cond_status,
            "etag": etag_1,
            "intermediary_url": intermediary_url.rstrip("/"),
            "prefix_matches": range_body == body_1[:slice_len],
            "range_status": range_status,
            "size_bytes": len(body_1),
            "status_1": status_1,
            "warm_origin_delta": hits_t3 - hits_t1,
            "warm_post_cond_delta": hits_t2 - hits_t1,
        }
        return PathObservationEvidence(
            locus_id=locus_id,
            kind="http_conditional_cache",
            observations=obs,
            evidence_digest=domain_digest("vld.path-observation.v1", obs),
        )

    # --- Multi-Hop Authenticated Transit Projection Path ---

    def observe_transit_chain(
        self,
        locus_id: str,
        hops: Sequence[tuple[str, int, int, bytes, Callable[[bytes], tuple[bytes, str]]]],
        probe_inputs: Sequence[bytes],
    ) -> PathObservationEvidence:
        """Stage 1 (`observe`) for authenticated in-transit byte transformation hops $(h_i, a_i, b_i, k_i, f_i)$."""
        cur_probes = [bytes(p) for p in probe_inputs]
        if not hops or len(cur_probes) < 2 or len(set(cur_probes)) < 2 or any(len(p) == 0 for p in cur_probes):
            raise PathGroundError("Transit observation requires >= 1 hop and >= 2 distinct non-empty probe inputs.")

        hop_records: list[dict[str, Any]] = []
        for hop_id, decl_a, decl_b, secret, hop_fn in hops:
            next_probes: list[bytes] = []
            valid_macs = True
            matches_decl = True
            non_identity_seen = False
            for inp in cur_probes:
                out_b, witness_mac = hop_fn(inp)
                exp_mac = hmac.new(secret, hashlib.sha256(inp).digest() + out_b, hashlib.sha256).hexdigest()
                if not hmac.compare_digest(witness_mac, exp_mac):
                    valid_macs = False
                expected_b = bytes(((decl_a * x + decl_b) & 0xFF) for x in inp)
                if out_b != expected_b:
                    matches_decl = False
                if out_b != inp:
                    non_identity_seen = True
                next_probes.append(out_b)
            hop_records.append(
                {
                    "a": decl_a & 0xFF,
                    "b": decl_b & 0xFF,
                    "hop_id": hop_id,
                    "matches_declared_law": matches_decl,
                    "non_identity_transformation": non_identity_seen,
                    "valid_witness_mac": valid_macs,
                }
            )
            cur_probes = next_probes

        obs = {
            "final_probe_digests": [hashlib.sha256(p).hexdigest() for p in cur_probes],
            "hops": hop_records,
        }
        return PathObservationEvidence(
            locus_id=locus_id,
            kind="transit_affine_chain",
            observations=obs,
            evidence_digest=domain_digest("vld.path-observation.v1", obs),
        )

    def qualify(self, evidence: PathObservationEvidence) -> QualifiedPathCertificate:
        """Stage 2 (`qualify`): Enforce repeatability AND mandatory negative controls against endpoint-only fakes."""
        obs = evidence.observations
        if evidence.kind == "http_conditional_cache":
            if obs["status_1"] != 200 or not str(obs["etag"]).strip():
                raise PathQualificationRefused(
                    f"Qualification refused on {evidence.locus_id!r}: initial fetch failed or lacked strong ETag."
                )
            if obs["body_1_sha256"] != obs["body_2_sha256"]:
                raise PathQualificationRefused(
                    f"Negative control failed on {evidence.locus_id!r}: nondeterministic/mutating response bytes."
                )
            if obs["warm_origin_delta"] != 0 or obs["warm_post_cond_delta"] != 0:
                raise PathQualificationRefused(
                    f"Negative control failed on {evidence.locus_id!r}: warm queries hit upstream origin "
                    f"(delta={obs['warm_origin_delta']}), proving pass-through endpoint compute rather than in-path state."
                )
            if obs["conditional_status"] != 304 or (obs["size_bytes"] > 0 and obs["range_status"] != 206) or not obs["prefix_matches"]:
                raise PathQualificationRefused(
                    f"Qualification refused on {evidence.locus_id!r}: missing RFC 9110 304/206 invariants."
                )
            controls = ("no_warm_origin_passthrough", "deterministic_sha256_repeatability", "rfc9110_etag_304_and_range_206")
            params = {
                "asset_key": obs["asset_key"],
                "content_sha256": obs["body_1_sha256"],
                "etag": obs["etag"],
                "intermediary_url": obs["intermediary_url"],
                "size_bytes": obs["size_bytes"],
            }
        elif evidence.kind == "transit_affine_chain":
            # Compose affine byte hops F_pi(x) = (A * x + B) mod 256!
            comp_a, comp_b = 1, 0
            for h in obs["hops"]:
                if not h["valid_witness_mac"]:
                    raise PathQualificationRefused(
                        f"Negative control failed on hop {h['hop_id']!r}: missing or forged in-path witness MAC."
                    )
                if not h["matches_declared_law"]:
                    raise PathQualificationRefused(
                        f"Qualification refused on hop {h['hop_id']!r}: output disagreed with declared transit law."
                    )
                if not h["non_identity_transformation"]:
                    raise PathQualificationRefused(
                        f"Negative control failed on hop {h['hop_id']!r}: trivial identity pass-through cannot be credited as in-path compute."
                    )
                comp_a = (int(h["a"]) * comp_a) & 0xFF
                comp_b = (int(h["a"]) * comp_b + int(h["b"])) & 0xFF
            controls = ("authenticated_hop_witness_mac", "non_identity_in_path_transform", "exact_affine_z256_match")
            params = {"composite_a": comp_a, "composite_b": comp_b, "hop_count": len(obs["hops"])}
        else:
            raise PathQualificationRefused(f"Unsupported path Ground kind {evidence.kind!r}.")

        body = {
            "evidence_digest": evidence.evidence_digest,
            "kind": evidence.kind,
            "locus_id": evidence.locus_id,
            "negative_controls_passed": list(controls),
            "parameters": params,
        }
        return QualifiedPathCertificate(
            locus_id=evidence.locus_id,
            kind=evidence.kind,
            evidence_digest=evidence.evidence_digest,
            negative_controls_passed=controls,
            parameters=params,
            qualification_digest=domain_digest("vld.path-qualification.v1", body),
        )

    def construct_and_admit(self, cert: QualifiedPathCertificate) -> AdmittedPathHost:
        """Stages 3–4 (`construct` + `admit`): Build `VirtualGroundSpec`, bind navigator, and journal admission."""
        spec = VirtualGroundSpec(
            ground_id=f"vld.path.{cert.locus_id}",
            version="1.0.0",
            language=f"path.{cert.kind}.v1",
            state_domain="Z/256Z*",
            transitions=("navigate", "verify"),
            invariants=cert.negative_controls_passed,
            observers=("sha256_verified_slice",),
        )
        p = cert.parameters
        if cert.kind == "http_conditional_cache":
            url = f"{p['intermediary_url']}/{p['asset_key']}"

            total_sz = int(p["size_bytes"])

            def _nav_http(req: Mapping[str, int]) -> bytes:
                off = int(req["offset"])
                ln = int(req["length"])
                if off < 0 or ln < 0 or off + ln > total_sz:
                    raise PathGroundError("Requested path-cache byte range out of bounds.")
                if ln == 0:
                    return b""
                with urlopen(
                    Request(url, method="GET", headers={"Range": f"bytes={off}-{off + ln - 1}"}),
                    timeout=5.0,
                ) as resp:
                    if resp.status != 206:
                        raise PathGroundError(f"Expected 206 from admitted cache host, got {resp.status}.")
                    return resp.read()

            navigator = _nav_http
        else:
            ca = int(p["composite_a"])
            cb = int(p["composite_b"])
            law = AffineOrbitLaw(multiplier=ca, increment=cb, modulus=256)

            def _nav_transit(payload: Mapping[str, Any]) -> bytes:
                raw = bytes(payload["data"])
                repeats = int(payload.get("iterations", 1))
                if repeats < 0:
                    raise PathGroundError("Transit iterations must be non-negative.")
                # Algebraic path shortcut: K passes through the multi-hop chain collapse via AffineOrbitLaw!
                r0 = law.navigate(0, repeats)
                mul = r0.affine_multiplier
                inc = r0.affine_increment
                return bytes(((mul * b + inc) & 0xFF) for b in raw)

            navigator = _nav_transit

        host = AdmittedPathHost(locus_id=cert.locus_id, ground_spec=spec, certificate=cert, navigator=navigator)
        self._admitted[cert.locus_id] = host
        self.journal.record(
            request_id=f"admit-path:{cert.locus_id}:{cert.qualification_digest[:12]}",
            ground_fingerprint=spec.fingerprint,
            operation="path.admit_host",
            payload={"kind": cert.kind, "locus_id": cert.locus_id, "qualification_digest": cert.qualification_digest},
            result_fn=lambda: {"admitted": True, "ground_fingerprint": spec.fingerprint},
        )
        return host

    def navigate_and_verify(
        self,
        locus_id: str,
        query: Mapping[str, Any],
        oracle_fn: Callable[[], bytes],
        *,
        origin_hit_reader: Callable[[], int] | None = None,
    ) -> dict[str, Any]:
        """Stages 5–6 (`navigate` + `verify`): Execute over admitted path Ground and independently verify output."""
        host = self._admitted.get(locus_id)
        if host is None:
            raise PathGroundError(f"Path locus {locus_id!r} has not been admitted.")

        before_hits = origin_hit_reader() if origin_hit_reader is not None else 0
        actual_bytes = host.navigator(query)
        after_hits = origin_hit_reader() if origin_hit_reader is not None else 0
        expected_bytes = oracle_fn()

        if actual_bytes != expected_bytes:
            raise PathGroundError("Stage 6 verification failed: navigated path output disagreed with oracle.")
        if origin_hit_reader is not None and after_hits != before_hits:
            raise PathGroundError("Stage 6 verification failed: admitted cache unexpectedly leaked traffic to origin.")

        return {
            "bytes_length": len(actual_bytes),
            "ground_fingerprint": host.ground_spec.fingerprint,
            "locus_id": locus_id,
            "origin_hits_delta": after_hits - before_hits,
            "output_sha256": hashlib.sha256(actual_bytes).hexdigest(),
            "verified": True,
        }


__all__ = [
    "ActiveHostFormationPipeline",
    "AdmittedPathHost",
    "CountingHTTPOriginServer",
    "HTTPPathIntermediaryServer",
    "PathGroundError",
    "PathObservationEvidence",
    "PathQualificationRefused",
    "QualifiedPathCertificate",
]
