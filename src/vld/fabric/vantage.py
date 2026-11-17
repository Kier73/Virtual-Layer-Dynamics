r"""Internet & Information-Flow Vantage Infrastructure (`vld.fabric.vantage`).

Mathematical contract:
An **Information-Flow Vantage Point** $\mathcal{V} = (I_{\mathcal{V}}, \Lambda_{\text{wire}}, \mathcal{H}_{\text{recv}}, \mathcal{B})$
projects Virtual Layer Grounds, M1 byte ranges, Structural Memory queries, and Model Orbit checkpoints
onto any encodable byte/text carrier (HTTP/1.1 with RFC 9110 `Range` slicing, length-prefixed TCP/pipe
streams, newline-delimited JSONL streams, or URL/header-safe Base64url tokens) while enforcing:
  1. **Receiver-Owned Commit Authority:** Transport delivery never implies state mutation without
     receiver HMAC-SHA256 capability validation and `HashChainedJournal` idempotency fencing.
  2. **Bounded Range Locality:** Remote M1/Safetensors slices fetch exclusively `[start, end]` bytes
     via HTTP `206 Partial Content`, verified end-to-end by receiver/client digests.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import re
import ssl
import struct
import threading
from typing import Any, BinaryIO, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from vld.canonical import CanonicalizationError, canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.fabric.reliability import ReplayRejectedError, ReplayWindowGuard
from vld.ground import AffineOrbitLaw, VirtualGroundSpec
from vld.journal import HashChainedJournal, IdempotencyConflictError
from vld.memory import PagedByteManifest, PagedObjectStore, PagedStoreError, StructuralMemoryError, StructuralMemoryRegistry

ENVELOPE_DOMAIN = "vld.vantage-envelope.v1"
_RANGE_HEADER_RE = re.compile(r"bytes=(\d+)-(\d+)\Z")


class VantageError(ValueError):
    """Raised when an information-flow frame, capability token, or range request fails validation."""


class VantageAuthError(PermissionError):
    """Raised when a mutating or protected vantage operation fails HMAC capability authentication."""


class WireEncoding(str, Enum):
    """Supported wire encodings for carrying VLD envelopes over arbitrary information flows."""

    JSON_COMPACT = "json_compact"
    JSONL_STREAM = "jsonl_stream"
    HEADER_TOKEN_B64URL = "header_token_b64url"
    LENGTH_PREFIXED_BINARY = "length_prefixed_binary"


@dataclass(frozen=True, slots=True)
class VantageEnvelope:
    """Self-verifying, transport-neutral Virtual Layer message envelope."""

    vantage_id: str
    flow_kind: str
    sequence: int
    payload: Mapping[str, Any]
    hmac_tag: str | None = None
    envelope_digest: str = ""

    ALLOWED_FLOWS = frozenset(
        {"ground_spec", "op_request", "op_receipt", "m1_slice", "orbit_handoff", "structural_read"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.vantage_id, str) or not self.vantage_id.strip():
            raise VantageError("vantage_id must be a non-empty string.")
        if self.flow_kind not in self.ALLOWED_FLOWS:
            raise VantageError(f"Unsupported flow_kind {self.flow_kind!r}.")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 0:
            raise VantageError("sequence must be a non-negative integer.")
        norm_payload = parse_strict_json(canonical_json(dict(self.payload)))
        if not isinstance(norm_payload, dict):
            raise VantageError("Envelope payload must be a JSON object.")
        object.__setattr__(self, "payload", norm_payload)
        computed = domain_digest(ENVELOPE_DOMAIN, self.body_without_digest())
        if self.envelope_digest and self.envelope_digest != computed:
            raise VantageError("VantageEnvelope digest mismatch.")
        object.__setattr__(self, "envelope_digest", computed)

    def body_without_digest(self) -> dict[str, Any]:
        return {
            "flow_kind": self.flow_kind,
            "hmac_tag": self.hmac_tag,
            "payload": dict(self.payload),
            "sequence": self.sequence,
            "vantage_id": self.vantage_id,
        }

    def signing_bytes(self) -> bytes:
        return canonical_bytes(
            {
                "domain": ENVELOPE_DOMAIN,
                "flow_kind": self.flow_kind,
                "payload": dict(self.payload),
                "sequence": self.sequence,
                "vantage_id": self.vantage_id,
            }
        )

    def with_hmac(self, secret: bytes) -> VantageEnvelope:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise VantageError("HMAC capability secret must be at least 16 bytes.")
        tag = hmac.new(secret, self.signing_bytes(), hashlib.sha256).hexdigest()
        return VantageEnvelope(
            vantage_id=self.vantage_id,
            flow_kind=self.flow_kind,
            sequence=self.sequence,
            payload=self.payload,
            hmac_tag=tag,
        )

    def verify_hmac(self, secret: bytes) -> bool:
        if not self.hmac_tag or not isinstance(secret, bytes):
            return False
        expected = hmac.new(secret, self.signing_bytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.hmac_tag, expected)

    def to_dict(self) -> dict[str, Any]:
        body = self.body_without_digest()
        body["envelope_digest"] = self.envelope_digest
        return body

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VantageEnvelope:
        req = {"vantage_id", "flow_kind", "sequence", "payload", "hmac_tag", "envelope_digest"}
        if not isinstance(data, Mapping) or set(data.keys()) != req:
            raise VantageError("Invalid VantageEnvelope keyset.")
        return cls(
            vantage_id=data["vantage_id"],
            flow_kind=data["flow_kind"],
            sequence=data["sequence"],
            payload=data["payload"],
            hmac_tag=data["hmac_tag"],
            envelope_digest=data["envelope_digest"],
        )


class InformationFlowCodec:
    """Encode and decode `VantageEnvelope` instances across heterogeneous byte/text substrates."""

    MAX_FRAME_BYTES = 8 * 1024 * 1024

    @classmethod
    def encode(cls, envelope: VantageEnvelope, encoding: WireEncoding = WireEncoding.JSON_COMPACT) -> bytes:
        raw_json = canonical_bytes(envelope.to_dict())
        if len(raw_json) > cls.MAX_FRAME_BYTES:
            raise VantageError("Envelope exceeds MAX_FRAME_BYTES.")
        if encoding is WireEncoding.JSON_COMPACT:
            return raw_json
        if encoding is WireEncoding.JSONL_STREAM:
            return raw_json + b"\n"
        if encoding is WireEncoding.HEADER_TOKEN_B64URL:
            return base64.urlsafe_b64encode(raw_json).rstrip(b"=")
        if encoding is WireEncoding.LENGTH_PREFIXED_BINARY:
            return struct.pack(">I", len(raw_json)) + raw_json
        raise VantageError(f"Unsupported wire encoding: {encoding!r}")

    @classmethod
    def decode(cls, data: bytes | str, encoding: WireEncoding = WireEncoding.JSON_COMPACT) -> VantageEnvelope:
        raw = data.encode("ascii") if isinstance(data, str) else data
        if len(raw) > cls.MAX_FRAME_BYTES * 2:
            raise VantageError("Encoded frame exceeds safety ceiling.")
        try:
            if encoding is WireEncoding.JSON_COMPACT:
                parsed = parse_strict_json(raw)
            elif encoding is WireEncoding.JSONL_STREAM:
                stripped = raw.rstrip(b"\r\n")
                if b"\n" in stripped:
                    raise VantageError("JSONL_STREAM frame contains embedded newline.")
                parsed = parse_strict_json(stripped)
            elif encoding is WireEncoding.HEADER_TOKEN_B64URL:
                pad = b"=" * ((4 - (len(raw) % 4)) % 4)
                decoded = base64.urlsafe_b64decode(raw + pad)
                parsed = parse_strict_json(decoded)
            elif encoding is WireEncoding.LENGTH_PREFIXED_BINARY:
                if len(raw) < 4:
                    raise VantageError("LENGTH_PREFIXED_BINARY frame shorter than 4-byte header.")
                (decl_len,) = struct.unpack(">I", raw[:4])
                if decl_len != len(raw) - 4:
                    raise VantageError(f"Declared frame length {decl_len} != payload length {len(raw) - 4}.")
                parsed = parse_strict_json(raw[4:])
            else:
                raise VantageError(f"Unsupported wire encoding: {encoding!r}")
        except (CanonicalizationError, ValueError) as exc:
            if isinstance(exc, VantageError):
                raise
            raise VantageError(f"Failed to decode {encoding.value} frame: {exc}") from exc

        if not isinstance(parsed, dict):
            raise VantageError("Decoded envelope root must be a JSON object.")
        return VantageEnvelope.from_dict(parsed)

    @classmethod
    def read_length_prefixed(cls, stream: BinaryIO) -> VantageEnvelope:
        hdr = stream.read(4)
        if len(hdr) != 4:
            raise VantageError("Truncated length-prefixed stream header.")
        (length,) = struct.unpack(">I", hdr)
        if length > cls.MAX_FRAME_BYTES:
            raise VantageError(f"Stream frame length {length} exceeds MAX_FRAME_BYTES.")
        body = stream.read(length)
        if len(body) != length:
            raise VantageError("Truncated length-prefixed stream payload.")
        return cls.decode(hdr + body, WireEncoding.LENGTH_PREFIXED_BINARY)


class VLDVantageNode:
    """Receiver-authoritative Virtual Layer host kernel attachable to HTTP, TCP, or pipe vantages."""

    def __init__(
        self,
        node_id: str,
        *,
        store: PagedObjectStore | None = None,
        structural_memory: StructuralMemoryRegistry | None = None,
        journal: HashChainedJournal | None = None,
        capability_secret: bytes | None = None,
        replay_guard: ReplayWindowGuard | None = None,
    ) -> None:
        if not isinstance(node_id, str) or not node_id.strip():
            raise VantageError("node_id must be a non-empty string.")
        if capability_secret is not None and (not isinstance(capability_secret, bytes) or len(capability_secret) < 16):
            raise VantageError("capability_secret must be bytes of length >= 16 when supplied.")
        self.node_id = node_id
        self.journal = journal if journal is not None else HashChainedJournal()
        self.store = store if store is not None else PagedObjectStore()
        self.structural_memory = (
            structural_memory if structural_memory is not None else StructuralMemoryRegistry(self.journal)
        )
        self._capability_secret = capability_secret
        self._replay_guard = replay_guard
        self._grounds: dict[str, VirtualGroundSpec] = {}
        self._seq_lock = threading.Lock()
        self._out_seq = 0

    def _next_seq(self) -> int:
        with self._seq_lock:
            val = self._out_seq
            self._out_seq += 1
            return val

    def register_ground(self, spec: VirtualGroundSpec) -> str:
        with self._seq_lock:
            self._grounds[spec.ground_id] = spec
        return spec.fingerprint

    def health_summary(self) -> dict[str, Any]:
        with self._seq_lock:
            hosted = sorted(self._grounds.keys())
        return {
            "capability_protected": self._capability_secret is not None,
            "hosted_grounds": hosted,
            "journal_events": len(self.journal),
            "journal_head_digest": self.journal.head_digest,
            "node_id": self.node_id,
            "status": "READY",
        }

    def handle_envelope(self, request_env: VantageEnvelope) -> VantageEnvelope:
        """Validate capability & idempotency, execute admitted operation, and emit receipt envelope."""
        if request_env.flow_kind != "op_request":
            raise VantageError(f"Expected flow_kind='op_request', got {request_env.flow_kind!r}.")
        if self._capability_secret is not None and not request_env.verify_hmac(self._capability_secret):
            raise VantageAuthError("Missing or invalid HMAC capability token for receiver node.")
        if self._replay_guard is not None:
            with self._seq_lock:
                try:
                    self._replay_guard.check_and_record(request_env.vantage_id, request_env.sequence)
                except ReplayRejectedError as exc:
                    raise VantageAuthError(str(exc)) from exc

        p = request_env.payload
        req_id = p.get("request_id")
        op = p.get("operation")
        args = p.get("args", {})
        if not isinstance(req_id, str) or not req_id or not isinstance(op, str) or not isinstance(args, dict):
            raise VantageError("op_request payload requires {'request_id', 'operation', 'args'}.")

        ground_fp = str(p.get("ground_fingerprint", f"vantage:{self.node_id}:{op}"))

        def _execute_op() -> dict[str, Any]:
            if op == "affine_navigate":
                law = AffineOrbitLaw(
                    multiplier=int(args["multiplier"]),
                    increment=int(args["increment"]),
                    modulus=int(args["modulus"]),
                )
                jump = law.navigate(int(args["initial_state"]), int(args["coordinate"]))
                return {
                    "binary_steps": jump.binary_steps,
                    "coordinate": jump.coordinate,
                    "law_digest": jump.law_digest,
                    "value": jump.value,
                }
            if op == "admit_structural_law":
                spec, cert = self.structural_memory.certify_and_admit(
                    str(args["memory_id"]),
                    [int(v) for v in args["samples"]],
                    domain=str(args.get("domain", "Z")),
                    declared_modulus=args.get("declared_modulus"),
                    allow_unbounded_poly=bool(args.get("allow_unbounded_poly", False)),
                )
                return {
                    "certificate_digest": cert.digest,
                    "family": spec.law.family,
                    "revision": spec.revision,
                    "spec_digest": spec.digest,
                }
            if op == "structural_read":
                vals, read_rec = self.structural_memory.read_range(
                    str(args["memory_id"]),
                    int(args["coordinate_start"]),
                    int(args.get("length", 1)),
                )
                return {
                    "certificate_digest": read_rec.certificate_digest,
                    "expanded_intermediate_states": read_rec.expanded_intermediate_states,
                    "receipt_digest": read_rec.receipt_digest,
                    "values": list(vals),
                }
            if op == "m1_put_hex":
                raw = bytes.fromhex(str(args["hex_bytes"]))
                manifest = self.store.put_bytes(
                    raw,
                    media_type=str(args.get("media_type", "application/octet-stream")),
                )
                return {"manifest": manifest.to_dict(), "manifest_digest": manifest.manifest_digest}
            raise VantageError(f"Unsupported vantage operation: {op!r}.")

        journal_receipt, replayed = self.journal.record(
            request_id=req_id,
            ground_fingerprint=ground_fp,
            operation=f"vantage.{op}",
            payload=args,
            result_fn=_execute_op,
        )
        resp = VantageEnvelope(
            vantage_id=self.node_id,
            flow_kind="op_receipt",
            sequence=self._next_seq(),
            payload={
                "journal_event_digest": journal_receipt.event_digest,
                "journal_seq": journal_receipt.seq,
                "replayed": replayed,
                "request_id": req_id,
                "result": journal_receipt.result,
            },
        )
        if self._capability_secret is not None:
            resp = resp.with_hmac(self._capability_secret)
        return resp


class StreamPipeVantage:
    """Duplex stream adapter hosting `VLDVantageNode` over any `BinaryIO` transport (pipes, stdio, sockets)."""

    def __init__(self, node: VLDVantageNode) -> None:
        self.node = node

    def process_one_frame(self, inp: BinaryIO, out: BinaryIO) -> VantageEnvelope:
        req = InformationFlowCodec.read_length_prefixed(inp)
        resp = self.node.handle_envelope(req)
        out.write(InformationFlowCodec.encode(resp, WireEncoding.LENGTH_PREFIXED_BINARY))
        out.flush()
        return resp


class VLDHttpVantageServer:
    """Multi-threaded HTTP/1.1 Vantage Server supporting capability-gated ops and RFC 9110 `Range` byte reads."""

    MAX_POST_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        node: VLDVantageNode,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.node = node
        self._ssl_context = ssl_context
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        if ssl_context is not None:
            self._httpd.socket = ssl_context.wrap_socket(self._httpd.socket, server_side=True)
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        addr, port = self._httpd.server_address[:2]
        scheme = "https" if self._ssl_context is not None else "http"
        return f"{scheme}://{addr}:{port}"

    def start_background(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                kwargs={"poll_interval": 0.02},
                name=f"vld-http-vantage-{self.node.node_id}",
                daemon=True,
            )
            self._thread.start()
        return self.base_url

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> VLDHttpVantageServer:
        self.start_background()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                # Silence noisy stderr HTTP logs during programmatic/test runs.
                return

            def _send_json(self, status: int, obj: Mapping[str, Any]) -> None:
                body = canonical_bytes(dict(obj))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/v1/health":
                    self._send_json(200, parent.node.health_summary())
                    return
                if path == "/v1/grounds":
                    self._send_json(
                        200,
                        {"grounds": {gid: g.to_dict() for gid, g in sorted(parent.node._grounds.items())}},
                    )
                    return
                if path.startswith("/v1/grounds/"):
                    gid = path[len("/v1/grounds/") :]
                    spec = parent.node._grounds.get(gid)
                    if spec is None:
                        self._send_json(404, {"error": "ground_not_found"})
                    else:
                        self._send_json(200, spec.to_dict())
                    return
                if path.startswith("/v1/m1/") and path.endswith("/manifest"):
                    mdigest = path[len("/v1/m1/") : -len("/manifest")]
                    try:
                        m = parent.node.store.get_manifest(mdigest)
                        self._send_json(200, m.to_dict())
                    except PagedStoreError as exc:
                        self._send_json(404, {"error": str(exc)})
                    return
                if path.startswith("/v1/m1/") and path.endswith("/bytes"):
                    mdigest = path[len("/v1/m1/") : -len("/bytes")]
                    try:
                        m = parent.node.store.get_manifest(mdigest)
                    except PagedStoreError as exc:
                        self._send_json(404, {"error": str(exc)})
                        return

                    range_hdr = self.headers.get("Range")
                    if range_hdr is None:
                        raw = parent.node.store.read_bytes(m)
                        self.send_response(200)
                        self.send_header("Content-Type", m.media_type)
                        self.send_header("Content-Length", str(len(raw)))
                        self.send_header("X-VLD-Slice-SHA256", hashlib.sha256(raw).hexdigest())
                        self.end_headers()
                        self.wfile.write(raw)
                        return

                    match = _RANGE_HEADER_RE.fullmatch(range_hdr.strip())
                    if match is None:
                        self._send_json(416, {"error": "invalid_range_header"})
                        return
                    start, end_incl = int(match.group(1)), int(match.group(2))
                    if start > end_incl or end_incl >= m.size_bytes:
                        self._send_json(416, {"error": "range_not_satisfiable", "size_bytes": m.size_bytes})
                        return
                    length = end_incl - start + 1
                    raw_slice = parent.node.store.read_range(m, start, length)
                    self.send_response(206)
                    self.send_header("Content-Type", m.media_type)
                    self.send_header("Content-Range", f"bytes {start}-{end_incl}/{m.size_bytes}")
                    self.send_header("Content-Length", str(len(raw_slice)))
                    self.send_header("X-VLD-Slice-SHA256", hashlib.sha256(raw_slice).hexdigest())
                    self.end_headers()
                    self.wfile.write(raw_slice)
                    return

                self._send_json(404, {"error": "unknown_route"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/v1/ops":
                    self._send_json(404, {"error": "unknown_route"})
                    return
                try:
                    content_len = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._send_json(400, {"error": "invalid_content_length"})
                    return
                if content_len <= 0 or content_len > parent.MAX_POST_BYTES:
                    self._send_json(413, {"error": "payload_size_out_of_bounds"})
                    return
                raw = self.rfile.read(content_len)
                try:
                    env = InformationFlowCodec.decode(raw, WireEncoding.JSON_COMPACT)
                    resp_env = parent.node.handle_envelope(env)
                except VantageAuthError as exc:
                    self._send_json(403, {"error": str(exc)})
                    return
                except IdempotencyConflictError as exc:
                    self._send_json(409, {"error": str(exc)})
                    return
                except (VantageError, ValueError, KeyError, StructuralMemoryError) as exc:
                    self._send_json(400, {"error": str(exc)})
                    return

                out_bytes = InformationFlowCodec.encode(resp_env, WireEncoding.JSON_COMPACT)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out_bytes)))
                self.end_headers()
                self.wfile.write(out_bytes)

        return _Handler


class _StrictNoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        raise VantageError(f"HTTP redirect ({code}) refused by receiver vantage policy.")


class VLDHttpVantageClient:
    """Strict, redirect-refusing HTTP Vantage Client with Range-slice SHA-256 verification."""

    def __init__(
        self,
        base_url: str,
        *,
        client_id: str = "vld.client",
        capability_secret: bytes | None = None,
        timeout_s: float = 10.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self._capability_secret = capability_secret
        self.timeout_s = timeout_s
        handlers: list[Any] = [_StrictNoRedirect]
        if ssl_context is not None:
            handlers.append(HTTPSHandler(context=ssl_context))
        self._opener = build_opener(*handlers)
        self._seq_lock = threading.Lock()
        self._seq = 0

    def health(self) -> dict[str, Any]:
        req = Request(f"{self.base_url}/v1/health", method="GET")
        with self._opener.open(req, timeout=self.timeout_s) as resp:
            return parse_strict_json(resp.read(1_048_576))

    def fetch_manifest(self, manifest_digest: str) -> PagedByteManifest:
        req = Request(f"{self.base_url}/v1/m1/{manifest_digest}/manifest", method="GET")
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                data = parse_strict_json(resp.read(4_194_304))
        except HTTPError as exc:
            raise VantageError(f"Manifest fetch failed ({exc.code}).") from exc
        manifest = PagedByteManifest.from_dict(data)
        if manifest.manifest_digest != manifest_digest:
            raise VantageError("Remote manifest digest mismatch.")
        return manifest

    def fetch_m1_range(self, manifest_digest: str, offset: int, length: int) -> bytes:
        """Fetch exact `[offset, offset + length)` slice via HTTP `Range` (`206 Partial Content`) and verify SHA-256."""
        if offset < 0 or length < 0:
            raise VantageError("offset and length must be non-negative.")
        if length == 0:
            return b""
        end_incl = offset + length - 1
        req = Request(
            f"{self.base_url}/v1/m1/{manifest_digest}/bytes",
            method="GET",
            headers={"Range": f"bytes={offset}-{end_incl}"},
        )
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                if resp.status != 206:
                    raise VantageError(f"Expected HTTP 206 Partial Content, got {resp.status}.")
                raw = resp.read(length + 1)
                reported_sha = resp.headers.get("X-VLD-Slice-SHA256", "")
        except (HTTPError, URLError) as exc:
            raise VantageError(f"M1 Range read failed: {exc}") from exc
        if len(raw) != length:
            raise VantageError(f"Expected {length} bytes in Range response, got {len(raw)}.")
        if reported_sha and hashlib.sha256(raw).hexdigest() != reported_sha:
            raise VantageError("M1 Range slice SHA-256 mismatch.")
        return raw

    def invoke_op(
        self,
        operation: str,
        args: Mapping[str, Any],
        *,
        request_id: str,
        ground_fingerprint: str | None = None,
    ) -> VantageEnvelope:
        payload: dict[str, Any] = {
            "args": dict(args),
            "operation": operation,
            "request_id": request_id,
        }
        if ground_fingerprint is not None:
            payload["ground_fingerprint"] = ground_fingerprint
        with self._seq_lock:
            seq = self._seq
            self._seq += 1
        env = VantageEnvelope(
            vantage_id=self.client_id,
            flow_kind="op_request",
            sequence=seq,
            payload=payload,
        )
        if self._capability_secret is not None:
            env = env.with_hmac(self._capability_secret)

        body = InformationFlowCodec.encode(env, WireEncoding.JSON_COMPACT)
        req = Request(
            f"{self.base_url}/v1/ops",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                raw_resp = resp.read(InformationFlowCodec.MAX_FRAME_BYTES)
        except HTTPError as exc:
            err_body = exc.read(4096).decode("utf-8", errors="replace")
            if exc.code == 403:
                raise VantageAuthError(f"Remote receiver denied capability ({exc.code}): {err_body}") from exc
            if exc.code == 409:
                raise IdempotencyConflictError(f"Remote receiver idempotency conflict ({exc.code}): {err_body}") from exc
            raise VantageError(f"Remote operation failed ({exc.code}): {err_body}") from exc

        resp_env = InformationFlowCodec.decode(raw_resp, WireEncoding.JSON_COMPACT)
        if self._capability_secret is not None and not resp_env.verify_hmac(self._capability_secret):
            raise VantageAuthError("Remote response failed HMAC capability verification.")
        return resp_env


__all__ = [
    "InformationFlowCodec",
    "StreamPipeVantage",
    "VLDHttpVantageClient",
    "VLDHttpVantageServer",
    "VLDVantageNode",
    "VantageAuthError",
    "VantageEnvelope",
    "VantageError",
    "WireEncoding",
]
