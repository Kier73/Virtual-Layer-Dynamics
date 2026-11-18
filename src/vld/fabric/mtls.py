r"""Phase 4 — Zero-Dependency X.509 v3 PKI, Mutual-TLS Peer Channels & Range-Resumable M1 Streaming (`vld.fabric.mtls`).

Mathematical & security contract:
1. **Mutual TLS Channel Invariant:**
   Both client and server enforce `minimum_version >= TLSv1.2`, `verify_mode == ssl.CERT_REQUIRED`,
   non-null negotiated cipher, and explicit peer X.509 `subjectAltName` (`DNS:<peer_id>`) verification.
2. **Control-Plane vs. Bulk Data-Plane Separation:**
   Control frames are strictly bounded to `MAX_CONTROL_FRAME_BYTES = 65_536` bytes (64 KiB).
   Multi-megabyte M1 objects stream out-of-band as binary `[u32 length][raw_bytes][32B sha256]` chunks
   and resume from the exact interrupted page/byte offset after link drops without re-sending verified prefixes.
3. **Self-Contained Pure-Python RSA-2048 / X.509 v3 PKI Provisioner (`generate_ephemeral_mtls_pki`):**
   Synthesizes standards-compliant DER/PEM Root CA and SAN-bound leaf certificates using pure Python
   number theory (`pow(a, d, n)` Miller-Rabin + ASN.1 DER + PKCS#1 v1.5 SHA-256) with zero external CLI
   or third-party package dependencies.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import math
import os
from pathlib import Path
import secrets
import socket
import ssl
import struct
import threading
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import canonical_bytes, parse_strict_json
from vld.memory import PageChunkRef, PagedByteManifest, PagedObjectStore, PagedStoreError

MAX_CONTROL_FRAME_BYTES = 65_536
DEFAULT_STREAM_CHUNK_BYTES = 65_536
MAX_MANIFEST_STREAM_BYTES = 16 * 1024 * 1024

_SMALL_PRIMES = (
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
    157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
    239, 241, 251,
)
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


class MutualTLSError(RuntimeError):
    """Raised when TLS configuration, X.509 chain/SAN identity, or stream framing fails."""


# ============================================================================
# 1. Pure-Python ASN.1 DER + RSA-2048 X.509 v3 Ephemeral PKI Builder
# ============================================================================


def _is_probable_prime(n: int) -> bool:
    if n < 2 or (n & 1) == 0:
        return n == 2
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    while True:
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if _is_probable_prime(cand):
            return cand


def _generate_rsa_keypair(bits: int = 2048) -> tuple[int, int, int, int, int]:
    e = 65537
    half = bits // 2
    while True:
        p = _generate_prime(half)
        q = _generate_prime(half)
        if p == q or abs(p - q) < (1 << (half - 100)):
            continue
        phi = (p - 1) * (q - 1)
        if math.gcd(e, phi) == 1:
            n = p * q
            if n.bit_length() == bits:
                d = pow(e, -1, phi)
                return (n, e, d, p, q)


def _der_len(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    enc = length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(enc)]) + enc


def _der_tlv(tag: int, content: bytes) -> bytes:
    return bytes([tag]) + _der_len(len(content)) + content


def _der_int(val: int) -> bytes:
    if val == 0:
        return _der_tlv(0x02, b"\x00")
    blen = (val.bit_length() + 7) // 8
    raw = val.to_bytes(blen, "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return _der_tlv(0x02, raw)


def _der_oid(oid: str) -> bytes:
    parts = [int(x) for x in oid.split(".")]
    out = bytearray([40 * parts[0] + parts[1]])
    for arc in parts[2:]:
        if arc == 0:
            out.append(0)
        else:
            buf: list[int] = []
            v = arc
            while v > 0:
                buf.append(v & 0x7F)
                v >>= 7
            for idx in range(len(buf) - 1, -1, -1):
                out.append(buf[idx] | (0x80 if idx > 0 else 0))
    return _der_tlv(0x06, bytes(out))


def _der_seq(*items: bytes) -> bytes:
    return _der_tlv(0x30, b"".join(items))


def _der_set(*items: bytes) -> bytes:
    return _der_tlv(0x31, b"".join(items))


def _der_octet(raw: bytes) -> bytes:
    return _der_tlv(0x04, raw)


def _der_bitstring(raw: bytes, unused_bits: int = 0) -> bytes:
    return _der_tlv(0x03, bytes([unused_bits]) + raw)


def _der_utf8(text: str) -> bytes:
    return _der_tlv(0x0C, text.encode("utf-8"))


def _der_utctime(dt: datetime) -> bytes:
    utc = dt.astimezone(timezone.utc)
    return _der_tlv(0x17, utc.strftime("%y%m%d%H%M%SZ").encode("ascii"))


def _der_name_cn(cn: str) -> bytes:
    attr = _der_seq(_der_oid("2.5.4.3"), _der_utf8(cn))
    return _der_seq(_der_set(attr))


def _der_ext(oid: str, value_der: bytes, *, critical: bool = False) -> bytes:
    if critical:
        return _der_seq(_der_oid(oid), _der_tlv(0x01, b"\xff"), _der_octet(value_der))
    return _der_seq(_der_oid(oid), _der_octet(value_der))


def _pem_encode(label: str, der_bytes: bytes) -> str:
    b64 = base64.b64encode(der_bytes).decode("ascii")
    lines = [b64[i : i + 64] for i in range(0, len(b64), 64)]
    return f"-----BEGIN {label}-----\n" + "\n".join(lines) + f"\n-----END {label}-----\n"


def _encode_rsa_private_key_pem(key: tuple[int, int, int, int, int]) -> str:
    n, e, d, p, q = key
    dp = d % (p - 1)
    dq = d % (q - 1)
    qinv = pow(q, -1, p)
    der = _der_seq(
        _der_int(0),
        _der_int(n),
        _der_int(e),
        _der_int(d),
        _der_int(p),
        _der_int(q),
        _der_int(dp),
        _der_int(dq),
        _der_int(qinv),
    )
    return _pem_encode("RSA PRIVATE KEY", der)


def _sign_pkcs1_v15_sha256(tbs_der: bytes, ca_key: tuple[int, int, int, int, int]) -> bytes:
    n, _, d, _, _ = ca_key
    k_bytes = (n.bit_length() + 7) // 8
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + hashlib.sha256(tbs_der).digest()
    pad_len = k_bytes - 3 - len(digest_info)
    em = b"\x00\x01" + (b"\xff" * pad_len) + b"\x00" + digest_info
    sig_int = pow(int.from_bytes(em, "big"), d, n)
    return sig_int.to_bytes(k_bytes, "big")


def _build_x509_v3_cert_pem(
    *,
    subject_cn: str,
    issuer_cn: str,
    subject_pub: tuple[int, int],
    signer_key: tuple[int, int, int, int, int],
    is_ca: bool,
    san_dns: Sequence[str] = (),
    issuer_ski: bytes | None = None,
) -> tuple[str, bytes]:
    pub_pkcs1 = _der_seq(_der_int(subject_pub[0]), _der_int(subject_pub[1]))
    ski = hashlib.sha1(pub_pkcs1).digest()
    spki = _der_seq(
        _der_seq(_der_oid("1.2.840.113549.1.1.1"), _der_tlv(0x05, b"")),
        _der_bitstring(pub_pkcs1, 0),
    )
    sig_alg = _der_seq(_der_oid("1.2.840.113549.1.1.11"), _der_tlv(0x05, b""))
    now = datetime.now(timezone.utc)
    validity = _der_seq(_der_utctime(now - timedelta(hours=1)), _der_utctime(now + timedelta(days=30)))
    serial = secrets.randbits(120) | 1

    exts: list[bytes] = []
    if is_ca:
        exts.append(_der_ext("2.5.29.19", _der_seq(_der_tlv(0x01, b"\xff")), critical=True))
        # keyCertSign (bit 5) | cRLSign (bit 6) -> 0b00000110, 1 unused bit
        exts.append(_der_ext("2.5.29.15", _der_bitstring(b"\x06", 1), critical=True))
        exts.append(_der_ext("2.5.29.14", _der_octet(ski)))
    else:
        exts.append(_der_ext("2.5.29.19", _der_seq(), critical=True))
        # digitalSignature (bit 0) | keyEncipherment (bit 2) -> 0b10100000 = 0xA0, 5 unused bits
        exts.append(_der_ext("2.5.29.15", _der_bitstring(b"\xa0", 5), critical=True))
        # EKU: serverAuth + clientAuth
        eku = _der_seq(_der_oid("1.3.6.1.5.5.7.3.1"), _der_oid("1.3.6.1.5.5.7.3.2"))
        exts.append(_der_ext("2.5.29.37", eku))
        san_items: list[bytes] = []
        for entry in san_dns:
            san_items.append(_der_tlv(0x82, entry.encode("ascii")))
            try:
                ip_packed = ipaddress.ip_address(entry).packed
                san_items.append(_der_tlv(0x87, ip_packed))
            except ValueError:
                pass
        exts.append(_der_ext("2.5.29.17", _der_seq(*san_items)))
        exts.append(_der_ext("2.5.29.14", _der_octet(ski)))
        if issuer_ski is not None:
            exts.append(_der_ext("2.5.29.35", _der_seq(_der_tlv(0x80, issuer_ski))))

    tbs = _der_seq(
        _der_tlv(0xA0, _der_int(2)),  # version v3
        _der_int(serial),
        sig_alg,
        _der_name_cn(issuer_cn),
        validity,
        _der_name_cn(subject_cn),
        spki,
        _der_tlv(0xA3, _der_seq(*exts)),
    )
    signature = _sign_pkcs1_v15_sha256(tbs, signer_key)
    cert_der = _der_seq(tbs, sig_alg, _der_bitstring(signature, 0))
    return _pem_encode("CERTIFICATE", cert_der), ski


@dataclass(frozen=True, slots=True)
class PeerIdentityBundle:
    """Materialized CA + leaf X.509 v3 certificate & RSA-2048 private key files for mTLS."""

    peer_dns: str
    ca_cert_path: Path
    cert_path: Path
    key_path: Path


def _write_private_key_secure(path: Path, pem_text: str) -> None:
    """Write PEM private key material with owner-only `0600` mode from file creation (zero `0644` TOCTOU window)."""
    raw = pem_text.encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(fd, 0o600)  # type: ignore[attr-defined]
        except OSError:
            pass
        written = 0
        while written < len(raw):
            n = os.write(fd, raw[written:])
            if n <= 0:
                raise MutualTLSError(f"Short write persisting private key {path}.")
            written += n
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def generate_ephemeral_mtls_pki(
    out_dir: str | Path,
    peer_dns_names: Sequence[str],
    *,
    ca_common_name: str = "VLD Ephemeral Root CA",
) -> dict[str, PeerIdentityBundle]:
    """Generate a fresh Root CA and SAN-bound mTLS leaf certificates on disk with zero external tools."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(root, 0o700)
    except OSError:
        pass
    ca_key = _generate_rsa_keypair(2048)
    ca_pem, ca_ski = _build_x509_v3_cert_pem(
        subject_cn=ca_common_name,
        issuer_cn=ca_common_name,
        subject_pub=(ca_key[0], ca_key[1]),
        signer_key=ca_key,
        is_ca=True,
    )
    ca_path = root / "ca.pem"
    ca_path.write_text(ca_pem, encoding="ascii")

    bundles: dict[str, PeerIdentityBundle] = {}
    for dns in peer_dns_names:
        leaf_key = _generate_rsa_keypair(2048)
        leaf_pem, _ = _build_x509_v3_cert_pem(
            subject_cn=dns,
            issuer_cn=ca_common_name,
            subject_pub=(leaf_key[0], leaf_key[1]),
            signer_key=ca_key,
            is_ca=False,
            san_dns=(dns,),
            issuer_ski=ca_ski,
        )
        safe = dns.replace(":", "_").replace("/", "_")
        cert_p = root / f"{safe}.crt.pem"
        key_p = root / f"{safe}.key.pem"
        cert_p.write_text(leaf_pem, encoding="ascii")
        _write_private_key_secure(key_p, _encode_rsa_private_key_pem(leaf_key))
        bundles[dns] = PeerIdentityBundle(peer_dns=dns, ca_cert_path=ca_path, cert_path=cert_p, key_path=key_p)
    return bundles


# ============================================================================
# 2. Strict Mutual-TLS Context Validation & Peer SAN Enforcement
# ============================================================================


def create_mtls_server_context(bundle: PeerIdentityBundle) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = False
    ctx.load_verify_locations(cafile=str(bundle.ca_cert_path))
    ctx.load_cert_chain(certfile=str(bundle.cert_path), keyfile=str(bundle.key_path))
    return ctx


def create_mtls_client_context(bundle: PeerIdentityBundle) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    ctx.load_verify_locations(cafile=str(bundle.ca_cert_path))
    ctx.load_cert_chain(certfile=str(bundle.cert_path), keyfile=str(bundle.key_path))
    return ctx


def _extract_peer_dns_names(tls_sock: ssl.SSLSocket) -> frozenset[str]:
    if tls_sock.version() not in ("TLSv1.2", "TLSv1.3") or tls_sock.cipher() is None:
        raise MutualTLSError("Negotiated TLS protocol or cipher violates security floor.")
    cert = tls_sock.getpeercert()
    if not cert:
        raise MutualTLSError("Peer did not present a verified X.509 certificate.")
    return frozenset(val for kind, val in cert.get("subjectAltName", ()) if kind == "DNS")


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    try:
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise MutualTLSError(f"Unexpected EOF reading {n} bytes (got {len(buf)}).")
            buf.extend(chunk)
    except OSError as exc:
        raise MutualTLSError(f"mTLS socket receive failed: {exc}") from exc
    return bytes(buf)


def _send_control_frame(sock: socket.socket, payload: Mapping[str, Any]) -> None:
    raw = canonical_bytes(dict(payload))
    if len(raw) > MAX_CONTROL_FRAME_BYTES:
        raise MutualTLSError(
            f"Control frame size {len(raw)} exceeds MAX_CONTROL_FRAME_BYTES ({MAX_CONTROL_FRAME_BYTES}); "
            "bulk data must use out-of-band M1 range streaming."
        )
    try:
        sock.sendall(struct.pack(">I", len(raw)) + raw)
    except OSError as exc:
        raise MutualTLSError(f"mTLS socket send failed: {exc}") from exc


def _recv_control_frame(sock: socket.socket) -> dict[str, Any]:
    (length,) = struct.unpack(">I", _recv_exact(sock, 4))
    if length == 0 or length > MAX_CONTROL_FRAME_BYTES:
        raise MutualTLSError(f"Inbound control frame length {length} violates 64 KiB control ceiling.")
    parsed = parse_strict_json(_recv_exact(sock, length))
    if not isinstance(parsed, dict):
        raise MutualTLSError("Control frame root must be a JSON object.")
    return parsed


# ============================================================================
# 3. Mutual-TLS Peer Server & Range-Resumable M1 Streaming Client
# ============================================================================


class MTLSPeerServer:
    """Mutually authenticated TLS peer listener serving bounded control ops and range-resumable M1 streams."""

    def __init__(
        self,
        *,
        peer_id: str,
        ssl_context: ssl.SSLContext,
        allowed_peer_dns: frozenset[str],
        store: PagedObjectStore | None = None,
        control_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        drop_stream_after_bytes: int | None = None,
    ) -> None:
        if ssl_context.verify_mode is not ssl.CERT_REQUIRED or ssl_context.minimum_version < ssl.TLSVersion.TLSv1_2:
            raise MutualTLSError("MTLSPeerServer requires TLS >= 1.2 and CERT_REQUIRED.")
        self.peer_id = peer_id
        self._ssl_ctx = ssl_context
        self.allowed_peer_dns = frozenset(allowed_peer_dns)
        self.store = store if store is not None else PagedObjectStore()
        self._control_handler = control_handler
        self._drop_after = drop_stream_after_bytes  # One-shot fault injection for testing resume!
        self._drop_lock = threading.Lock()

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.control_bytes_received = 0
        self.bulk_bytes_streamed = 0

    @property
    def endpoint(self) -> tuple[str, int]:
        addr, port = self._listener.getsockname()[:2]
        return str(addr), int(port)

    def start_background(self) -> tuple[str, int]:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._accept_loop,
                name=f"vld-mtls-{self.peer_id}",
                daemon=True,
            )
            self._thread.start()
        return self.endpoint

    def close(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> MTLSPeerServer:
        self.start_background()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw_conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(raw_conn,), daemon=True).start()

    def _handle_conn(self, raw_conn: socket.socket) -> None:
        tls_conn: ssl.SSLSocket | None = None
        try:
            raw_conn.settimeout(5.0)
            tls_conn = self._ssl_ctx.wrap_socket(raw_conn, server_side=True, do_handshake_on_connect=True)
            peer_names = _extract_peer_dns_names(tls_conn)
            matched = peer_names & self.allowed_peer_dns
            if not matched:
                # Drain inbound request frame before replying + half-closing so Windows Winsock never emits TCP RST!
                try:
                    _recv_control_frame(tls_conn)
                except MutualTLSError:
                    pass
                _send_control_frame(tls_conn, {"error": "peer_san_unauthorized", "status": "DENIED"})
                try:
                    tls_conn.shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                return
            caller_dns = sorted(matched)[0]

            while not self._stop.is_set():
                try:
                    req = _recv_control_frame(tls_conn)
                except MutualTLSError:
                    break
                self.control_bytes_received += len(canonical_bytes(req))
                kind = req.get("kind")

                if kind == "ping":
                    _send_control_frame(tls_conn, {"authenticated_as": caller_dns, "peer_id": self.peer_id, "status": "OK"})
                elif kind == "m1_manifest":
                    mdigest = str(req.get("manifest_digest", ""))
                    try:
                        m = self.store.get_manifest(mdigest)
                        raw_m = m.to_bytes()
                        _send_control_frame(
                            tls_conn,
                            {
                                "manifest_bytes": len(raw_m),
                                "manifest_sha256": hashlib.sha256(raw_m).hexdigest(),
                                "status": "MANIFEST_STREAM",
                            },
                        )
                        tls_conn.sendall(raw_m)
                    except PagedStoreError as exc:
                        _send_control_frame(tls_conn, {"error": str(exc), "status": "NOT_FOUND"})
                elif kind == "m1_range_stream":
                    mdigest = str(req.get("manifest_digest", ""))
                    offset = int(req.get("offset", 0))
                    length = int(req.get("length", 0))
                    chunk_sz = min(DEFAULT_STREAM_CHUNK_BYTES, max(1024, int(req.get("chunk_size", DEFAULT_STREAM_CHUNK_BYTES))))
                    try:
                        m = self.store.get_manifest(mdigest)
                        if offset < 0 or length < 0 or offset + length > m.size_bytes:
                            raise PagedStoreError("Requested range outside M1 object bounds.")
                    except PagedStoreError as exc:
                        _send_control_frame(tls_conn, {"error": str(exc), "status": "ERROR"})
                        continue

                    _send_control_frame(tls_conn, {"length": length, "offset": offset, "status": "STREAM_START"})
                    sent = 0
                    with self._drop_lock:
                        drop_budget = self._drop_after
                        self._drop_after = None  # Trigger once so subsequent resume succeeds!
                    while sent < length:
                        step = min(chunk_sz, length - sent)
                        raw_slice = self.store.read_range(m, offset + sent, step)
                        if drop_budget is not None and sent + step > drop_budget:
                            # Simulate sudden TCP/TLS link drop mid-stream!
                            tls_conn.close()
                            return
                        digest32 = hashlib.sha256(raw_slice).digest()
                        tls_conn.sendall(struct.pack(">I", len(raw_slice)) + raw_slice + digest32)
                        sent += step
                        self.bulk_bytes_streamed += len(raw_slice)
                    tls_conn.sendall(struct.pack(">I", 0))  # Clean EOF sentinel
                elif kind == "control_op":
                    if self._control_handler is None:
                        _send_control_frame(tls_conn, {"error": "no_control_handler", "status": "ERROR"})
                    else:
                        try:
                            out = self._control_handler(caller_dns, dict(req.get("body", {})))
                            _send_control_frame(tls_conn, {"result": out, "status": "OK"})
                        except PermissionError as exc:
                            _send_control_frame(tls_conn, {"error": str(exc), "status": "DENIED"})
                        except Exception as exc:
                            _send_control_frame(tls_conn, {"error": str(exc), "status": "ERROR"})
                else:
                    _send_control_frame(tls_conn, {"error": f"unknown_kind:{kind}", "status": "ERROR"})
        except Exception:
            pass
        finally:
            if tls_conn is not None:
                try:
                    tls_conn.close()
                except OSError:
                    pass
            else:
                try:
                    raw_conn.close()
                except OSError:
                    pass


class MTLSPeerClient:
    """Mutually authenticated TLS client supporting 64 KiB control RPCs and range-resumable M1 transfers."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        expected_server_dns: str,
        ssl_context: ssl.SSLContext,
        timeout_s: float = 5.0,
    ) -> None:
        if ssl_context.verify_mode is not ssl.CERT_REQUIRED or not ssl_context.check_hostname:
            raise MutualTLSError("MTLSPeerClient requires CERT_REQUIRED and check_hostname=True.")
        self.host = host
        self.port = int(port)
        self.expected_server_dns = expected_server_dns
        self._ssl_ctx = ssl_context
        self.timeout_s = timeout_s

    def _connect(self) -> ssl.SSLSocket:
        raw_sock: socket.socket | None = None
        tls_sock: ssl.SSLSocket | None = None
        try:
            raw_sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
            raw_sock.settimeout(self.timeout_s)
            tls_sock = self._ssl_ctx.wrap_socket(
                raw_sock,
                server_hostname=self.expected_server_dns,
                do_handshake_on_connect=True,
            )
            server_names = _extract_peer_dns_names(tls_sock)
            if self.expected_server_dns not in server_names:
                raise MutualTLSError(f"Server certificate lacks expected SAN DNS {self.expected_server_dns!r}.")
            return tls_sock
        except Exception as exc:
            if tls_sock is not None:
                tls_sock.close()
            elif raw_sock is not None:
                raw_sock.close()
            if isinstance(exc, MutualTLSError):
                raise
            raise MutualTLSError(f"mTLS handshake failed: {exc}") from exc

    def invoke_control(self, body: Mapping[str, Any]) -> dict[str, Any]:
        with self._connect() as conn:
            _send_control_frame(conn, {"body": dict(body), "kind": "control_op"})
            resp = _recv_control_frame(conn)
        if resp.get("status") != "OK":
            raise MutualTLSError(f"Remote mTLS control_op refused: {resp}")
        return dict(resp["result"])

    def ping(self) -> dict[str, Any]:
        with self._connect() as conn:
            _send_control_frame(conn, {"kind": "ping"})
            resp = _recv_control_frame(conn)
        if resp.get("status") != "OK":
            raise MutualTLSError(f"Remote mTLS ping refused: {resp}")
        return resp

    def fetch_manifest(self, manifest_digest: str) -> PagedByteManifest:
        with self._connect() as conn:
            _send_control_frame(conn, {"kind": "m1_manifest", "manifest_digest": manifest_digest})
            resp = _recv_control_frame(conn)
            status = resp.get("status")
            if status == "MANIFEST_STREAM":
                m_len = int(resp.get("manifest_bytes", 0))
                if m_len <= 0 or m_len > MAX_MANIFEST_STREAM_BYTES:
                    raise MutualTLSError(f"Invalid streamed M1 manifest byte length: {m_len}.")
                raw_m = _recv_exact(conn, m_len)
                if hashlib.sha256(raw_m).hexdigest() != resp.get("manifest_sha256"):
                    raise MutualTLSError("Streamed M1 manifest SHA-256 mismatch.")
                manifest = PagedByteManifest.from_dict(parse_strict_json(raw_m))
            elif status == "OK":
                manifest = PagedByteManifest.from_dict(resp["manifest"])
            else:
                raise MutualTLSError(f"Failed to fetch M1 manifest: {resp}")
        if manifest.manifest_digest != manifest_digest:
            raise MutualTLSError("Remote M1 manifest digest mismatch.")
        return manifest

    def fetch_m1_object_resumable(
        self,
        manifest_digest: str,
        target_store: PagedObjectStore,
        *,
        max_reconnects: int = 6,
        chunk_size: int = DEFAULT_STREAM_CHUNK_BYTES,
        _fault_hook_after_pages: Callable[[int], None] | None = None,
    ) -> tuple[PagedByteManifest, int]:
        r"""Stream an M1 object into `target_store` in $O(\text{page\_size})$ constant RAM, resuming across link drops & process loss (`G14`).

        Returns `(verified_manifest, reconnect_count)`.
        """
        manifest = self.fetch_manifest(manifest_digest)
        if manifest.size_bytes == 0:
            if target_store.page_size == manifest.page_size:
                return target_store.register_manifest(manifest), 0
            return target_store.put_bytes(b"", media_type=manifest.media_type), 0

        eff_chunk = min(DEFAULT_STREAM_CHUNK_BYTES, max(1024, int(chunk_size)))
        reconnects = 0
        pages_newly_flushed = 0

        if target_store.page_size == manifest.page_size:
            # Scan already-persisted contiguous M1 pages (enabling exact suffix resume after os._exit)!
            content_hasher = hashlib.sha256()
            next_chunk_idx = 0
            while next_chunk_idx < len(manifest.chunks):
                ch = manifest.chunks[next_chunk_idx]
                if not target_store.backing.has_page(ch.page_digest):
                    break
                persisted = target_store.backing.get_page(ch.page_digest)
                if ch.byte_length < manifest.page_size and any(b != 0 for b in persisted[ch.byte_length :]):
                    raise MutualTLSError(f"Non-zero canonical padding in persisted M1 page {ch.page_digest}.")
                content_hasher.update(persisted[: ch.byte_length])
                next_chunk_idx += 1

            while next_chunk_idx < len(manifest.chunks):
                offset = next_chunk_idx * manifest.page_size
                remaining = manifest.size_bytes - offset
                page_buf = bytearray()
                conn_remaining = remaining
                try:
                    with self._connect() as conn:
                        _send_control_frame(
                            conn,
                            {
                                "chunk_size": eff_chunk,
                                "kind": "m1_range_stream",
                                "length": remaining,
                                "manifest_digest": manifest_digest,
                                "offset": offset,
                            },
                        )
                        hdr = _recv_control_frame(conn)
                        if hdr.get("status") != "STREAM_START":
                            raise ValueError(f"Stream rejected: {hdr}")
                        while True:
                            (clen,) = struct.unpack(">I", _recv_exact(conn, 4))
                            if clen == 0:
                                break
                            if clen > eff_chunk or clen > conn_remaining:
                                raise ValueError(f"Invalid M1 stream chunk length {clen}.")
                            chunk = _recv_exact(conn, clen)
                            expected_sha = _recv_exact(conn, 32)
                            if hashlib.sha256(chunk).digest() != expected_sha:
                                raise ValueError("Corrupted M1 stream chunk SHA-256 detected.")
                            conn_remaining -= clen

                            cursor = 0
                            while cursor < len(chunk) and next_chunk_idx < len(manifest.chunks):
                                target_ch = manifest.chunks[next_chunk_idx]
                                need = target_ch.byte_length - len(page_buf)
                                take = min(need, len(chunk) - cursor)
                                page_buf.extend(chunk[cursor : cursor + take])
                                cursor += take
                                if len(page_buf) == target_ch.byte_length:
                                    logical = bytes(page_buf)
                                    page_buf.clear()
                                    padded = logical + b"\x00" * (manifest.page_size - target_ch.byte_length)
                                    if hashlib.sha256(padded).hexdigest() != target_ch.page_digest:
                                        raise ValueError("Streamed M1 page SHA-256 disagrees with manifest chunk.")
                                    target_store.backing.put_page(padded)
                                    content_hasher.update(logical)
                                    next_chunk_idx += 1
                                    pages_newly_flushed += 1
                                    if _fault_hook_after_pages is not None:
                                        _fault_hook_after_pages(pages_newly_flushed)
                        if next_chunk_idx < len(manifest.chunks):
                            raise MutualTLSError("Premature stream EOF before all M1 pages completed.")
                except ValueError as exc:
                    raise MutualTLSError(str(exc)) from exc
                except (MutualTLSError, OSError) as exc:
                    reconnects += 1
                    if reconnects > max_reconnects:
                        raise MutualTLSError(
                            f"Exceeded max_reconnects={max_reconnects} while streaming M1 {manifest_digest}: {exc}"
                        ) from exc

            if content_hasher.hexdigest() != manifest.content_digest:
                raise MutualTLSError("Reassembled M1 object content_digest disagrees with source manifest.")
            return target_store.register_manifest(manifest), reconnects

        # Heterogeneous page_size fallback: still O(target_store.page_size) constant RAM!
        tgt_ps = target_store.page_size
        total_tgt_chunks = (manifest.size_bytes + tgt_ps - 1) // tgt_ps
        tgt_chunks: list[PageChunkRef] = []
        content_hasher = hashlib.sha256()
        next_tgt_idx = 0

        while next_tgt_idx < total_tgt_chunks:
            offset = next_tgt_idx * tgt_ps
            remaining = manifest.size_bytes - offset
            page_buf = bytearray()
            conn_remaining = remaining
            try:
                with self._connect() as conn:
                    _send_control_frame(
                        conn,
                        {
                            "chunk_size": eff_chunk,
                            "kind": "m1_range_stream",
                            "length": remaining,
                            "manifest_digest": manifest_digest,
                            "offset": offset,
                        },
                    )
                    hdr = _recv_control_frame(conn)
                    if hdr.get("status") != "STREAM_START":
                        raise ValueError(f"Stream rejected: {hdr}")
                    while True:
                        (clen,) = struct.unpack(">I", _recv_exact(conn, 4))
                        if clen == 0:
                            break
                        if clen > eff_chunk or clen > conn_remaining:
                            raise ValueError(f"Invalid M1 stream chunk length {clen}.")
                        chunk = _recv_exact(conn, clen)
                        expected_sha = _recv_exact(conn, 32)
                        if hashlib.sha256(chunk).digest() != expected_sha:
                            raise ValueError("Corrupted M1 stream chunk SHA-256 detected.")
                        conn_remaining -= clen

                        cursor = 0
                        while cursor < len(chunk) and next_tgt_idx < total_tgt_chunks:
                            exp_blen = min(tgt_ps, manifest.size_bytes - next_tgt_idx * tgt_ps)
                            need = exp_blen - len(page_buf)
                            take = min(need, len(chunk) - cursor)
                            page_buf.extend(chunk[cursor : cursor + take])
                            cursor += take
                            if len(page_buf) == exp_blen:
                                logical = bytes(page_buf)
                                page_buf.clear()
                                padded = logical + b"\x00" * (tgt_ps - exp_blen)
                                p_sha = target_store.backing.put_page(padded)
                                tgt_chunks.append(PageChunkRef(index=next_tgt_idx, page_digest=p_sha, byte_length=exp_blen))
                                content_hasher.update(logical)
                                next_tgt_idx += 1
                                pages_newly_flushed += 1
                                if _fault_hook_after_pages is not None:
                                    _fault_hook_after_pages(pages_newly_flushed)
                    if next_tgt_idx < total_tgt_chunks:
                        raise MutualTLSError("Premature stream EOF before all target M1 pages completed.")
            except ValueError as exc:
                raise MutualTLSError(str(exc)) from exc
            except (MutualTLSError, OSError) as exc:
                reconnects += 1
                if reconnects > max_reconnects:
                    raise MutualTLSError(
                        f"Exceeded max_reconnects={max_reconnects} while streaming M1 {manifest_digest}: {exc}"
                    ) from exc

        if content_hasher.hexdigest() != manifest.content_digest:
            raise MutualTLSError("Reassembled M1 object content_digest disagrees with source manifest.")
        rebuilt = PagedByteManifest(
            media_type=manifest.media_type,
            size_bytes=manifest.size_bytes,
            page_size=tgt_ps,
            content_digest=manifest.content_digest,
            chunks=tuple(tgt_chunks),
        )
        return target_store.register_manifest(rebuilt), reconnects


__all__ = [
    "DEFAULT_STREAM_CHUNK_BYTES",
    "MAX_CONTROL_FRAME_BYTES",
    "MTLSPeerClient",
    "MTLSPeerServer",
    "MutualTLSError",
    "PeerIdentityBundle",
    "create_mtls_client_context",
    "create_mtls_server_context",
    "generate_ephemeral_mtls_pki",
]
