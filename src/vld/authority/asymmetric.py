r"""Pure-Stdlib RFC 8032 Ed25519 Public-Key Authority & Host Keystore (`vld.authority.asymmetric`).

Mathematical specification (RFC 8032 §5.1 — Edwards-Curve Digital Signature Algorithm):
- Base field: $\mathbb{F}_p$ with prime $p = 2^{255} - 19$.
- Twisted Edwards curve $E(\mathbb{F}_p)$: $-x^2 + y^2 \equiv 1 + d x^2 y^2 \pmod p$,
  where $d = -121665 \cdot 121666^{-1} \bmod p$.
- Prime subgroup order: $\ell = 2^{252} + 27742317777372353535851937790883648493$, cofactor $h = 8$.
- Extended homogeneous coordinates $(X : Y : Z : T)$ satisfying $x = X/Z,\; y = Y/Z,\; x y = T/Z$
  admit complete unified addition (`add-2008-hwcd-3`) across distinct and identical points.
- Verification (`RFC 8032 §5.1.7` cofactored equation, interoperable with OpenSSL / Node.js `node:crypto`):
  Given canonical 32-byte public key $A \in E(\mathbb{F}_p)$, message $M \in \{0,1\}^*$, and 64-byte signature
  $(R_{\text{enc}} \,\|\, S_{\text{enc}})$ with $0 \le S < \ell$ and canonical $R \in E(\mathbb{F}_p)$:
  $$k = \operatorname{SHA512}(R_{\text{enc}} \,\|\, A_{\text{enc}} \,\|\, M) \bmod \ell, \qquad [8 S] B = [8] R + [8 k] A.$$

Security separation:
Private 32-byte seeds stay strictly inside individual `QuorumAuthorityMember` instances or `0600`
host-local keystores; `QuorumMembership`, `QuorumLineageCoordinator`, and published receipt bundles
carry **only** 32-byte public keys (`ed25519-rfc8032`), eliminating shared-secret disclosure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX128_RE = re.compile(r"[0-9a-f]{128}\Z")

# RFC 8032 §5.1 curve constants
_P: int = (1 << 255) - 19
_L: int = (1 << 252) + 27742317777372353535851937790883648493
_D: int = (-121665 * pow(121666, _P - 2, _P)) % _P
_TWO_D: int = (2 * _D) % _P
_SQRT_M1: int = pow(2, (_P - 1) // 4, _P)

# ASN.1 DER prefixes (RFC 8410) for OpenSSL / WebCrypto / Node.js `node:crypto` interop
ED25519_SPKI_PREFIX: bytes = bytes.fromhex("302a300506032b6570032100")
ED25519_PKCS8_PREFIX: bytes = bytes.fromhex("302e020100300506032b657004220420")

ExtendedPoint = tuple[int, int, int, int]
_NEUTRAL_POINT: ExtendedPoint = (0, 1, 1, 0)


class AsymmetricAuthorityError(ValueError):
    """Raised when an Ed25519 key, point encoding, keystore permission, or signature is invalid."""


def _recover_x(y: int, sign: int) -> int:
    r"""Recover $x \in \mathbb{F}_p$ from $y$ and parity bit `sign` per RFC 8032 §5.1.1."""
    if y < 0 or y >= _P:
        raise AsymmetricAuthorityError("Non-canonical Ed25519 y-coordinate (>= p).")
    y2 = (y * y) % _P
    u = (y2 - 1) % _P
    v = (_D * y2 + 1) % _P
    v3 = (v * v * v) % _P
    v7 = (v3 * v3 * v) % _P
    x = (u * v3 * pow((u * v7) % _P, (_P - 5) // 8, _P)) % _P
    vx2 = (v * x * x) % _P
    if vx2 == u:
        pass
    elif vx2 == (_P - u) % _P:
        x = (x * _SQRT_M1) % _P
    else:
        raise AsymmetricAuthorityError("Point encoding does not lie on Curve25519/Ed25519.")
    if x == 0 and sign == 1:
        raise AsymmetricAuthorityError("Invalid negative zero x-coordinate in Ed25519 point.")
    if (x & 1) != sign:
        x = _P - x
    return x


def _make_base_point() -> ExtendedPoint:
    by = (4 * pow(5, _P - 2, _P)) % _P
    bx = _recover_x(by, 0)
    return (bx, by, 1, (bx * by) % _P)


_BASE_POINT: ExtendedPoint = _make_base_point()


def _point_add(p1: ExtendedPoint, p2: ExtendedPoint) -> ExtendedPoint:
    """Unified extended twisted Edwards addition (`add-2008-hwcd-3`, RFC 8032 §5.1.4)."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % _P
    b = ((y1 + x1) * (y2 + x2)) % _P
    c = (t1 * _TWO_D * t2) % _P
    d = (z1 * 2 * z2) % _P
    e = (b - a) % _P
    f = (d - c) % _P
    g = (d + c) % _P
    h = (b + a) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _point_double(pt: ExtendedPoint) -> ExtendedPoint:
    """Dedicated doubling on $a = -1$ extended twisted Edwards (`dbl-2008-hwcd`, RFC 8032 §5.1.4)."""
    x1, y1, z1, _ = pt
    a = (x1 * x1) % _P
    b = (y1 * y1) % _P
    c = (2 * z1 * z1) % _P
    h = (a + b) % _P
    xpy = (x1 + y1) % _P
    e = (h - xpy * xpy) % _P
    g = (a - b) % _P
    f = (c + g) % _P
    return ((e * f) % _P, (g * h) % _P, (f * g) % _P, (e * h) % _P)


def _scalar_mul(scalar: int, pt: ExtendedPoint) -> ExtendedPoint:
    if scalar < 0:
        raise AsymmetricAuthorityError("Scalar must be non-negative.")
    res = _NEUTRAL_POINT
    addend = pt
    k = scalar
    while k > 0:
        if k & 1:
            res = _point_add(res, addend)
        addend = _point_double(addend)
        k >>= 1
    return res


def _point_equal(p1: ExtendedPoint, p2: ExtendedPoint) -> bool:
    x1, y1, z1, _ = p1
    x2, y2, z2, _ = p2
    return ((x1 * z2 - x2 * z1) % _P == 0) and ((y1 * z2 - y2 * z1) % _P == 0)


def _encode_point(pt: ExtendedPoint) -> bytes:
    x, y, z, _ = pt
    zi = pow(z, _P - 2, _P)
    affine_x = (x * zi) % _P
    affine_y = (y * zi) % _P
    enc = affine_y | ((affine_x & 1) << 255)
    return enc.to_bytes(32, "little")


def _decode_point_strict(raw: bytes) -> ExtendedPoint:
    if not isinstance(raw, (bytes, bytearray)) or len(raw) != 32:
        raise AsymmetricAuthorityError("Ed25519 encoded point must be exactly 32 bytes.")
    raw_bytes = bytes(raw)
    unpacked = int.from_bytes(raw_bytes, "little")
    sign = (unpacked >> 255) & 1
    y = unpacked & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    pt: ExtendedPoint = (x, y, 1, (x * y) % _P)
    # Enforce canonical encoding round-trip
    if _encode_point(pt) != raw_bytes:
        raise AsymmetricAuthorityError("Non-canonical Ed25519 point encoding.")
    return pt


def _expand_secret_seed(seed32: bytes) -> tuple[int, bytes, bytes]:
    if not isinstance(seed32, bytes) or len(seed32) != 32:
        raise AsymmetricAuthorityError("Ed25519 private seed must be exactly 32 bytes.")
    digest = hashlib.sha512(seed32).digest()
    clamped = bytearray(digest[:32])
    clamped[0] &= 248
    clamped[31] &= 127
    clamped[31] |= 64
    scalar = int.from_bytes(clamped, "little")
    prefix = digest[32:64]
    pub_bytes = _encode_point(_scalar_mul(scalar, _BASE_POINT))
    return scalar, prefix, pub_bytes


def ed25519_derive_public_key(seed32: bytes) -> bytes:
    """Derive canonical 32-byte Ed25519 public key from a 32-byte private seed (RFC 8032 §5.1.5)."""
    _, _, pub = _expand_secret_seed(seed32)
    return pub


def ed25519_validate_public_key_hex(public_key_hex: str) -> bytes:
    """Validate a 64-char lowercase hex Ed25519 public key on Curve25519 and return its 32 bytes."""
    if not isinstance(public_key_hex, str) or _HEX64_RE.fullmatch(public_key_hex) is None:
        raise AsymmetricAuthorityError("Ed25519 public_key_hex must be 64 lowercase hex characters.")
    raw = bytes.fromhex(public_key_hex)
    _decode_point_strict(raw)
    return raw


def ed25519_sign(seed32: bytes, message: bytes) -> bytes:
    """Deterministically sign `message` under `seed32` returning 64-byte `(R || S)` signature (RFC 8032 §5.1.6)."""
    if not isinstance(message, (bytes, bytearray)):
        raise AsymmetricAuthorityError("Message must be bytes.")
    msg = bytes(message)
    scalar, prefix, pub_bytes = _expand_secret_seed(seed32)
    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % _L
    r_enc = _encode_point(_scalar_mul(r, _BASE_POINT))
    k = int.from_bytes(hashlib.sha512(r_enc + pub_bytes + msg).digest(), "little") % _L
    s = (r + k * scalar) % _L
    return r_enc + s.to_bytes(32, "little")


def ed25519_verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Strictly verify a 64-byte Ed25519 signature against 32-byte `public_key` (RFC 8032 §5.1.7)."""
    if (
        not isinstance(public_key, (bytes, bytearray))
        or len(public_key) != 32
        or not isinstance(signature, (bytes, bytearray))
        or len(signature) != 64
        or not isinstance(message, (bytes, bytearray))
    ):
        return False
    pub_b = bytes(public_key)
    sig_b = bytes(signature)
    msg_b = bytes(message)
    r_enc = sig_b[:32]
    s_int = int.from_bytes(sig_b[32:], "little")
    if s_int >= _L:
        return False
    try:
        a_pt = _decode_point_strict(pub_b)
        r_pt = _decode_point_strict(r_enc)
    except AsymmetricAuthorityError:
        return False
    k = int.from_bytes(hashlib.sha512(r_enc + pub_b + msg_b).digest(), "little") % _L
    # Cofactored verification equation: [8*S]B == [8]R + [8*k]A
    lhs = _scalar_mul(8 * s_int, _BASE_POINT)
    rhs = _point_add(_scalar_mul(8, r_pt), _scalar_mul(8 * k, a_pt))
    return _point_equal(lhs, rhs)


@dataclass(frozen=True, slots=True)
class Ed25519KeyPair:
    """Host-private Ed25519 signing identity whose public key can be safely published in evidence bundles."""

    key_id: str
    _seed: bytes
    public_key_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.key_id, str) or not self.key_id.strip():
            raise AsymmetricAuthorityError("key_id must be a non-empty string.")
        expected_pub = ed25519_derive_public_key(self._seed)
        if self.public_key_bytes != expected_pub:
            raise AsymmetricAuthorityError("Ed25519 public_key_bytes disagrees with private seed.")

    def __repr__(self) -> str:
        # Never leak `_seed` in repr/logs!
        return f"Ed25519KeyPair(key_id={self.key_id!r}, public_key_hex={self.public_key_hex!r})"

    @classmethod
    def generate(cls, key_id: str) -> Ed25519KeyPair:
        """Generate a cryptographically random Ed25519 identity via `os.urandom(32)`."""
        seed = os.urandom(32)
        return cls.from_seed(key_id, seed)

    @classmethod
    def from_seed(cls, key_id: str, seed32: bytes) -> Ed25519KeyPair:
        """Construct an Ed25519 identity from a 32-byte private seed."""
        pub = ed25519_derive_public_key(seed32)
        return cls(key_id=key_id, _seed=bytes(seed32), public_key_bytes=pub)

    @property
    def public_key_hex(self) -> str:
        return self.public_key_bytes.hex()

    def to_spki_der(self) -> bytes:
        """Export RFC 8410 ASN.1 DER `SubjectPublicKeyInfo` (44 bytes, compatible with OpenSSL & Node.js)."""
        return ED25519_SPKI_PREFIX + self.public_key_bytes

    def to_pkcs8_der(self) -> bytes:
        """Export RFC 8410 ASN.1 DER `OneAsymmetricKey` v1 PKCS#8 private key (48 bytes)."""
        return ED25519_PKCS8_PREFIX + self._seed

    def sign(self, message: bytes) -> bytes:
        return ed25519_sign(self._seed, message)

    def sign_hex(self, message: bytes) -> str:
        return ed25519_sign(self._seed, message).hex()

    def public_identity_dict(self) -> dict[str, str]:
        """Public-only descriptor safe for inclusion in public manifests and evidence bundles."""
        return {
            "key_id": self.key_id,
            "public_key_hex": self.public_key_hex,
            "public_key_spki_hex": self.to_spki_der().hex(),
            "scheme": "ed25519-rfc8032",
        }

    def save_private_keystore(self, path: str | Path) -> Path:
        """Persist private seed atomically with `0600` owner-only permissions outside evidence bundles."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{self.key_id}\n{self._seed.hex()}\n".encode("ascii")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(p), flags, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        return p

    @classmethod
    def load_private_keystore(cls, path: str | Path) -> Ed25519KeyPair:
        raw = Path(path).read_text(encoding="ascii").splitlines()
        if len(raw) < 2 or _HEX64_RE.fullmatch(raw[1].strip()) is None:
            raise AsymmetricAuthorityError("Malformed Ed25519 private keystore file.")
        return cls.from_seed(raw[0].strip(), bytes.fromhex(raw[1].strip()))


def build_ed25519_quorum_identities(
    member_ids: tuple[str, ...] | list[str],
    *,
    deterministic_domain_seed: bytes | None = None,
) -> tuple[dict[str, Ed25519KeyPair], dict[str, str]]:
    """Create isolated per-member Ed25519 keypairs + a publishable `{member_id: public_key_hex}` map."""
    keypairs: dict[str, Ed25519KeyPair] = {}
    public_keys: dict[str, str] = {}
    for mid in sorted(set(str(m) for m in member_ids)):
        if deterministic_domain_seed is None:
            kp = Ed25519KeyPair.generate(mid)
        else:
            seed = hashlib.sha256(b"vld.ed25519-keygen.v1\x00" + deterministic_domain_seed + b"\x00" + mid.encode("utf-8")).digest()
            kp = Ed25519KeyPair.from_seed(mid, seed)
        keypairs[mid] = kp
        public_keys[mid] = kp.public_key_hex
    return keypairs, public_keys


def verify_ed25519_hex(public_key_hex: str, message: bytes, signature_hex: str) -> bool:
    """Verify lowercase hex Ed25519 public key (64 chars) and signature (128 chars) on `message`."""
    if (
        not isinstance(public_key_hex, str)
        or _HEX64_RE.fullmatch(public_key_hex) is None
        or not isinstance(signature_hex, str)
        or _HEX128_RE.fullmatch(signature_hex) is None
    ):
        return False
    return ed25519_verify(bytes.fromhex(public_key_hex), message, bytes.fromhex(signature_hex))


def extract_public_key_map(identities: Mapping[str, Any]) -> dict[str, str]:
    """Normalize `{member_id: Ed25519KeyPair | hex_str | bytes}` into canonical `{member_id: hex_str}`."""
    out: dict[str, str] = {}
    for mid, val in sorted(identities.items()):
        if isinstance(val, Ed25519KeyPair):
            out[str(mid)] = val.public_key_hex
        elif isinstance(val, (bytes, bytearray)):
            raw = bytes(val)
            _decode_point_strict(raw)
            out[str(mid)] = raw.hex()
        elif isinstance(val, str):
            ed25519_validate_public_key_hex(val)
            out[str(mid)] = val
        else:
            raise AsymmetricAuthorityError(f"Unsupported public key descriptor for {mid!r}.")
    return out


__all__ = [
    "AsymmetricAuthorityError",
    "ED25519_PKCS8_PREFIX",
    "ED25519_SPKI_PREFIX",
    "Ed25519KeyPair",
    "build_ed25519_quorum_identities",
    "ed25519_derive_public_key",
    "ed25519_sign",
    "ed25519_validate_public_key_hex",
    "ed25519_verify",
    "extract_public_key_map",
    "verify_ed25519_hex",
]
