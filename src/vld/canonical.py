r"""Deterministic canonical serialization and domain-separated SHA-256 addressing.

Mathematical contract:
For canonicalizable finite JSON-compatible values $x \in \mathcal{J}$:
  1. $\operatorname{decode}(\operatorname{canonical\_bytes}(x)) = x$
  2. $x \equiv_{\mathcal{J}} y \iff \operatorname{canonical\_bytes}(x) = \operatorname{canonical\_bytes}(y)$
  3. Domain-separated digest:
       $H_d(x) = \operatorname{SHA256}(\operatorname{UTF8}(d) \,\|\, \text{0x00} \,\|\, \operatorname{canonical\_bytes}(x))$

Non-finite floats ($\text{NaN}, \pm\infty$) and non-string mapping keys are explicitly rejected
so that identity hashes are total, platform-invariant, and unambiguous.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class CanonicalizationError(ValueError):
    """Raised when a value violates the canonical serialization domain."""


def _normalize_utf8_str(value: str, context: str) -> str:
    """Extract exact plain str and reject lone UTF-16 surrogates (RFC 8259 / I-JSON)."""
    plain = str.__str__(value)
    try:
        plain.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CanonicalizationError(f"Non-UTF8 surrogate string rejected in {context}: {exc}") from exc
    return plain


def _validate_canonical_value(value: Any) -> Any:
    """Recursively normalize tuples/subclasses and reject non-canonical values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return _normalize_utf8_str(value, "canonical value")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(f"Non-finite float is forbidden in canonical data: {value!r}")
        # Normalize -0.0 to 0.0 so numerically identical zeros share one byte representation.
        return 0.0 if value == 0.0 else float(value)
    if isinstance(value, (list, tuple)):
        return [_validate_canonical_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise CanonicalizationError(f"Canonical object keys must be str, got {type(k).__name__}")
            norm_k = _normalize_utf8_str(k, "canonical object key")
            if norm_k in normalized:
                raise CanonicalizationError(f"Duplicate canonical object key rejected: {norm_k!r}")
            normalized[norm_k] = _validate_canonical_value(v)
        return normalized
    raise CanonicalizationError(f"Unsupported type for canonical serialization: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return compact, key-sorted, UTF-8-safe canonical JSON string."""
    normalized = _validate_canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Return UTF-8 bytes of the canonical JSON representation."""
    return canonical_json(value).encode("utf-8")


def domain_digest(domain: str, payload: bytes | Any) -> str:
    r"""Compute domain-separated SHA-256 hex digest.

    Equation:
        $H_d(payload) = \operatorname{SHA256}(\operatorname{UTF8}(d) \,\|\, \text{NUL} \,\|\, B)$
    where $B = payload$ if `payload` is raw `bytes`, else $\operatorname{canonical\_bytes}(payload)$.
    """
    if not isinstance(domain, str) or not domain or "\x00" in domain:
        raise CanonicalizationError("Domain tag must be a non-empty str without NUL.")
    norm_domain = _normalize_utf8_str(domain, "domain tag")
    raw = payload if isinstance(payload, bytes) else canonical_bytes(payload)
    hasher = hashlib.sha256()
    hasher.update(norm_domain.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(raw)
    return hasher.hexdigest()


def parse_strict_json(raw: str | bytes) -> Any:
    """Parse JSON while rejecting duplicate object keys and non-finite constants."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, val in pairs:
            if key in result:
                raise CanonicalizationError(f"Duplicate JSON key rejected: {key!r}")
            result[key] = val
        return result

    def _reject_constant(token: str) -> Any:
        raise CanonicalizationError(f"Non-finite JSON constant rejected: {token!r}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise CanonicalizationError(f"Invalid JSON input: {exc}") from exc

    return _validate_canonical_value(parsed)
