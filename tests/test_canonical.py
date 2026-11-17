"""Unit and edge-case tests for vld.canonical."""

from __future__ import annotations

import math
import pytest

from vld.canonical import (
    CanonicalizationError,
    canonical_bytes,
    canonical_json,
    domain_digest,
    parse_strict_json,
)


def test_key_order_and_tuple_normalization() -> None:
    left = {"b": (1, 2, {"y": True, "x": None}), "a": "hello"}
    right = {"a": "hello", "b": [1, 2, {"x": None, "y": True}]}
    assert canonical_json(left) == canonical_json(right)
    assert canonical_bytes(left) == canonical_bytes(right)
    assert domain_digest("vld.test", left) == domain_digest("vld.test", right)


def test_negative_zero_normalizes_to_positive_zero() -> None:
    assert canonical_json({"v": -0.0}) == '{"v":0.0}'
    assert canonical_json(-0.0) == canonical_json(0.0)


@pytest.mark.parametrize("bad_float", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_rejected(bad_float: float) -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json({"bad": bad_float})


def test_non_string_keys_and_unsupported_types_rejected() -> None:
    with pytest.raises(CanonicalizationError):
        canonical_json({1: "one"})
    with pytest.raises(CanonicalizationError):
        canonical_json({"set": {1, 2}})


def test_domain_digest_separation_and_validation() -> None:
    payload = {"k": 42}
    assert domain_digest("vld.alpha", payload) != domain_digest("vld.beta", payload)
    with pytest.raises(CanonicalizationError):
        domain_digest("", payload)
    with pytest.raises(CanonicalizationError):
        domain_digest("bad\x00domain", payload)


def test_parse_strict_json_rejects_duplicates_and_nan_constants() -> None:
    with pytest.raises(CanonicalizationError, match="Duplicate JSON key"):
        parse_strict_json('{"a": 1, "a": 2}')
    with pytest.raises(CanonicalizationError, match="Non-finite JSON constant"):
        parse_strict_json('{"a": NaN}')
    with pytest.raises(CanonicalizationError, match="Non-finite JSON constant"):
        parse_strict_json('{"a": Infinity}')
    with pytest.raises(CanonicalizationError, match="Invalid JSON"):
        parse_strict_json("{unclosed")


def test_surrogates_and_custom_str_key_collisions_rejected() -> None:
    # Lone UTF-16 surrogates must be rejected consistently across canonical_json, canonical_bytes, and parse_strict_json
    for fn in (canonical_json, canonical_bytes):
        with pytest.raises(CanonicalizationError, match="surrogate"):
            fn("\ud800")
        with pytest.raises(CanonicalizationError, match="surrogate"):
            fn({"\ud800": 1})
    with pytest.raises(CanonicalizationError, match="surrogate"):
        domain_digest("bad\ud800domain", {"k": 1})
    with pytest.raises(CanonicalizationError, match="surrogate"):
        parse_strict_json('"\\ud800"')

    # str subclasses with custom hash/eq must not emit duplicate keys in canonical_json
    class DistinctIdStr(str):
        def __hash__(self) -> int:
            return id(self)

        def __eq__(self, other: object) -> bool:
            return self is other

    colliding = {DistinctIdStr("k"): 1, "k": 2}
    with pytest.raises(CanonicalizationError, match="Duplicate canonical object key"):
        canonical_json(colliding)
