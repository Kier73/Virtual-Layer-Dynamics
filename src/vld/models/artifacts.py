r"""Phase 6 — Source-Pinned Model Artifacts, Safetensors Range Reader (`HFR`), BPE Tokenizer Ground & Lazy Parameter Manifolds."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.memory import PagedByteManifest, PagedObjectStore

MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_TENSOR_NAME_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,255}\Z")
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")
_HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z")

SAFETENSORS_DTYPE_WIDTHS: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class ModelArtifactError(ValueError):
    """Raised when a Safetensors header, tokenizer Ground, or parameter space violates invariants."""


def _strict_nonneg_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val < 0:
        raise ModelArtifactError(f"{name} must be a non-negative integer.")
    return val


def _strict_pos_int(val: Any, name: str) -> int:
    if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
        raise ModelArtifactError(f"{name} must be a positive integer.")
    return val


# ============================================================================
# 1. Bounded Safetensors Range Reader (HFR) & Builder
# ============================================================================


@dataclass(frozen=True, slots=True)
class SafetensorsTensorEntry:
    """Verified tensor entry from a Safetensors shard header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]
    shard: str

    @property
    def byte_length(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True, slots=True)
class SafetensorsShardObservation:
    """Bounded header-only observation of a Safetensors shard (payload bytes untouched)."""

    shard_name: str
    file_size_bytes: int
    header_length_bytes: int
    data_section_bytes: int
    header_sha256: str
    metadata: Mapping[str, str]
    tensors: tuple[SafetensorsTensorEntry, ...]

    @property
    def data_section_offset(self) -> int:
        return 8 + self.header_length_bytes

    def tensor_entry(self, name: str) -> SafetensorsTensorEntry:
        for t in self.tensors:
            if t.name == name:
                return t
        raise ModelArtifactError(f"Tensor {name!r} not found in shard {self.shard_name!r}.")

    def absolute_byte_range(self, name: str) -> tuple[int, int]:
        """Return `(absolute_offset, byte_length)` for range-fetching a single tensor payload."""
        entry = self.tensor_entry(name)
        return self.data_section_offset + entry.data_offsets[0], entry.byte_length


def observe_safetensors_header(
    range_reader: Callable[[int, int], bytes],
    total_size_bytes: int,
    *,
    shard_name: str = "model.safetensors",
) -> SafetensorsShardObservation:
    r"""Observe a Safetensors shard via bounded `[0..8)` and `[8..8+H)` range reads without reading tensor weights."""
    _strict_pos_int(total_size_bytes, "total_size_bytes")
    if total_size_bytes < 10:
        raise ModelArtifactError(f"{shard_name}: file smaller than minimal Safetensors header.")

    prefix = range_reader(0, 8)
    if not isinstance(prefix, bytes) or len(prefix) != 8:
        raise ModelArtifactError(f"{shard_name}: failed to read 8-byte header prefix.")
    (header_len,) = struct.unpack("<Q", prefix)
    if header_len < 2 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
        raise ModelArtifactError(f"{shard_name}: invalid Safetensors header length {header_len}.")
    if 8 + header_len > total_size_bytes:
        raise ModelArtifactError(f"{shard_name}: declared header extends past file boundary.")

    header_raw = range_reader(8, header_len)
    if not isinstance(header_raw, bytes) or len(header_raw) != header_len:
        raise ModelArtifactError(f"{shard_name}: truncated Safetensors header bytes.")

    try:
        parsed = parse_strict_json(header_raw.rstrip(b" "))
    except Exception as exc:
        raise ModelArtifactError(f"{shard_name}: malformed Safetensors JSON header: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ModelArtifactError(f"{shard_name}: Safetensors header root must be an object.")

    raw_meta = parsed.pop("__metadata__", {})
    if not isinstance(raw_meta, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in raw_meta.items()
    ):
        raise ModelArtifactError(f"{shard_name}: __metadata__ must map strings to strings.")

    data_bytes = total_size_bytes - 8 - header_len
    entries: list[SafetensorsTensorEntry] = []
    intervals: list[tuple[int, int, str]] = []
    for name, info in sorted(parsed.items()):
        if not isinstance(name, str) or not _TENSOR_NAME_RE.fullmatch(name) or ".." in name:
            raise ModelArtifactError(f"{shard_name}: invalid tensor name {name!r}.")
        if not isinstance(info, dict) or set(info.keys()) != {"dtype", "shape", "data_offsets"}:
            raise ModelArtifactError(f"{shard_name}: invalid descriptor keys for {name!r}.")
        dtype = info["dtype"]
        shape_raw = info["shape"]
        offsets_raw = info["data_offsets"]
        if not isinstance(dtype, str) or dtype not in SAFETENSORS_DTYPE_WIDTHS:
            raise ModelArtifactError(f"{shard_name}: unsupported dtype {dtype!r} for {name!r}.")
        if not isinstance(shape_raw, list) or len(shape_raw) > 32:
            raise ModelArtifactError(f"{shard_name}: invalid shape list for {name!r}.")
        shape = tuple(_strict_nonneg_int(d, f"{name}.shape") for d in shape_raw)
        if not isinstance(offsets_raw, list) or len(offsets_raw) != 2:
            raise ModelArtifactError(f"{shard_name}: data_offsets must have 2 integers for {name!r}.")
        start = _strict_nonneg_int(offsets_raw[0], f"{name}.start")
        end = _strict_nonneg_int(offsets_raw[1], f"{name}.end")
        expected_bytes = math.prod(shape) * SAFETENSORS_DTYPE_WIDTHS[dtype]
        if end < start or end > data_bytes or (end - start) != expected_bytes:
            raise ModelArtifactError(
                f"{shard_name}: tensor {name!r} extent [{start}, {end}) disagrees with shape {shape} and dtype {dtype}."
            )
        intervals.append((start, end, name))
        entries.append(
            SafetensorsTensorEntry(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offsets=(start, end),
                shard=shard_name,
            )
        )

    if not entries:
        raise ModelArtifactError(f"{shard_name}: Safetensors shard declares zero tensors.")

    prev_end = 0
    for start, end, tname in sorted(intervals):
        if end > start and start < prev_end:
            raise ModelArtifactError(f"{shard_name}: overlapping tensor data range at {tname!r}.")
        prev_end = max(prev_end, end)

    header_sha = hashlib.sha256(prefix + header_raw).hexdigest()
    return SafetensorsShardObservation(
        shard_name=shard_name,
        file_size_bytes=total_size_bytes,
        header_length_bytes=header_len,
        data_section_bytes=data_bytes,
        header_sha256=header_sha,
        metadata=MappingProxyType(dict(raw_meta)),
        tensors=tuple(entries),
    )


def build_safetensors_bytes(
    tensors: Mapping[str, tuple[str, Sequence[int], bytes]],
    *,
    metadata: Mapping[str, str] | None = None,
) -> bytes:
    """Serialize `(dtype, shape, raw_bytes)` entries into a valid, canonical `.safetensors` byte buffer."""
    if not tensors:
        raise ModelArtifactError("build_safetensors_bytes requires at least one tensor.")
    header_obj: dict[str, Any] = {}
    if metadata:
        header_obj["__metadata__"] = {str(k): str(v) for k, v in sorted(metadata.items())}

    payload_parts: list[bytes] = []
    cursor = 0
    for name in sorted(tensors.keys()):
        if not _TENSOR_NAME_RE.fullmatch(name) or ".." in name:
            raise ModelArtifactError(f"Invalid tensor name {name!r}.")
        dtype, shape_seq, raw = tensors[name]
        if dtype not in SAFETENSORS_DTYPE_WIDTHS:
            raise ModelArtifactError(f"Unsupported dtype {dtype!r}.")
        shape = [int(d) for d in shape_seq]
        expected_len = math.prod(shape) * SAFETENSORS_DTYPE_WIDTHS[dtype]
        if not isinstance(raw, bytes) or len(raw) != expected_len:
            raise ModelArtifactError(f"Tensor {name!r} byte length {len(raw)} != expected {expected_len}.")
        header_obj[name] = {
            "data_offsets": [cursor, cursor + expected_len],
            "dtype": dtype,
            "shape": shape,
        }
        payload_parts.append(raw)
        cursor += expected_len

    header_bytes = json.dumps(header_obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    # Pad header with spaces to 8-byte alignment per Safetensors convention
    pad_len = (8 - (len(header_bytes) % 8)) % 8
    header_padded = header_bytes + (b" " * pad_len)
    prefix = struct.pack("<Q", len(header_padded))
    return prefix + header_padded + b"".join(payload_parts)


# ============================================================================
# 2. Source-Pinned Deterministic Byte-Level BPE Tokenizer Ground
# ============================================================================


@dataclass(frozen=True, slots=True)
class TokenizerRoundTripReceipt:
    """Verified receipt proving exact UTF-8 byte round-trip conservation across tokenization."""

    tokenizer_digest: str
    source_utf8_sha256: str
    byte_length: int
    token_ids: tuple[int, ...]
    receipt_digest: str


class ByteLevelBPETokenizerGround:
    r"""Deterministic byte-level BPE Tokenizer Ground with bijective UTF-8 byte base (`0..255`).

    Merge rules `(left_id, right_id) -> merged_id` ($256 + k$) are applied deterministically
    by lowest merge priority index, guaranteeing $\operatorname{decode}(\operatorname{encode}(s)) = s$
    for every valid UTF-8 string $s$.
    """

    def __init__(self, merges: Sequence[tuple[int, int]] = (), *, tokenizer_id: str = "vld.bpe.byte256") -> None:
        if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
            raise ModelArtifactError("tokenizer_id must be a non-empty string.")
        self.tokenizer_id = tokenizer_id
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        ordered_merges: list[tuple[int, int, int]] = []
        for idx, pair in enumerate(merges):
            if len(pair) != 2:
                raise ModelArtifactError("Each BPE merge must be a (left_id, right_id) pair.")
            left, right = int(pair[0]), int(pair[1])
            if left not in vocab or right not in vocab:
                raise ModelArtifactError(f"BPE merge {(left, right)} references undefined token ID.")
            if (left, right) in merge_ranks:
                raise ModelArtifactError(f"Duplicate BPE merge pair {(left, right)}.")
            new_id = 256 + idx
            vocab[new_id] = vocab[left] + vocab[right]
            merge_ranks[(left, right)] = (idx, new_id)
            ordered_merges.append((left, right, new_id))

        self._vocab = vocab
        self._merge_ranks = merge_ranks
        self.merges = tuple(ordered_merges)
        self.vocab_size = len(vocab)

    @classmethod
    def from_corpus(cls, texts: Sequence[str], *, num_merges: int = 32, tokenizer_id: str = "vld.bpe.learned") -> ByteLevelBPETokenizerGround:
        """Deterministically induce `num_merges` BPE merges from sample UTF-8 strings."""
        sequences = [list(t.encode("utf-8", errors="strict")) for t in texts]
        merges: list[tuple[int, int]] = []
        next_id = 256
        for _ in range(num_merges):
            counts: dict[tuple[int, int], int] = {}
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            # Deterministic tie-breaking: highest count, then lexicographically smallest pair
            best_pair = min(counts.keys(), key=lambda p: (-counts[p], p[0], p[1]))
            if counts[best_pair] < 2:
                break
            merges.append(best_pair)
            new_seqs: list[list[int]] = []
            for seq in sequences:
                merged_seq: list[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best_pair:
                        merged_seq.append(next_id)
                        i += 2
                    else:
                        merged_seq.append(seq[i])
                        i += 1
                new_seqs.append(merged_seq)
            sequences = new_seqs
            next_id += 1
        return cls(merges, tokenizer_id=tokenizer_id)

    @property
    def digest(self) -> str:
        return domain_digest(
            "vld.bpe-tokenizer-ground.v1",
            {
                "merges": [list(m) for m in self.merges],
                "tokenizer_id": self.tokenizer_id,
                "vocab_size": self.vocab_size,
            },
        )

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ModelArtifactError("encode requires a str.")
        try:
            raw = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ModelArtifactError(f"Non-UTF8 string rejected by TokenizerGround: {exc}") from exc
        ids = list(raw)
        if not self._merge_ranks or len(ids) < 2:
            return tuple(ids)

        while len(ids) >= 2:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            best_new_id = -1
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                info = self._merge_ranks.get(pair)
                if info is not None and (best_rank is None or info[0] < best_rank):
                    best_rank, best_new_id = info
                    best_pair = pair
            if best_pair is None:
                break
            nxt: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best_pair:
                    nxt.append(best_new_id)
                    i += 2
                else:
                    nxt.append(ids[i])
                    i += 1
            ids = nxt
        return tuple(ids)

    def decode_bytes(self, token_ids: Sequence[int]) -> bytes:
        parts: list[bytes] = []
        for tid in token_ids:
            if isinstance(tid, bool) or not isinstance(tid, int) or tid not in self._vocab:
                raise ModelArtifactError(f"Unknown token ID {tid!r} for TokenizerGround.")
            parts.append(self._vocab[tid])
        return b"".join(parts)

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = self.decode_bytes(token_ids)
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ModelArtifactError(f"Token sequence does not decode to valid UTF-8: {exc}") from exc

    def encode_with_receipt(self, text: str) -> TokenizerRoundTripReceipt:
        tokens = self.encode(text)
        recovered = self.decode(tokens)
        if recovered != text:
            raise ModelArtifactError("Tokenizer round-trip conservation check failed.")
        raw = text.encode("utf-8")
        src_sha = hashlib.sha256(raw).hexdigest()
        rec_digest = domain_digest(
            "vld.tokenizer-receipt.v1",
            {
                "byte_length": len(raw),
                "source_utf8_sha256": src_sha,
                "token_ids": list(tokens),
                "tokenizer_digest": self.digest,
            },
        )
        return TokenizerRoundTripReceipt(
            tokenizer_digest=self.digest,
            source_utf8_sha256=src_sha,
            byte_length=len(raw),
            token_ids=tokens,
            receipt_digest=rec_digest,
        )


# ============================================================================
# 3. Content-Addressed Model Parameter Manifolds (MWS) & Pinned Qwen3 Profiles
# ============================================================================


@dataclass(frozen=True, slots=True)
class TensorSpace:
    """Typed coordinate space of a single named model tensor."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _TENSOR_NAME_RE.fullmatch(self.name) or ".." in self.name:
            raise ModelArtifactError(f"Invalid tensor name {self.name!r}.")
        if self.dtype not in SAFETENSORS_DTYPE_WIDTHS:
            raise ModelArtifactError(f"Unsupported tensor dtype {self.dtype!r}.")
        dims = tuple(_strict_nonneg_int(d, "shape") for d in self.shape)
        if len(dims) > 32:
            raise ModelArtifactError("Tensor rank exceeds 32.")
        object.__setattr__(self, "shape", dims)

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def byte_size(self) -> int:
        return self.elements * SAFETENSORS_DTYPE_WIDTHS[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_order": "little",
            "dtype": self.dtype,
            "layout": "C-contiguous",
            "name": self.name,
            "shape": list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class ModelParameterSpace:
    r"""Storage-independent parameter space $\mathcal{W}_A = \prod_i D_i^{s_i}$."""

    architecture_id: str
    config: Mapping[str, Any]
    tensors: tuple[TensorSpace, ...]
    _config_json: str = field(init=False, repr=False, compare=False)
    _by_name: Mapping[str, TensorSpace] = field(init=False, repr=False, compare=False)
    _identity_cache: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_id, str) or not self.architecture_id.strip():
            raise ModelArtifactError("architecture_id must be non-empty.")
        cfg_json = canonical_json(dict(self.config))
        ts = tuple(sorted(self.tensors, key=lambda t: t.name))
        by_name = {t.name: t for t in ts}
        if not ts or len(by_name) != len(ts):
            raise ModelArtifactError("ModelParameterSpace requires uniquely named TensorSpace entries.")
        object.__setattr__(self, "_config_json", cfg_json)
        object.__setattr__(self, "config", MappingProxyType(parse_strict_json(cfg_json)))
        object.__setattr__(self, "tensors", ts)
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(self, "_identity_cache", domain_digest("vld.model-parameter-space.v1", self.to_dict()))

    @property
    def identity(self) -> str:
        return self._identity_cache

    def tensor(self, name: str) -> TensorSpace:
        found = self._by_name.get(name)
        if found is None:
            raise ModelArtifactError(f"Tensor {name!r} not declared in {self.architecture_id!r}.")
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "config": parse_strict_json(self._config_json),
            "tensors": [t.to_dict() for t in self.tensors],
        }


@dataclass(frozen=True, slots=True)
class ModelWeightSnapshot:
    """Content-addressed model weight snapshot backed lazily by M1 `PagedObjectStore` manifests."""

    space: ModelParameterSpace
    manifests: Mapping[str, PagedByteManifest]

    def __post_init__(self) -> None:
        expected_names = {t.name for t in self.space.tensors}
        if set(self.manifests.keys()) != expected_names:
            raise ModelArtifactError("ModelWeightSnapshot manifests must cover exact declared tensors.")
        clean: dict[str, PagedByteManifest] = {}
        for t in self.space.tensors:
            m = self.manifests[t.name]
            if not isinstance(m, PagedByteManifest) or m.size_bytes != t.byte_size:
                raise ModelArtifactError(f"Manifest byte size mismatch for {t.name!r}.")
            clean[t.name] = m
        object.__setattr__(self, "manifests", MappingProxyType(clean))

    @classmethod
    def publish_tensors(
        cls,
        space: ModelParameterSpace,
        store: PagedObjectStore,
        raw_tensors: Mapping[str, bytes],
    ) -> ModelWeightSnapshot:
        if set(raw_tensors.keys()) != {t.name for t in space.tensors}:
            raise ModelArtifactError("raw_tensors keyset disagrees with ModelParameterSpace.")
        manifests: dict[str, PagedByteManifest] = {}
        for t in space.tensors:
            raw = raw_tensors[t.name]
            if not isinstance(raw, bytes) or len(raw) != t.byte_size:
                raise ModelArtifactError(f"Raw byte length mismatch for {t.name!r}: expected {t.byte_size}.")
            manifests[t.name] = store.put_bytes(raw)
        return cls(space=space, manifests=manifests)

    @property
    def identity(self) -> str:
        """Storage-geometry-independent parameter point digest."""
        return domain_digest(
            "vld.model-weight-point.v1",
            {
                "space": self.space.identity,
                "tensors": [[t.name, self.manifests[t.name].content_digest] for t in self.space.tensors],
            },
        )

    def read_tensor_bytes(
        self,
        name: str,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
    ) -> bytes:
        """Materialize a single tensor's verified bytes lazily from M1 store or remote HTTP Range fetcher."""
        t = self.space.tensor(name)
        m = self.manifests[name]
        if store is not None:
            raw = store.read_bytes(m)
        elif remote_range_fetcher is not None:
            raw = remote_range_fetcher(m.manifest_digest, 0, m.size_bytes)
            if hashlib.sha256(raw).hexdigest() != m.content_digest:
                raise ModelArtifactError(f"Remote tensor {name!r} failed content_digest verification.")
        else:
            raise ModelArtifactError("Either local M1 store or remote_range_fetcher is required.")
        if len(raw) != t.byte_size:
            raise ModelArtifactError(f"Materialized byte length mismatch for {name!r}.")
        return raw

    def read_tensor_slice_bytes(
        self,
        name: str,
        byte_offset: int,
        byte_length: int,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
    ) -> bytes:
        """Materialize a verified sub-range `[byte_offset, byte_offset + byte_length)` of tensor `name`."""
        t = self.space.tensor(name)
        _strict_nonneg_int(byte_offset, "byte_offset")
        _strict_nonneg_int(byte_length, "byte_length")
        if byte_offset + byte_length > t.byte_size:
            raise ModelArtifactError(f"Tensor slice [{byte_offset}, {byte_offset + byte_length}) exceeds {name!r} byte_size {t.byte_size}.")
        if byte_length == 0:
            return b""
        m = self.manifests[name]
        if store is not None:
            return store.read_range(m, byte_offset, byte_length)
        if remote_range_fetcher is not None:
            raw = remote_range_fetcher(m.manifest_digest, byte_offset, byte_length)
            if len(raw) != byte_length:
                raise ModelArtifactError(f"Remote slice length mismatch for {name!r}.")
            return raw
        raise ModelArtifactError("Either local M1 store or remote_range_fetcher is required.")


PUBLISHED_QWEN3_PROFILES: dict[str, dict[str, Any]] = {
    "qwen3-0.6b": {
        "source_model": "Qwen/Qwen3-0.6B",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "dtype": "BF16",
    },
    "qwen3-30b-a3b": {
        "source_model": "Qwen/Qwen3-30B-A3B",
        "revision": "e187836637046dedaf8a1f27c91bf65c5175cc97",
        "model_type": "qwen3_moe",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 768,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "dtype": "BF16",
    },
    "qwen3-235b-a22b": {
        "source_model": "Qwen/Qwen3-235B-A22B",
        "revision": "58ccb6e3fb22606f13e07de68b71f8fbce40a375",
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "num_hidden_layers": 94,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 1536,
        "vocab_size": 151936,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "dtype": "BF16",
    },
    "deepseek-r1-distill-qwen-1.5b": {
        "source_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "revision": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
        "model_type": "qwen2",
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "max_position_embeddings": 131072,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "attention_bias": True,
        "dtype": "BF16",
    },
    "smollm2-135m-instruct": {
        "source_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
        "model_type": "llama",
        "hidden_size": 576,
        "num_hidden_layers": 30,
        "num_attention_heads": 9,
        "num_key_value_heads": 3,
        "head_dim": 64,
        "intermediate_size": 1536,
        "vocab_size": 49152,
        "max_position_embeddings": 8192,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100000.0,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "dtype": "BF16",
    },
    "glm-z1-9b-0414": {
        "source_model": "THUDM/GLM-Z1-9B-0414",
        "revision": "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
        "model_type": "glm4",
        "hidden_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 13696,
        "vocab_size": 151552,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "attention_bias": True,
        "partial_rotary_factor": 0.5,
        "dtype": "BF16",
    },
}


@dataclass(frozen=True, slots=True)
class TransformerArchitectureSpec:
    """Lazy architectural Ground supporting `qwen3`, `qwen3_moe`, `qwen2` (`DeepSeek-R1-Distill`), `llama` (`SmolLM2`), and `glm4` (`GLM-Z1`)."""

    source_model: str
    revision: str
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    intermediate_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    norm_topk_prob: bool = True
    tie_word_embeddings: bool = False
    dtype: str = "BF16"
    attention_bias: bool | None = None
    partial_rotary_factor: float = 1.0

    def __post_init__(self) -> None:
        if not _REPO_RE.fullmatch(self.source_model):
            raise ModelArtifactError(f"source_model must match 'owner/name', got {self.source_model!r}.")
        if not _HEX40_RE.fullmatch(self.revision):
            raise ModelArtifactError("revision must be a pinned 40-char lowercase Git SHA.")
        if self.model_type not in ("qwen3", "qwen3_moe", "qwen2", "llama", "glm4"):
            raise ModelArtifactError(f"Unsupported model_type {self.model_type!r}.")
        for name, val in (
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ):
            _strict_pos_int(val, name)
        if self.num_attention_heads % self.num_key_value_heads != 0 or self.head_dim % 2 != 0:
            raise ModelArtifactError("Invalid GQA head grouping or odd RoPE head_dim.")
        if not (0.0 < self.rms_norm_eps < 1.0) or not math.isfinite(self.rope_theta) or self.rope_theta <= 0.0:
            raise ModelArtifactError("Invalid rms_norm_eps or rope_theta.")
        if not math.isfinite(self.partial_rotary_factor) or not (0.0 < self.partial_rotary_factor <= 1.0):
            raise ModelArtifactError("partial_rotary_factor must lie in (0, 1].")
        rot_dim = int(round(self.head_dim * self.partial_rotary_factor))
        if rot_dim <= 0 or rot_dim % 2 != 0:
            raise ModelArtifactError("rotary_dim must be a positive even integer.")
        if self.dtype not in ("BF16", "F16", "F32"):
            raise ModelArtifactError(f"Unsupported weight dtype {self.dtype!r}.")

        if self.model_type != "qwen3_moe":
            _strict_pos_int(self.intermediate_size, "intermediate_size")
        else:
            _strict_pos_int(self.num_experts, "num_experts")
            _strict_pos_int(self.num_experts_per_tok, "num_experts_per_tok")
            _strict_pos_int(self.moe_intermediate_size, "moe_intermediate_size")
            assert self.num_experts is not None and self.num_experts_per_tok is not None
            if self.num_experts_per_tok > self.num_experts:
                raise ModelArtifactError("num_experts_per_tok cannot exceed num_experts.")

    @classmethod
    def from_published_profile(cls, profile_name: str) -> TransformerArchitectureSpec:
        key = profile_name.lower()
        if key not in PUBLISHED_QWEN3_PROFILES:
            raise ModelArtifactError(f"Unknown published profile {profile_name!r}.")
        return cls(**PUBLISHED_QWEN3_PROFILES[key])

    @classmethod
    def from_huggingface_config(
        cls,
        source_model: str,
        revision: str,
        config: Mapping[str, Any],
        *,
        dtype: str | None = None,
    ) -> TransformerArchitectureSpec:
        """Lift any supported Hugging Face `config.json` (`qwen3`, `qwen3_moe`, `qwen2`, `llama`, `glm4`) into a pinned Spec."""
        mtype = str(config.get("model_type", "")).lower()
        h = int(config["hidden_size"])
        nq = int(config["num_attention_heads"])
        nkv = int(config.get("num_key_value_heads", nq))
        hd = int(config.get("head_dim") or (h // nq))
        dt_map = {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}
        raw_dt = str(config.get("torch_dtype", "bfloat16")).lower()
        resolved_dt = dtype if dtype is not None else dt_map.get(raw_dt, "BF16")
        prf = float(config.get("partial_rotary_factor") or (0.5 if mtype == "glm4" else 1.0))
        abias = config.get("attention_bias")
        return cls(
            source_model=source_model,
            revision=revision,
            model_type=mtype,
            hidden_size=h,
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=nq,
            num_key_value_heads=nkv,
            head_dim=hd,
            vocab_size=int(config["vocab_size"]),
            max_position_embeddings=int(config.get("max_position_embeddings", 8192)),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
            rope_theta=float(config.get("rope_theta", 10000.0)),
            intermediate_size=int(config["intermediate_size"]) if config.get("intermediate_size") is not None else None,
            num_experts=int(config["num_experts"]) if config.get("num_experts") is not None else None,
            num_experts_per_tok=int(config["num_experts_per_tok"]) if config.get("num_experts_per_tok") is not None else None,
            moe_intermediate_size=int(config["moe_intermediate_size"]) if config.get("moe_intermediate_size") is not None else None,
            norm_topk_prob=bool(config.get("norm_topk_prob", True)),
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
            dtype=resolved_dt,
            attention_bias=None if abias is None else bool(abias),
            partial_rotary_factor=prf,
        )

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_moe"

    @property
    def has_qk_norm(self) -> bool:
        return self.model_type in ("qwen3", "qwen3_moe")

    @property
    def has_qkv_bias(self) -> bool:
        if self.attention_bias is not None:
            return bool(self.attention_bias)
        return self.model_type in ("qwen2", "glm4")

    @property
    def has_sandwich_norms(self) -> bool:
        return self.model_type == "glm4"

    @property
    def has_fused_gate_up(self) -> bool:
        return self.model_type == "glm4"

    @property
    def rotary_dim(self) -> int:
        return int(round(self.head_dim * self.partial_rotary_factor))

    @property
    def has_interleaved_rope(self) -> bool:
        return self.model_type == "glm4"

    @property
    def fingerprint(self) -> str:
        return domain_digest("vld.transformer-arch.v1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        base = {
            "dtype": self.dtype,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "model_type": self.model_type,
            "moe_intermediate_size": self.moe_intermediate_size,
            "norm_topk_prob": self.norm_topk_prob,
            "num_attention_heads": self.num_attention_heads,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "revision": self.revision,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "source_model": self.source_model,
            "tie_word_embeddings": self.tie_word_embeddings,
            "vocab_size": self.vocab_size,
        }
        if self.model_type not in ("qwen3", "qwen3_moe"):
            base["attention_bias"] = self.has_qkv_bias
            base["partial_rotary_factor"] = self.partial_rotary_factor
        return base

    def parameter_inventory(self) -> dict[str, int]:
        """Compute exact declared and per-token active parameter counts without allocating weights."""
        h = self.hidden_size
        nq = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim
        layers = self.num_hidden_layers
        emb = self.vocab_size * h
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h
        qk_norm_elems = (2 * hd) if self.has_qk_norm else 0
        qkv_bias_elems = (nq * hd + 2 * nkv * hd) if self.has_qkv_bias else 0
        attn_per_layer = h * (2 * nq * hd + 2 * nkv * hd) + qk_norm_elems + qkv_bias_elems
        norms_per_layer = (4 * h) if self.has_sandwich_norms else (2 * h)
        if not self.is_moe:
            assert self.intermediate_size is not None
            ffn_total = 3 * h * self.intermediate_size
            ffn_active = ffn_total
            router = 0
        else:
            assert self.num_experts is not None and self.num_experts_per_tok is not None and self.moe_intermediate_size is not None
            router = self.num_experts * h
            ffn_total = self.num_experts * 3 * h * self.moe_intermediate_size
            ffn_active = self.num_experts_per_tok * 3 * h * self.moe_intermediate_size

        total = emb + lm_head + h + layers * (attn_per_layer + norms_per_layer + router + ffn_total)
        active = emb + lm_head + h + layers * (attn_per_layer + norms_per_layer + router + ffn_active)
        elem_bytes = SAFETENSORS_DTYPE_WIDTHS[self.dtype]
        return {
            "active_bytes_per_token_upper_bound": active * elem_bytes,
            "active_elements_per_token_upper_bound": active,
            "declared_bytes": total * elem_bytes,
            "declared_elements": total,
        }

    def layer_tensor_spaces(self, layer_index: int) -> tuple[TensorSpace, ...]:
        """Return exact `TensorSpace` descriptors for one decoder layer."""
        if not (0 <= layer_index < self.num_hidden_layers):
            raise ModelArtifactError(f"layer_index {layer_index} out of range [0, {self.num_hidden_layers}).")
        h = self.hidden_size
        nq = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim
        p = f"model.layers.{layer_index}."
        spaces = [
            TensorSpace(p + "input_layernorm.weight", self.dtype, (h,)),
            TensorSpace(p + "post_attention_layernorm.weight", self.dtype, (h,)),
            TensorSpace(p + "self_attn.q_proj.weight", self.dtype, (nq * hd, h)),
            TensorSpace(p + "self_attn.k_proj.weight", self.dtype, (nkv * hd, h)),
            TensorSpace(p + "self_attn.v_proj.weight", self.dtype, (nkv * hd, h)),
            TensorSpace(p + "self_attn.o_proj.weight", self.dtype, (h, nq * hd)),
        ]
        if self.has_qk_norm:
            spaces.extend(
                [
                    TensorSpace(p + "self_attn.q_norm.weight", self.dtype, (hd,)),
                    TensorSpace(p + "self_attn.k_norm.weight", self.dtype, (hd,)),
                ]
            )
        if self.has_qkv_bias:
            spaces.extend(
                [
                    TensorSpace(p + "self_attn.q_proj.bias", self.dtype, (nq * hd,)),
                    TensorSpace(p + "self_attn.k_proj.bias", self.dtype, (nkv * hd,)),
                    TensorSpace(p + "self_attn.v_proj.bias", self.dtype, (nkv * hd,)),
                ]
            )
        if self.has_sandwich_norms:
            spaces.extend(
                [
                    TensorSpace(p + "post_self_attn_layernorm.weight", self.dtype, (h,)),
                    TensorSpace(p + "post_mlp_layernorm.weight", self.dtype, (h,)),
                ]
            )
        if not self.is_moe:
            assert self.intermediate_size is not None
            f = self.intermediate_size
            if self.has_fused_gate_up:
                spaces.extend(
                    [
                        TensorSpace(p + "mlp.gate_up_proj.weight", self.dtype, (2 * f, h)),
                        TensorSpace(p + "mlp.down_proj.weight", self.dtype, (h, f)),
                    ]
                )
            else:
                spaces.extend(
                    [
                        TensorSpace(p + "mlp.gate_proj.weight", self.dtype, (f, h)),
                        TensorSpace(p + "mlp.up_proj.weight", self.dtype, (f, h)),
                        TensorSpace(p + "mlp.down_proj.weight", self.dtype, (h, f)),
                    ]
                )
        else:
            assert self.num_experts is not None and self.moe_intermediate_size is not None
            mid = self.moe_intermediate_size
            spaces.append(TensorSpace(p + "mlp.gate.weight", self.dtype, (self.num_experts, h)))
            for exp_idx in range(self.num_experts):
                ep = f"{p}mlp.experts.{exp_idx}."
                spaces.extend(
                    [
                        TensorSpace(ep + "gate_proj.weight", self.dtype, (mid, h)),
                        TensorSpace(ep + "up_proj.weight", self.dtype, (mid, h)),
                        TensorSpace(ep + "down_proj.weight", self.dtype, (h, mid)),
                    ]
                )
        return tuple(sorted(spaces, key=lambda t: t.name))

    def build_parameter_space(self) -> ModelParameterSpace:
        """Build full `ModelParameterSpace` ( suitable for compact/slice models; 235B uses `layer_tensor_spaces` lazily)."""
        h = self.hidden_size
        spaces: list[TensorSpace] = [
            TensorSpace("model.embed_tokens.weight", self.dtype, (self.vocab_size, h)),
            TensorSpace("model.norm.weight", self.dtype, (h,)),
        ]
        if not self.tie_word_embeddings:
            spaces.append(TensorSpace("lm_head.weight", self.dtype, (self.vocab_size, h)))
        for l_idx in range(self.num_hidden_layers):
            spaces.extend(self.layer_tensor_spaces(l_idx))
        return ModelParameterSpace(
            architecture_id=f"vld.{self.source_model.replace('/', '.')}",
            config=self.to_dict(),
            tensors=tuple(spaces),
        )


# ============================================================================
# 4. Zero-Dependency Hugging Face `tokenizer.json` Ground & HTTP 206 Range Source
# ============================================================================


def _gpt2_bytes_to_unicode() -> dict[int, str]:
    """Exact 256-byte printable Unicode bijection used by GPT-2 / Qwen2 / Qwen3 ByteLevel BPE."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_TO_UNICODE = _gpt2_bytes_to_unicode()
_UNICODE_TO_BYTE = {ch: b for b, ch in _BYTE_TO_UNICODE.items()}
_CONTRACTIONS_3 = frozenset({"'re", "'ve", "'ll"})
_CONTRACTIONS_2 = frozenset({"'s", "'t", "'m", "'d"})


def _pretokenize_unicode_bpe(text: str, *, split_digits: bool = True) -> list[str]:
    r"""Pure-stdlib Unicode-category scanner matching Qwen2/Qwen3 (`split_digits=True`) & GPT-2 (`False`) regex."""
    n = len(text)
    i = 0
    chunks: list[str] = []

    def _is_l(ch: str) -> bool:
        return unicodedata.category(ch).startswith("L")

    def _is_n(ch: str) -> bool:
        return unicodedata.category(ch).startswith("N")

    def _is_ws(ch: str) -> bool:
        return ch.isspace()

    def _is_nl(ch: str) -> bool:
        return ch in ("\r", "\n")

    while i < n:
        ch = text[i]
        # 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch == "'":
            if i + 3 <= n and text[i : i + 3].lower() in _CONTRACTIONS_3:
                chunks.append(text[i : i + 3])
                i += 3
                continue
            if i + 2 <= n and text[i : i + 2].lower() in _CONTRACTIONS_2:
                chunks.append(text[i : i + 2])
                i += 2
                continue

        # 2. [^\r\n\p{L}\p{N}]?\p{L}+ (for Qwen2/3) or ` ?\p{L}+`
        if _is_l(ch):
            j = i + 1
            while j < n and _is_l(text[j]):
                j += 1
            chunks.append(text[i:j])
            i = j
            continue
        if split_digits and (not _is_nl(ch) and not _is_n(ch)) and i + 1 < n and _is_l(text[i + 1]):
            j = i + 2
            while j < n and _is_l(text[j]):
                j += 1
            chunks.append(text[i:j])
            i = j
            continue
        if (not split_digits) and ch == " " and i + 1 < n and _is_l(text[i + 1]):
            j = i + 2
            while j < n and _is_l(text[j]):
                j += 1
            chunks.append(text[i:j])
            i = j
            continue

        # 3. \p{N} (Qwen2/3) vs ` ?\p{N}+` (GPT-2)
        if split_digits and _is_n(ch):
            chunks.append(ch)
            i += 1
            continue
        if not split_digits:
            if _is_n(ch) or (ch == " " and i + 1 < n and _is_n(text[i + 1])):
                j = i + (2 if ch == " " else 1)
                while j < n and _is_n(text[j]):
                    j += 1
                chunks.append(text[i:j])
                i = j
                continue

        # 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        if (not _is_ws(ch) and not _is_l(ch) and not _is_n(ch)) or (
            ch == " " and i + 1 < n and (not _is_ws(text[i + 1]) and not _is_l(text[i + 1]) and not _is_n(text[i + 1]))
        ):
            j = i + (2 if ch == " " else 1)
            while j < n and (not _is_ws(text[j]) and not _is_l(text[j]) and not _is_n(text[j])):
                j += 1
            if split_digits:
                while j < n and _is_nl(text[j]):
                    j += 1
            chunks.append(text[i:j])
            i = j
            continue

        # 5. \s*[\r\n]+ | \s+(?!\S) | \s+
        w_end = i + 1
        while w_end < n and _is_ws(text[w_end]):
            w_end += 1
        if split_digits:
            last_nl = -1
            for k in range(i, w_end):
                if _is_nl(text[k]):
                    last_nl = k
            if last_nl != -1:
                chunks.append(text[i : last_nl + 1])
                i = last_nl + 1
                continue
        if w_end == n:
            chunks.append(text[i:w_end])
            i = w_end
        elif w_end - i >= 2:
            chunks.append(text[i : w_end - 1])
            i = w_end - 1
        else:
            chunks.append(text[i:w_end])
            i = w_end
    return chunks


class HuggingFaceBPETokenizerGround:
    """Zero-dependency deterministic BPE Tokenizer Ground loading Hugging Face `tokenizer.json`."""

    def __init__(
        self,
        raw_config: Mapping[str, Any],
        *,
        tokenizer_id: str = "vld.hf.tokenizer",
        source_sha256: str = "",
    ) -> None:
        if not isinstance(raw_config, Mapping) or "model" not in raw_config:
            raise ModelArtifactError("Invalid Hugging Face tokenizer.json structure.")
        model_cfg = raw_config["model"]
        if model_cfg.get("type") not in (None, "BPE"):
            raise ModelArtifactError(f"Unsupported HF tokenizer model type {model_cfg.get('type')!r}.")

        self.tokenizer_id = tokenizer_id
        vocab_raw = model_cfg["vocab"]
        self._vocab: dict[str, int] = {str(k): int(v) for k, v in vocab_raw.items()}
        self._id_to_token: dict[int, str] = {v: k for k, v in self._vocab.items()}
        self._ignore_merges = bool(model_cfg.get("ignore_merges", False))

        norm_cfg = raw_config.get("normalizer")
        self._norm_form: str | None = None
        if isinstance(norm_cfg, Mapping) and norm_cfg.get("type") in ("NFC", "NFKC", "NFD", "NFKD"):
            self._norm_form = str(norm_cfg["type"])

        pre_str = json.dumps(raw_config.get("pre_tokenizer", {}))
        self._split_digits = ("\\p{N}+" not in pre_str)

        merge_ranks: dict[tuple[str, str], int] = {}
        for rank, entry in enumerate(model_cfg.get("merges", ())):
            if isinstance(entry, str):
                parts = entry.split(" ", 1)
                if len(parts) != 2:
                    raise ModelArtifactError(f"Malformed BPE merge string {entry!r}.")
                pair = (parts[0], parts[1])
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                pair = (str(entry[0]), str(entry[1]))
            else:
                raise ModelArtifactError(f"Malformed BPE merge entry {entry!r}.")
            merge_ranks.setdefault(pair, rank)
        self._merge_ranks = merge_ranks

        added_all_map: dict[str, int] = {}
        added_nonspecial_map: dict[str, int] = {}
        added_id_to_token: dict[int, str] = {}
        special_ids: dict[int, str] = {}
        for item in raw_config.get("added_tokens", ()):
            if not isinstance(item, Mapping):
                continue
            tid = int(item["id"])
            content = str(item["content"])
            is_special = bool(item.get("special", False))
            added_all_map[content] = tid
            added_id_to_token[tid] = content
            if is_special:
                special_ids[tid] = content
            else:
                added_nonspecial_map[content] = tid
        self._added_all_map = added_all_map
        self._added_nonspecial_map = added_nonspecial_map
        self._added_id_to_token = added_id_to_token
        self._special_ids = special_ids
        # Sort added tokens longest-first for deterministic greedy matching
        self._added_all_sorted = tuple(sorted(added_all_map.keys(), key=lambda s: (-len(s), s)))
        self._added_nonspecial_sorted = tuple(sorted(added_nonspecial_map.keys(), key=lambda s: (-len(s), s)))
        self.vocab_size = max(max(self._id_to_token.keys(), default=-1), max(added_id_to_token.keys(), default=-1)) + 1
        self._bpe_cache: dict[str, tuple[int, ...]] = {}
        self.source_sha256 = source_sha256 or domain_digest(
            "vld.hf-tokenizer-config.v1",
            {
                "added_count": len(added_all_map),
                "merges_count": len(merge_ranks),
                "norm": self._norm_form,
                "special_count": len(special_ids),
                "tokenizer_id": tokenizer_id,
                "vocab_count": len(self._vocab),
            },
        )

    @property
    def digest(self) -> str:
        return domain_digest(
            "vld.hf-bpe-ground.v1",
            {
                "source_sha256": self.source_sha256,
                "tokenizer_id": self.tokenizer_id,
                "vocab_size": self.vocab_size,
            },
        )

    @classmethod
    def from_json_bytes(cls, raw_bytes: bytes, *, tokenizer_id: str = "vld.hf.tokenizer") -> HuggingFaceBPETokenizerGround:
        src_sha = hashlib.sha256(raw_bytes).hexdigest()
        cfg = json.loads(raw_bytes.decode("utf-8"))
        return cls(cfg, tokenizer_id=tokenizer_id, source_sha256=src_sha)

    @classmethod
    def from_file(cls, path: Path | str, *, tokenizer_id: str | None = None) -> HuggingFaceBPETokenizerGround:
        p = Path(path)
        raw = p.read_bytes()
        return cls.from_json_bytes(raw, tokenizer_id=tokenizer_id or f"vld.hf.{p.name}")

    @classmethod
    def from_huggingface_hub(
        cls,
        repo_id: str,
        *,
        revision: str,
        filename: str = "tokenizer.json",
        timeout_sec: float = 30.0,
    ) -> HuggingFaceBPETokenizerGround:
        if not _REPO_RE.fullmatch(repo_id) or not _HEX40_RE.fullmatch(revision):
            raise ModelArtifactError("from_huggingface_hub requires 'owner/name' and 40-char Git revision SHA.")
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        req = urllib.request.Request(url, headers={"User-Agent": "vld-hf-observer/0.4"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read()
        return cls.from_json_bytes(raw, tokenizer_id=f"vld.hf.{repo_id}@{revision[:12]}")

    def _bpe_word(self, encoded_word: str) -> tuple[int, ...]:
        cached = self._bpe_cache.get(encoded_word)
        if cached is not None:
            return cached
        if self._ignore_merges and encoded_word in self._vocab:
            res = (self._vocab[encoded_word],)
            self._bpe_cache[encoded_word] = res
            return res
        word = list(encoded_word)
        while len(word) >= 2:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                r = self._merge_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pair = pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            nxt: list[str] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    nxt.append(merged)
                    i += 2
                else:
                    nxt.append(word[i])
                    i += 1
            word = nxt
        out_ids = tuple(self._vocab[tok] for tok in word)
        self._bpe_cache[encoded_word] = out_ids
        return out_ids

    def _encode_ordinary_span(self, span: str) -> list[int]:
        if not span:
            return []
        normed = unicodedata.normalize(self._norm_form, span) if self._norm_form else span
        ids: list[int] = []
        for piece in _pretokenize_unicode_bpe(normed, split_digits=self._split_digits):
            raw_bytes = piece.encode("utf-8", errors="strict")
            byte_str = "".join(_BYTE_TO_UNICODE[b] for b in raw_bytes)
            ids.extend(self._bpe_word(byte_str))
        return ids

    def encode(self, text: str, *, allow_special: bool = True) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise ModelArtifactError("encode requires a str.")
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ModelArtifactError(f"Non-UTF8 string rejected: {exc}") from exc

        active_sorted = self._added_all_sorted if allow_special else self._added_nonspecial_sorted
        active_map = self._added_all_map if allow_special else self._added_nonspecial_map
        if not active_sorted:
            return tuple(self._encode_ordinary_span(text))

        out: list[int] = []
        pos = 0
        n = len(text)
        while pos < n:
            best_idx = -1
            best_tok = ""
            for st in active_sorted:
                idx = text.find(st, pos)
                if idx != -1 and (best_idx == -1 or idx < best_idx):
                    best_idx = idx
                    best_tok = st
                    if best_idx == pos:
                        break
            if best_idx == -1:
                out.extend(self._encode_ordinary_span(text[pos:]))
                break
            if best_idx > pos:
                out.extend(self._encode_ordinary_span(text[pos:best_idx]))
            out.append(active_map[best_tok])
            pos = best_idx + len(best_tok)
        return tuple(out)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        byte_buf = bytearray()

        def _flush_bytes() -> None:
            if byte_buf:
                parts.append(byte_buf.decode("utf-8", errors="replace"))
                byte_buf.clear()

        for tid in token_ids:
            if isinstance(tid, bool) or not isinstance(tid, int):
                raise ModelArtifactError(f"Invalid token ID {tid!r}.")
            if tid in self._added_id_to_token:
                _flush_bytes()
                if not (skip_special_tokens and tid in self._special_ids):
                    parts.append(self._added_id_to_token[tid])
            elif tid in self._id_to_token:
                tok_str = self._id_to_token[tid]
                for ch in tok_str:
                    b_val = _UNICODE_TO_BYTE.get(ch)
                    if b_val is None:
                        _flush_bytes()
                        parts.append(ch)
                    else:
                        byte_buf.append(b_val)
            else:
                raise ModelArtifactError(f"Unknown token ID {tid} for {self.tokenizer_id!r}.")
        _flush_bytes()
        return "".join(parts)

    def encode_with_receipt(self, text: str) -> TokenizerRoundTripReceipt:
        tokens = self.encode(text)
        recovered = self.decode(tokens)
        expected = unicodedata.normalize(self._norm_form, text) if self._norm_form else text
        if recovered != expected:
            raise ModelArtifactError("HuggingFaceBPETokenizerGround round-trip conservation check failed.")
        raw = expected.encode("utf-8")
        src_sha = hashlib.sha256(raw).hexdigest()
        rec_digest = domain_digest(
            "vld.tokenizer-receipt.v1",
            {
                "byte_length": len(raw),
                "source_utf8_sha256": src_sha,
                "token_ids": list(tokens),
                "tokenizer_digest": self.digest,
            },
        )
        return TokenizerRoundTripReceipt(
            tokenizer_digest=self.digest,
            source_utf8_sha256=src_sha,
            byte_length=len(raw),
            token_ids=tokens,
            receipt_digest=rec_digest,
        )


class HttpRangeSource:
    """Zero-dependency HTTP/1.1 `206 Partial Content` Range reader with keep-alive pooling & telemetry."""

    def __init__(
        self,
        url: str,
        *,
        total_size_bytes: int | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_sec: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ModelArtifactError(f"HttpRangeSource requires an http(s) URL, got {url!r}.")
        self.url = url
        origin_parts = urllib.parse.urlsplit(self.url)
        self._origin_key = (origin_parts.scheme.lower(), origin_parts.netloc.lower())
        self._resolved_url: str | None = None
        self._conn: http.client.HTTPConnection | None = None
        self._conn_host_key: tuple[str, str, int] | None = None
        self._lock = threading.RLock()
        self._total_size: int | None = total_size_bytes
        self._headers = {"User-Agent": "vld-hfr-range/0.4", "Connection": "keep-alive", **dict(headers or {})}
        self._timeout = float(timeout_sec)
        self._max_retries = max(1, int(max_retries))
        self.requests_made = 0
        self.bytes_transferred = 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    @property
    def total_size_bytes(self) -> int:
        if self._total_size is None:
            self._probe_size()
        assert self._total_size is not None
        return self._total_size

    def _headers_for_target(self, scheme: str, netloc: str, range_hdr: str) -> dict[str, str]:
        same_origin = (scheme.lower(), netloc.lower()) == self._origin_key
        out = {
            k: v
            for k, v in self._headers.items()
            if same_origin or k.lower() != "authorization"
        }
        out["Host"] = netloc
        out["Range"] = range_hdr
        return out

    def _probe_size(self) -> None:
        with self._lock:
            target = self.url
            for _ in range(6):
                status, body, loc, resp_headers = self._fetch_via_keepalive(target, "bytes=0-7", return_headers=True)
                if status in (301, 302, 303, 307, 308) and loc:
                    target = urllib.parse.urljoin(target, loc)
                    continue
                self._resolved_url = target
                self.requests_made += 1
                self.bytes_transferred += len(body)
                cr = resp_headers.get("content-range", "")
                if "/" in cr:
                    self._total_size = int(cr.split("/")[-1].strip())
                    return
                if status == 200:
                    cl = resp_headers.get("content-length")
                    self._total_size = int(cl) if cl is not None else len(body)
                    return
                raise ModelArtifactError(f"Server did not advertise Content-Range total size for {self.url!r} (HTTP {status}).")
            raise ModelArtifactError(f"Too many HTTP redirects probing {self.url!r}.")

    def _fetch_via_keepalive(
        self,
        target_url: str,
        range_hdr: str,
        *,
        return_headers: bool = False,
    ) -> Any:
        parts = urllib.parse.urlsplit(target_url)
        scheme = parts.scheme.lower()
        host = parts.hostname or ""
        port = parts.port or (443 if scheme == "https" else 80)
        path_q = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        key = (scheme, host, port)
        if self._conn is None or self._conn_host_key != key:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            self._conn = (
                http.client.HTTPSConnection(host, port, timeout=self._timeout)
                if scheme == "https"
                else http.client.HTTPConnection(host, port, timeout=self._timeout)
            )
            self._conn_host_key = key

        req_headers = self._headers_for_target(scheme, parts.netloc, range_hdr)
        self._conn.request("GET", path_q, headers=req_headers)
        resp = self._conn.getresponse()
        loc = resp.getheader("Location")
        hdrs = {k.lower(): v for k, v in resp.getheaders()} if return_headers else None
        data = resp.read()
        if return_headers:
            return resp.status, data, loc, hdrs
        return resp.status, data, loc

    def _follow_redirects_keepalive(self, initial_url: str, range_hdr: str, *, max_hops: int = 5) -> tuple[int, bytes, str]:
        cur = initial_url
        for _ in range(max_hops + 1):
            status, data, loc = self._fetch_via_keepalive(cur, range_hdr)
            if status in (301, 302, 303, 307, 308) and loc:
                cur = urllib.parse.urljoin(cur, loc)
                continue
            return status, data, cur
        raise ModelArtifactError(f"Too many HTTP redirects ({max_hops}) fetching {self.url!r}.")

    def read_range(self, offset: int, length: int) -> bytes:
        _strict_nonneg_int(offset, "offset")
        _strict_nonneg_int(length, "length")
        if length == 0:
            return b""
        end_incl = offset + length - 1
        range_hdr = f"bytes={offset}-{end_incl}"
        last_err: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                with self._lock:
                    had_cached_redirect = self._resolved_url is not None and self._resolved_url != self.url
                    target = self._resolved_url or self.url
                    status, data, final_url = self._follow_redirects_keepalive(target, range_hdr)
                    if status in (400, 401, 403) and had_cached_redirect:
                        # Presigned CDN URL expired -> re-resolve from origin URL with cross-origin auth isolation
                        self._resolved_url = None
                        self.close()
                        status, data, final_url = self._follow_redirects_keepalive(self.url, range_hdr)
                    if status not in (200, 206):
                        raise ModelArtifactError(f"HTTP status {status} on Range {range_hdr} for {self.url!r}.")
                    self._resolved_url = final_url
                    self.requests_made += 1
                    self.bytes_transferred += len(data)
                    if status == 200 and (offset > 0 or len(data) > length):
                        data = data[offset : offset + length]
                    if len(data) != length:
                        raise ModelArtifactError(
                            f"HTTP Range [{offset}, {offset + length}) returned {len(data)} bytes from {self.url!r}."
                        )
                    return data
            except Exception as exc:
                last_err = exc
                self.close()
                if attempt + 1 < self._max_retries:
                    time.sleep(0.25 * (attempt + 1))
        raise ModelArtifactError(f"HttpRangeSource failed after {self._max_retries} attempts: {last_err}") from last_err


class LazySafetensorsWeightGround:
    """Lazy single-shard or multi-shard Safetensors Weight Ground over HTTP Range or local disk (`HFR` + `G16`/`G17`).

    Observes shard headers on demand without downloading inactive experts or redundant tied `lm_head` tensors,
    and optionally caches materialized tensors into a local M1 `PagedObjectStore`.
    """

    def __init__(
        self,
        *,
        space: ModelParameterSpace,
        source_pin: str,
        shard_sources: Mapping[str, tuple[Callable[[int, int], bytes], Callable[[], int]]],
        tensor_to_shard: Mapping[str, str],
        cache_store: PagedObjectStore | None = None,
    ) -> None:
        self.space = space
        self.source_pin = source_pin
        self._shard_sources = dict(shard_sources)
        self._tensor_to_shard = dict(tensor_to_shard)
        self.cache_store = cache_store
        self._lock = threading.RLock()
        self._observed_shards: dict[str, SafetensorsShardObservation] = {}
        self._cached_manifests: dict[str, PagedByteManifest] = {}
        self.tensors_materialized = 0
        self.slice_reads_executed = 0

    @property
    def identity(self) -> str:
        return domain_digest(
            "vld.lazy-safetensors-ground.v1",
            {
                "shards": sorted(self._shard_sources.keys()),
                "source_pin": self.source_pin,
                "space": self.space.identity,
            },
        )

    def observe_shard(self, shard_name: str) -> SafetensorsShardObservation:
        with self._lock:
            obs = self._observed_shards.get(shard_name)
            if obs is not None:
                return obs
            if shard_name not in self._shard_sources:
                raise ModelArtifactError(f"Unknown Safetensors shard {shard_name!r}.")
            reader_fn, size_fn = self._shard_sources[shard_name]
            obs = observe_safetensors_header(reader_fn, size_fn(), shard_name=shard_name)
            self._observed_shards[shard_name] = obs
            return obs

    def _locate_tensor(self, name: str) -> tuple[str, SafetensorsShardObservation, int, int]:
        t = self.space.tensor(name)
        shard_name = self._tensor_to_shard.get(name)
        if shard_name is None:
            if len(self._shard_sources) == 1:
                shard_name = next(iter(self._shard_sources.keys()))
            else:
                raise ModelArtifactError(f"Tensor {name!r} missing from Safetensors shard index.")
        obs = self.observe_shard(shard_name)
        entry = obs.tensor_entry(name)
        if entry.dtype != t.dtype or entry.shape != t.shape:
            raise ModelArtifactError(
                f"Safetensors header for {name!r} ({entry.dtype}, {entry.shape}) disagrees with declared ({t.dtype}, {t.shape})."
            )
        abs_off, byte_len = obs.absolute_byte_range(name)
        return shard_name, obs, abs_off, byte_len

    def read_tensor_bytes(
        self,
        name: str,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
    ) -> bytes:
        _ = (store, remote_range_fetcher)
        with self._lock:
            if self.cache_store is not None and name in self._cached_manifests:
                return self.cache_store.read_bytes(self._cached_manifests[name])
            shard_name, _obs, abs_off, byte_len = self._locate_tensor(name)
            reader_fn, _ = self._shard_sources[shard_name]
            raw = reader_fn(abs_off, byte_len)
            if len(raw) != byte_len:
                raise ModelArtifactError(f"Short Safetensors range read for {name!r}.")
            self.tensors_materialized += 1
            if self.cache_store is not None:
                self._cached_manifests[name] = self.cache_store.put_bytes(raw)
            return raw

    def read_tensor_slice_bytes(
        self,
        name: str,
        byte_offset: int,
        byte_length: int,
        *,
        store: PagedObjectStore | None = None,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
    ) -> bytes:
        _ = (store, remote_range_fetcher)
        t = self.space.tensor(name)
        _strict_nonneg_int(byte_offset, "byte_offset")
        _strict_nonneg_int(byte_length, "byte_length")
        if byte_offset + byte_length > t.byte_size:
            raise ModelArtifactError(f"Slice [{byte_offset}, {byte_offset + byte_length}) exceeds {name!r} byte_size {t.byte_size}.")
        if byte_length == 0:
            return b""
        with self._lock:
            if self.cache_store is not None and name in self._cached_manifests:
                return self.cache_store.read_range(self._cached_manifests[name], byte_offset, byte_length)
            shard_name, _obs, abs_off, _ = self._locate_tensor(name)
            reader_fn, _ = self._shard_sources[shard_name]
            raw = reader_fn(abs_off + byte_offset, byte_length)
            if len(raw) != byte_length:
                raise ModelArtifactError(
                    f"Short Safetensors slice read for {name!r}: expected {byte_length} bytes, got {len(raw)}."
                )
            self.slice_reads_executed += 1
            return raw

    def prefetch_tensors_coalesced(
        self,
        tensor_names: Sequence[str],
        *,
        max_gap_bytes: int = 65_536,
        max_span_bytes: int = 64 * 1024 * 1024,
    ) -> int:
        """Coalesce nearby tensor extents per shard into bounded streaming HTTP `206` spans into `cache_store`."""
        if self.cache_store is None:
            raise ModelArtifactError("prefetch_tensors_coalesced requires a configured M1 cache_store.")
        with self._lock:
            by_shard: dict[str, list[tuple[int, int, str]]] = {}
            for name in tensor_names:
                if name in self._cached_manifests:
                    continue
                sname, _obs, abs_off, blen = self._locate_tensor(name)
                by_shard.setdefault(sname, []).append((abs_off, blen, name))

            spans_fetched = 0
            for sname, items in sorted(by_shard.items()):
                items.sort(key=lambda t: t[0])
                reader_fn, _ = self._shard_sources[sname]
                i = 0
                while i < len(items):
                    span_start = items[i][0]
                    span_end = items[i][0] + items[i][1]
                    group = [items[i]]
                    j = i + 1
                    while j < len(items):
                        nxt_start, nxt_len, _ = items[j]
                        if nxt_start - span_end <= max_gap_bytes and (nxt_start + nxt_len - span_start) <= max_span_bytes:
                            span_end = max(span_end, nxt_start + nxt_len)
                            group.append(items[j])
                            j += 1
                        else:
                            break
                    expected_span_len = span_end - span_start
                    raw_span = reader_fn(span_start, expected_span_len)
                    if len(raw_span) != expected_span_len:
                        raise ModelArtifactError(
                            f"Short coalesced Safetensors span read on {sname!r}: expected {expected_span_len}, got {len(raw_span)}."
                        )
                    span_buf = memoryview(raw_span)
                    spans_fetched += 1
                    for t_off, t_len, t_name in group:
                        rel = t_off - span_start
                        self._cached_manifests[t_name] = self.cache_store.put_bytes(bytes(span_buf[rel : rel + t_len]))
                        self.tensors_materialized += 1
                    i = j
            return spans_fetched

    def export_cached_manifests(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: m.to_dict() for k, m in sorted(self._cached_manifests.items())}

    def load_cached_manifests(self, manifests_by_name: Mapping[str, PagedByteManifest | Mapping[str, Any]]) -> None:
        with self._lock:
            for name, m_raw in manifests_by_name.items():
                t = self.space.tensor(name)
                m = m_raw if isinstance(m_raw, PagedByteManifest) else PagedByteManifest.from_dict(m_raw)
                if m.size_bytes != t.byte_size:
                    raise ModelArtifactError(f"Cached manifest byte_size mismatch for {name!r}.")
                self._cached_manifests[name] = m

    @classmethod
    def from_local_checkpoint(
        cls,
        path_or_dir: Path | str,
        *,
        arch: TransformerArchitectureSpec,
        cache_store: PagedObjectStore | None = None,
    ) -> LazySafetensorsWeightGround:
        """Bind a local `.safetensors` file or sharded directory via bounded `seek`/`read` Range slices."""
        target = Path(path_or_dir)
        space = arch.build_parameter_space()
        declared_names = {t.name for t in space.tensors}

        def _make_file_reader(fp: Path) -> tuple[Callable[[int, int], bytes], Callable[[], int]]:
            def _read(off: int, length: int, p: Path = fp) -> bytes:
                with open(p, "rb") as handle:
                    handle.seek(off)
                    return handle.read(length)
            return _read, (lambda p=fp: p.stat().st_size)

        if target.is_file():
            sname = target.name
            reader_fn, size_fn = _make_file_reader(target)
            obs = observe_safetensors_header(reader_fn, size_fn(), shard_name=sname)
            g = cls(
                space=space,
                source_pin=f"file://{sname}:{obs.header_sha256}",
                shard_sources={sname: (reader_fn, size_fn)},
                tensor_to_shard={tname: sname for tname in declared_names},
                cache_store=cache_store,
            )
            g._observed_shards[sname] = obs
            return g

        idx_file = target / "model.safetensors.index.json"
        if idx_file.is_file():
            idx_raw = idx_file.read_bytes()
            wmap = json.loads(idx_raw.decode("utf-8")).get("weight_map", {})
            missing = [tname for tname in sorted(declared_names) if tname not in wmap]
            if missing:
                raise ModelArtifactError(f"Declared tensors missing from {idx_file}: {missing[:5]}")
            shard_names = sorted({str(s) for k, s in wmap.items() if k in declared_names})
            for sn in shard_names:
                if not (target / sn).is_file():
                    raise ModelArtifactError(f"Referenced Safetensors shard {target / sn} does not exist.")
            return cls(
                space=space,
                source_pin=f"dir://{target.name}:{hashlib.sha256(idx_raw).hexdigest()}",
                shard_sources={sn: _make_file_reader(target / sn) for sn in shard_names},
                tensor_to_shard={tname: str(wmap[tname]) for tname in declared_names},
                cache_store=cache_store,
            )

        single = target / "model.safetensors"
        if single.is_file():
            return cls.from_local_checkpoint(single, arch=arch, cache_store=cache_store)
        raise ModelArtifactError(f"No model.safetensors or model.safetensors.index.json found at {target}.")

    @classmethod
    def from_huggingface_hub(
        cls,
        repo_id: str,
        *,
        revision: str,
        arch: TransformerArchitectureSpec,
        cache_store: PagedObjectStore | None = None,
        timeout_sec: float = 30.0,
    ) -> tuple[LazySafetensorsWeightGround, dict[str, HttpRangeSource]]:
        """Bind a live single-shard or sharded Hugging Face Hub checkpoint via HTTP `206` Range readers."""
        if not _REPO_RE.fullmatch(repo_id) or not _HEX40_RE.fullmatch(revision):
            raise ModelArtifactError("from_huggingface_hub requires 'owner/name' and 40-char Git revision SHA.")
        base = f"https://huggingface.co/{repo_id}/resolve/{revision}"
        space = arch.build_parameter_space()
        declared_names = {t.name for t in space.tensors}

        # Check whether checkpoint is sharded (`model.safetensors.index.json`) or single-file (`model.safetensors`)
        http_sources: dict[str, HttpRangeSource] = {}
        tensor_to_shard: dict[str, str] = {}
        index_sha = ""
        try:
            idx_req = urllib.request.Request(
                f"{base}/model.safetensors.index.json",
                headers={"User-Agent": "vld-hfr-range/0.4"},
            )
            with urllib.request.urlopen(idx_req, timeout=timeout_sec) as resp:
                idx_raw = resp.read()
            index_sha = hashlib.sha256(idx_raw).hexdigest()
            idx_doc = json.loads(idx_raw.decode("utf-8"))
            wmap = idx_doc["weight_map"]
            shard_names = sorted({str(s) for k, s in wmap.items() if k in declared_names})
            for sname in shard_names:
                http_sources[sname] = HttpRangeSource(f"{base}/{sname}", timeout_sec=timeout_sec)
            for tname in declared_names:
                if tname not in wmap:
                    raise ModelArtifactError(f"Declared tensor {tname!r} missing from {repo_id}/model.safetensors.index.json.")
                tensor_to_shard[tname] = str(wmap[tname])
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise ModelArtifactError(f"Failed probing {repo_id} index: {exc}") from exc
            sname = "model.safetensors"
            src = HttpRangeSource(f"{base}/{sname}", timeout_sec=timeout_sec)
            http_sources[sname] = src
            obs = observe_safetensors_header(src.read_range, src.total_size_bytes, shard_name=sname)
            index_sha = obs.header_sha256
            obs_names = {e.name for e in obs.tensors}
            missing = declared_names - obs_names
            if missing:
                raise ModelArtifactError(f"Checkpoint {repo_id} missing declared tensors: {sorted(missing)[:5]}")
            for tname in declared_names:
                tensor_to_shard[tname] = sname

        shard_Adapters: dict[str, tuple[Callable[[int, int], bytes], Callable[[], int]]] = {
            sn: (src.read_range, (lambda s=src: s.total_size_bytes))
            for sn, src in http_sources.items()
        }
        ground = cls(
            space=space,
            source_pin=f"hf://{repo_id}@{revision}:{index_sha}",
            shard_sources=shard_Adapters,
            tensor_to_shard=tensor_to_shard,
            cache_store=cache_store,
        )
        if len(http_sources) == 1 and "model.safetensors" in http_sources:
            # Reuse already-observed single-shard header so zero duplicate Range requests occur!
            ground._observed_shards["model.safetensors"] = obs  # type: ignore[possibly-undefined]
        return ground, http_sources


__all__ = [
    "ByteLevelBPETokenizerGround",
    "HttpRangeSource",
    "HuggingFaceBPETokenizerGround",
    "LazySafetensorsWeightGround",
    "MAX_SAFETENSORS_HEADER_BYTES",
    "ModelArtifactError",
    "ModelParameterSpace",
    "ModelWeightSnapshot",
    "PUBLISHED_QWEN3_PROFILES",
    "SAFETENSORS_DTYPE_WIDTHS",
    "SafetensorsShardObservation",
    "SafetensorsTensorEntry",
    "TensorSpace",
    "TokenizerRoundTripReceipt",
    "TransformerArchitectureSpec",
    "build_safetensors_bytes",
    "observe_safetensors_header",
]
