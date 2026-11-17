r"""Phase 6 — Source-Pinned Model Artifacts, Safetensors Range Reader (`HFR`), BPE Tokenizer Ground & Lazy Parameter Manifolds."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import struct
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

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
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
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

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_id, str) or not self.architecture_id.strip():
            raise ModelArtifactError("architecture_id must be non-empty.")
        cfg_json = canonical_json(dict(self.config))
        ts = tuple(sorted(self.tensors, key=lambda t: t.name))
        if not ts or len({t.name for t in ts}) != len(ts):
            raise ModelArtifactError("ModelParameterSpace requires uniquely named TensorSpace entries.")
        object.__setattr__(self, "_config_json", cfg_json)
        object.__setattr__(self, "config", MappingProxyType(parse_strict_json(cfg_json)))
        object.__setattr__(self, "tensors", ts)

    @property
    def identity(self) -> str:
        return domain_digest("vld.model-parameter-space.v1", self.to_dict())

    def tensor(self, name: str) -> TensorSpace:
        for t in self.tensors:
            if t.name == name:
                return t
        raise ModelArtifactError(f"Tensor {name!r} not declared in {self.architecture_id!r}.")

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
}


@dataclass(frozen=True, slots=True)
class TransformerArchitectureSpec:
    """Lazy architectural Ground supporting dense `qwen3` (0.6B+) and sparse `qwen3_moe` (30B/235B+)."""

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

    def __post_init__(self) -> None:
        if not _REPO_RE.fullmatch(self.source_model):
            raise ModelArtifactError(f"source_model must match 'owner/name', got {self.source_model!r}.")
        if not _HEX40_RE.fullmatch(self.revision):
            raise ModelArtifactError("revision must be a pinned 40-char lowercase Git SHA.")
        if self.model_type not in ("qwen3", "qwen3_moe"):
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
        if self.dtype not in ("BF16", "F16", "F32"):
            raise ModelArtifactError(f"Unsupported weight dtype {self.dtype!r}.")

        if self.model_type == "qwen3":
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

    @property
    def is_moe(self) -> bool:
        return self.model_type == "qwen3_moe"

    @property
    def fingerprint(self) -> str:
        return domain_digest("vld.transformer-arch.v1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
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

    def parameter_inventory(self) -> dict[str, int]:
        """Compute exact declared and per-token active parameter counts without allocating weights."""
        h = self.hidden_size
        nq = self.num_attention_heads
        nkv = self.num_key_value_heads
        hd = self.head_dim
        layers = self.num_hidden_layers
        emb = self.vocab_size * h
        lm_head = 0 if self.tie_word_embeddings else self.vocab_size * h
        attn_per_layer = h * (2 * nq * hd + 2 * nkv * hd) + 2 * hd
        norms_per_layer = 2 * h
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
            TensorSpace(p + "self_attn.q_norm.weight", self.dtype, (hd,)),
            TensorSpace(p + "self_attn.k_norm.weight", self.dtype, (hd,)),
        ]
        if not self.is_moe:
            assert self.intermediate_size is not None
            f = self.intermediate_size
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


__all__ = [
    "ByteLevelBPETokenizerGround",
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
