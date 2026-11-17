r"""M1 Content-Addressed Paged Byte Store (`vld.memory.paged_store`).

Governing equations:
Given an immutable byte stream $B \in (\mathbb{Z}/256\mathbb{Z})^N$ and power-of-two page size $P$:
  $$K = \left\lceil \frac{N}{P} \right\rceil, \qquad B_i = B[i P : \min(N, (i+1)P)], \qquad \widetilde{B}_i = B_i \,\|\, 0^{P - |B_i|}$$
  $$p_i = \operatorname{SHA256}(\widetilde{B}_i), \qquad d_{\text{content}} = \operatorname{SHA256}(B)$$

Range-read locality guarantee:
Reading $[a, a + L) \subseteq [0, N)$ touches exclusively pages $i \in [\lfloor a/P \rfloor, \lfloor (a+L-1)/P \rfloor]$
($0$ pages when $L = 0$), re-verifying $\operatorname{SHA256}(\widetilde{B}_i) = p_i$ on every page access.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import re
import threading
from typing import Any, BinaryIO, Mapping, Protocol, runtime_checkable

from vld.canonical import canonical_bytes, domain_digest

MANIFEST_DOMAIN = "vld.m1-manifest.v1"
DEFAULT_PAGE_SIZE = 4096
MAX_PAGE_SIZE = 65_536
MAX_OBJECT_BYTES = 1 << 40
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_MEDIA_TYPE = re.compile(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+\Z")


class PagedStoreError(ValueError):
    """Raised when an M1 page, manifest, range boundary, or content hash fails validation."""


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PagedStoreError(f"{name} must be an exact int >= {minimum}.")
    return value


def _strict_hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise PagedStoreError(f"{name} must be a lowercase 64-char SHA-256 hex digest.")
    return value


def _validate_page_size(page_size: int) -> int:
    ps = _strict_int(page_size, "page_size", minimum=64)
    if ps > MAX_PAGE_SIZE or (ps & (ps - 1)) != 0:
        raise PagedStoreError(f"page_size must be a power of two in [64, {MAX_PAGE_SIZE}].")
    return ps


@runtime_checkable
class PageBacking(Protocol):
    """Fixed-size content-addressed page backing interface."""

    @property
    def page_size(self) -> int: ...

    def put_page(self, page: bytes) -> str: ...

    def get_page(self, page_digest: str) -> bytes: ...

    def has_page(self, page_digest: str) -> bool: ...


class InMemoryPageBacking:
    """Thread-safe in-memory fixed-size page backing with read/write telemetry counters."""

    def __init__(self, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._page_size = _validate_page_size(page_size)
        self._pages: dict[str, bytes] = {}
        self._lock = threading.RLock()
        self.pages_written = 0
        self.pages_read = 0

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def stored_page_count(self) -> int:
        with self._lock:
            return len(self._pages)

    def put_page(self, page: bytes) -> str:
        if not isinstance(page, bytes) or len(page) != self._page_size:
            raise PagedStoreError(f"Page must be exact {self._page_size} bytes.")
        digest = hashlib.sha256(page).hexdigest()
        with self._lock:
            self._pages.setdefault(digest, page)
            self.pages_written += 1
        return digest

    def get_page(self, page_digest: str) -> bytes:
        digest = _strict_hex64(page_digest, "page_digest")
        with self._lock:
            raw = self._pages.get(digest)
            if raw is None:
                raise PagedStoreError(f"Missing M1 page {digest}.")
            self.pages_read += 1
        if hashlib.sha256(raw).hexdigest() != digest:
            raise PagedStoreError(f"Corrupted in-memory M1 page {digest}.")
        return raw

    def has_page(self, page_digest: str) -> bool:
        digest = _strict_hex64(page_digest, "page_digest")
        with self._lock:
            return digest in self._pages

    def list_pages(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._pages.keys()))

    def delete_page(self, page_digest: str) -> bool:
        digest = _strict_hex64(page_digest, "page_digest")
        with self._lock:
            return self._pages.pop(digest, None) is not None


class DiskPageBacking:
    """Crash-safe content-addressed filesystem page backing with atomic rename and SHA-256 read verification."""

    def __init__(self, root_dir: Path | str, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self._root = Path(root_dir)
        self._page_size = _validate_page_size(page_size)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.pages_written = 0
        self.pages_read = 0

    @property
    def page_size(self) -> int:
        return self._page_size

    def _page_path(self, digest: str) -> Path:
        return self._root / digest[:2] / f"{digest}.page"

    def put_page(self, page: bytes) -> str:
        if not isinstance(page, bytes) or len(page) != self._page_size:
            raise PagedStoreError(f"Page must be exact {self._page_size} bytes.")
        digest = hashlib.sha256(page).hexdigest()
        target = self._page_path(digest)
        with self._lock:
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(f"{digest}.{os.getpid()}.{threading.get_ident()}.tmp")
                with open(tmp, "wb") as handle:
                    handle.write(page)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, target)
            self.pages_written += 1
        return digest

    def get_page(self, page_digest: str) -> bytes:
        digest = _strict_hex64(page_digest, "page_digest")
        target = self._page_path(digest)
        with self._lock:
            if not target.is_file():
                raise PagedStoreError(f"Missing persisted M1 page {digest}.")
            raw = target.read_bytes()
            self.pages_read += 1
        if len(raw) != self._page_size:
            raise PagedStoreError(f"Persisted M1 page {digest} has wrong byte length {len(raw)}.")
        if hashlib.sha256(raw).hexdigest() != digest:
            raise PagedStoreError(f"Persisted M1 page {digest} failed SHA-256 verification.")
        return raw

    def has_page(self, page_digest: str) -> bool:
        digest = _strict_hex64(page_digest, "page_digest")
        return self._page_path(digest).is_file()

    def list_pages(self) -> tuple[str, ...]:
        with self._lock:
            found: list[str] = []
            if not self._root.is_dir():
                return ()
            for sub in self._root.iterdir():
                if not sub.is_dir() or len(sub.name) != 2:
                    continue
                for entry in sub.glob("*.page"):
                    stem = entry.stem
                    if _HEX64.fullmatch(stem):
                        found.append(stem)
            return tuple(sorted(set(found)))

    def delete_page(self, page_digest: str) -> bool:
        digest = _strict_hex64(page_digest, "page_digest")
        target = self._page_path(digest)
        with self._lock:
            if not target.is_file():
                return False
            target.unlink()
            return True


@dataclass(frozen=True, slots=True)
class PageChunkRef:
    """Single ordered page reference and logical unpadded byte count."""

    index: int
    page_digest: str
    byte_length: int

    def __post_init__(self) -> None:
        _strict_int(self.index, "index", minimum=0)
        _strict_hex64(self.page_digest, "page_digest")
        bl = _strict_int(self.byte_length, "byte_length", minimum=1)
        if bl > MAX_PAGE_SIZE:
            raise PagedStoreError("Chunk byte_length exceeds MAX_PAGE_SIZE.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "byte_length": self.byte_length,
            "index": self.index,
            "page_digest": self.page_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PageChunkRef:
        if not isinstance(data, Mapping) or set(data.keys()) != {"index", "page_digest", "byte_length"}:
            raise PagedStoreError("Invalid PageChunkRef fields.")
        return cls(index=data["index"], page_digest=data["page_digest"], byte_length=data["byte_length"])


@dataclass(frozen=True, slots=True)
class PagedByteManifest:
    """Immutable content-addressed descriptor of an M1 paged byte object."""

    media_type: str
    size_bytes: int
    page_size: int
    content_digest: str
    chunks: tuple[PageChunkRef, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.media_type, str) or _MEDIA_TYPE.fullmatch(self.media_type) is None:
            raise PagedStoreError(f"Invalid media_type: {self.media_type!r}.")
        _strict_int(self.size_bytes, "size_bytes", minimum=0)
        if self.size_bytes > MAX_OBJECT_BYTES:
            raise PagedStoreError("size_bytes exceeds MAX_OBJECT_BYTES.")
        _validate_page_size(self.page_size)
        _strict_hex64(self.content_digest, "content_digest")
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise PagedStoreError("Unsupported PagedByteManifest schema_version.")
        chunks = tuple(self.chunks)
        if any(not isinstance(c, PageChunkRef) for c in chunks):
            raise PagedStoreError("chunks must contain PageChunkRef instances.")
        expected_chunks = (self.size_bytes + self.page_size - 1) // self.page_size
        if len(chunks) != expected_chunks:
            raise PagedStoreError(f"Expected {expected_chunks} chunk(s), got {len(chunks)}.")
        for idx, ch in enumerate(chunks):
            exp_len = min(self.page_size, self.size_bytes - idx * self.page_size)
            if ch.index != idx or ch.byte_length != exp_len:
                raise PagedStoreError("Chunks must be contiguous and match size_bytes.")
        if self.size_bytes == 0 and self.content_digest != hashlib.sha256(b"").hexdigest():
            raise PagedStoreError("Empty object must carry SHA-256 of empty bytes.")
        object.__setattr__(self, "chunks", chunks)

    def body_without_digest(self) -> dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "content_digest": self.content_digest,
            "kind": MANIFEST_DOMAIN,
            "media_type": self.media_type,
            "page_size": self.page_size,
            "schema_version": self.schema_version,
            "size_bytes": self.size_bytes,
        }

    @property
    def manifest_digest(self) -> str:
        return domain_digest(MANIFEST_DOMAIN, self.body_without_digest())

    def to_dict(self) -> dict[str, Any]:
        body = self.body_without_digest()
        body["manifest_digest"] = self.manifest_digest
        return body

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PagedByteManifest:
        required = {
            "kind",
            "schema_version",
            "media_type",
            "size_bytes",
            "page_size",
            "content_digest",
            "chunks",
            "manifest_digest",
        }
        if not isinstance(data, Mapping) or set(data.keys()) != required or data.get("kind") != MANIFEST_DOMAIN:
            raise PagedStoreError("Invalid PagedByteManifest dictionary.")
        raw_chunks = data["chunks"]
        if not isinstance(raw_chunks, list):
            raise PagedStoreError("Manifest chunks must be a list.")
        manifest = cls(
            media_type=data["media_type"],
            size_bytes=data["size_bytes"],
            page_size=data["page_size"],
            content_digest=data["content_digest"],
            chunks=tuple(PageChunkRef.from_dict(item) for item in raw_chunks),
            schema_version=data["schema_version"],
        )
        if _strict_hex64(data["manifest_digest"], "manifest_digest") != manifest.manifest_digest:
            raise PagedStoreError("PagedByteManifest digest mismatch.")
        return manifest


@dataclass(frozen=True, slots=True)
class GarbageCollectionReport:
    """Verified mark-and-sweep retention report for M1 `PagedObjectStore` (`G15`)."""

    pinned_manifest_count: int
    live_page_count: int
    reclaimed_manifest_count: int
    reclaimed_page_count: int
    reclaimed_bytes: int


class PagedObjectStore:
    """Content-addressed M1 byte object store supporting streaming ingest, retention leases & $O(L/P)$ range reads."""

    def __init__(
        self,
        backing: PageBacking | None = None,
        *,
        max_object_bytes: int = MAX_OBJECT_BYTES,
    ) -> None:
        self.backing: PageBacking = backing if backing is not None else InMemoryPageBacking()
        if not isinstance(self.backing, PageBacking):
            raise TypeError("backing must satisfy PageBacking protocol.")
        _validate_page_size(self.backing.page_size)
        self.page_size = self.backing.page_size
        self.max_object_bytes = _strict_int(max_object_bytes, "max_object_bytes", minimum=1)
        self._lock = threading.RLock()
        self._manifests: dict[str, PagedByteManifest] = {}
        self._pins: dict[str, set[str]] = {}  # manifest_digest -> set of lease_ids

    def register_manifest(self, manifest: PagedByteManifest | Mapping[str, Any]) -> PagedByteManifest:
        m = manifest if isinstance(manifest, PagedByteManifest) else PagedByteManifest.from_dict(manifest)
        if m.page_size != self.page_size:
            raise PagedStoreError(f"Manifest page_size {m.page_size} != store page_size {self.page_size}.")
        with self._lock:
            self._manifests[m.manifest_digest] = m
        return m

    def get_manifest(self, manifest_digest: str) -> PagedByteManifest:
        digest = _strict_hex64(manifest_digest, "manifest_digest")
        with self._lock:
            if digest not in self._manifests:
                raise PagedStoreError(f"Manifest {digest} is not registered in this store.")
            return self._manifests[digest]

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str = "application/octet-stream",
    ) -> PagedByteManifest:
        if not isinstance(data, bytes):
            raise PagedStoreError("put_bytes requires raw bytes.")
        return self.put_stream(io.BytesIO(data), len(data), media_type=media_type)

    def put_stream(
        self,
        stream: BinaryIO,
        total_size: int,
        *,
        media_type: str = "application/octet-stream",
    ) -> PagedByteManifest:
        _strict_int(total_size, "total_size", minimum=0)
        if total_size > self.max_object_bytes:
            raise PagedStoreError("Object size exceeds max_object_bytes.")
        if not isinstance(media_type, str) or _MEDIA_TYPE.fullmatch(media_type) is None:
            raise PagedStoreError(f"Invalid media_type {media_type!r}.")

        hasher = hashlib.sha256()
        chunks: list[PageChunkRef] = []
        remaining = total_size
        idx = 0
        while remaining > 0:
            want = min(self.page_size, remaining)
            buf = bytearray()
            while len(buf) < want:
                part = stream.read(want - len(buf))
                if not isinstance(part, bytes) or len(part) == 0:
                    raise PagedStoreError("Stream ended prematurely before total_size bytes.")
                buf.extend(part)
            raw = bytes(buf)
            hasher.update(raw)
            padded = raw + b"\x00" * (self.page_size - want)
            expected_page_sha = hashlib.sha256(padded).hexdigest()
            stored_sha = self.backing.put_page(padded)
            if stored_sha != expected_page_sha:
                raise PagedStoreError(f"Backing returned mismatched digest at chunk {idx}.")
            chunks.append(PageChunkRef(index=idx, page_digest=stored_sha, byte_length=want))
            remaining -= want
            idx += 1

        extra = stream.read(1)
        if extra:
            raise PagedStoreError("Stream contains trailing bytes beyond declared total_size.")

        manifest = PagedByteManifest(
            media_type=media_type,
            size_bytes=total_size,
            page_size=self.page_size,
            content_digest=hasher.hexdigest(),
            chunks=tuple(chunks),
        )
        with self._lock:
            self._manifests[manifest.manifest_digest] = manifest
        return manifest

    def read_range(
        self,
        manifest: PagedByteManifest | Mapping[str, Any],
        offset: int,
        length: int | None = None,
    ) -> bytes:
        m = manifest if isinstance(manifest, PagedByteManifest) else PagedByteManifest.from_dict(manifest)
        if m.page_size != self.page_size:
            raise PagedStoreError(f"Manifest page_size {m.page_size} != store page_size {self.page_size}.")
        _strict_int(offset, "offset", minimum=0)
        if offset > m.size_bytes:
            raise PagedStoreError(f"offset {offset} exceeds object size {m.size_bytes}.")
        sel_len = (m.size_bytes - offset) if length is None else _strict_int(length, "length", minimum=0)
        if offset + sel_len > m.size_bytes:
            raise PagedStoreError(f"Range [{offset}, {offset + sel_len}) exceeds object size {m.size_bytes}.")
        if sel_len == 0:
            return b""

        first_chunk = offset // self.page_size
        last_chunk = (offset + sel_len - 1) // self.page_size
        out = bytearray()
        for c_idx in range(first_chunk, last_chunk + 1):
            chunk_ref = m.chunks[c_idx]
            page = self.backing.get_page(chunk_ref.page_digest)
            logical = page[: chunk_ref.byte_length]
            # Verify zero padding invariant on trailing partial page
            if chunk_ref.byte_length < self.page_size and any(b != 0 for b in page[chunk_ref.byte_length :]):
                raise PagedStoreError(f"Non-zero padding bytes detected in M1 page {chunk_ref.page_digest}.")
            chunk_start_global = c_idx * self.page_size
            slice_start = max(0, offset - chunk_start_global)
            slice_end = min(chunk_ref.byte_length, offset + sel_len - chunk_start_global)
            out.extend(logical[slice_start:slice_end])

        return bytes(out)

    def read_bytes(self, manifest: PagedByteManifest | Mapping[str, Any]) -> bytes:
        m = manifest if isinstance(manifest, PagedByteManifest) else PagedByteManifest.from_dict(manifest)
        data = self.read_range(m, 0, m.size_bytes)
        if hashlib.sha256(data).hexdigest() != m.content_digest:
            raise PagedStoreError("Reconstructed M1 payload failed full content_digest verification.")
        return data

    def verify(self, manifest: PagedByteManifest | Mapping[str, Any]) -> bool:
        try:
            self.read_bytes(manifest)
            return True
        except PagedStoreError:
            return False

    def pin_manifest(
        self,
        manifest: PagedByteManifest | Mapping[str, Any] | str,
        *,
        lease_id: str = "default",
    ) -> PagedByteManifest:
        """Pin a manifest under `lease_id` so mark-and-sweep GC retains all of its referenced M1 pages (`G15`)."""
        if not isinstance(lease_id, str) or not lease_id.strip():
            raise PagedStoreError("lease_id must be a non-empty string.")
        with self._lock:
            m = self.get_manifest(manifest) if isinstance(manifest, str) else self.register_manifest(manifest)
            for ch in m.chunks:
                if not self.backing.has_page(ch.page_digest):
                    raise PagedStoreError(
                        f"Cannot pin manifest {m.manifest_digest}: missing dependency page {ch.page_digest}."
                    )
            self._pins.setdefault(m.manifest_digest, set()).add(lease_id)
            return m

    def unpin_manifest(self, manifest_digest: str, *, lease_id: str | None = None) -> bool:
        """Release `lease_id` (or all leases when `lease_id is None`) on `manifest_digest`."""
        digest = _strict_hex64(manifest_digest, "manifest_digest")
        with self._lock:
            leases = self._pins.get(digest)
            if not leases:
                return False
            if lease_id is None:
                del self._pins[digest]
                return True
            removed = lease_id in leases
            leases.discard(lease_id)
            if not leases:
                del self._pins[digest]
            return removed

    def pinned_manifests(self) -> tuple[PagedByteManifest, ...]:
        with self._lock:
            return tuple(self._manifests[d] for d in sorted(self._pins.keys()) if d in self._manifests)

    def collect_garbage(self) -> GarbageCollectionReport:
        r"""Sweep unpinned manifests and unreachable backing pages while verifying all pinned dependencies (`G15`)."""
        if not (hasattr(self.backing, "list_pages") and hasattr(self.backing, "delete_page")):
            raise PagedStoreError("Active PageBacking does not support page enumeration/deletion for GC.")
        with self._lock:
            live_pages: set[str] = set()
            for m_digest in sorted(self._pins.keys()):
                m = self._manifests.get(m_digest)
                if m is None:
                    raise PagedStoreError(f"Pinned manifest {m_digest} missing from store catalogue.")
                for ch in m.chunks:
                    if not self.backing.has_page(ch.page_digest):
                        raise PagedStoreError(
                            f"Refusing GC: pinned manifest {m_digest} is missing live page {ch.page_digest}."
                        )
                    live_pages.add(ch.page_digest)

            unpinned = [d for d in list(self._manifests.keys()) if d not in self._pins]
            for d in unpinned:
                del self._manifests[d]

            all_pages: tuple[str, ...] = self.backing.list_pages()  # type: ignore[attr-defined]
            reclaimed_pages = 0
            for p_digest in all_pages:
                if p_digest not in live_pages:
                    if self.backing.delete_page(p_digest):  # type: ignore[attr-defined]
                        reclaimed_pages += 1

            return GarbageCollectionReport(
                pinned_manifest_count=len(self._pins),
                live_page_count=len(live_pages),
                reclaimed_manifest_count=len(unpinned),
                reclaimed_page_count=reclaimed_pages,
                reclaimed_bytes=reclaimed_pages * self.page_size,
            )
