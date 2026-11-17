r"""Phase 2 — M1 Content-Addressed Paged Byte Store & Revisioned Structural Memory (`vld.memory`)."""

from __future__ import annotations

from vld.memory.certifier import (
    CertifiedLaw,
    HorizonExceededError,
    LawCertificationRefused,
    berlekamp_massey_gf2,
    certify_sequence_law,
    lfsr_jump_gf2,
    locate_gf2_bit,
    locate_int_progression,
    locate_lcg_2adic,
    locate_lcg_bsgs,
)
from vld.memory.paged_store import (
    DEFAULT_PAGE_SIZE,
    DiskPageBacking,
    GarbageCollectionReport,
    InMemoryPageBacking,
    MAX_PAGE_SIZE,
    PageBacking,
    PageChunkRef,
    PagedByteManifest,
    PagedObjectStore,
    PagedStoreError,
)
from vld.memory.registry import (
    CertificateStatus,
    StructuralMemoryCertificate,
    StructuralMemoryError,
    StructuralMemoryRegistry,
    StructuralMemorySpec,
    StructuralReadReceipt,
)

__all__ = [
    "CertificateStatus",
    "CertifiedLaw",
    "DEFAULT_PAGE_SIZE",
    "DiskPageBacking",
    "GarbageCollectionReport",
    "HorizonExceededError",
    "InMemoryPageBacking",
    "LawCertificationRefused",
    "MAX_PAGE_SIZE",
    "PageBacking",
    "PageChunkRef",
    "PagedByteManifest",
    "PagedObjectStore",
    "PagedStoreError",
    "StructuralMemoryCertificate",
    "StructuralMemoryError",
    "StructuralMemoryRegistry",
    "StructuralMemorySpec",
    "StructuralReadReceipt",
    "berlekamp_massey_gf2",
    "certify_sequence_law",
    "lfsr_jump_gf2",
    "locate_gf2_bit",
    "locate_int_progression",
    "locate_lcg_2adic",
    "locate_lcg_bsgs",
]
