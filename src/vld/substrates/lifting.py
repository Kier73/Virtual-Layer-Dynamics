r"""Phase 5 — ELF64, Binary WebAssembly & Process Ground Lifting (`Locate`, `Restore`, `Continue`, `Transform`) (`vld.substrates.lifting`, `G10`).

Mathematical contract:
1. **Transactional Binary-to-Ground Lifting:**
   Raw ELF64 (`\x7fELF`) and WebAssembly 1.0 (`\x00asm\x01\x00\x00\x00`) byte images are validated for
   structural bounds, non-overlapping load segments, W^X permission policies, and structured control-flow
   well-formedness before emitting a content-addressed `LiftedProcessGround`.
2. **Canonical Process Ground Algebra (`Locate`, `Restore`, `Continue`, `Transform`):**
   - `locate_in_process`: Queries byte signatures or exported entrypoints across lifted virtual segments.
   - `restore_process`: Rehydrates a content-addressed M1 VM snapshot with digest verification.
   - `continue_process`: Steps a lifted VM or WASM process deterministically under fuel bounds.
   - `transform_process`: Applies a verified state/program morphism preserving parent-child lineage digests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import domain_digest
from vld.ground import VirtualGroundSpec
from vld.memory import PagedByteManifest, PagedObjectStore
from vld.substrates.vm import (
    DeterministicVMKernel,
    DeterministicVMProcess,
    PF_R,
    PF_W,
    PF_X,
    VMError,
    VMInstruction,
    VirtualPageSpace,
)

ELF_MAGIC = b"\x7fELF"
PT_LOAD = 1
ET_EXEC = 2
ET_DYN = 3
ELF_MACHINES = {62: "x86_64", 183: "AArch64", 243: "RISC-V"}

WASM_MAGIC = b"\x00asm\x01\x00\x00\x00"


class LiftingError(ValueError):
    """Raised when an ELF64 image, WASM binary module, or process morphism fails validation."""


# ============================================================================
# 1. Strict ELF64 Binary Parser, Builder & VirtualPageSpace Lifter
# ============================================================================


@dataclass(frozen=True, slots=True)
class ElfSegmentDescriptor:
    """Validated ELF64 `PT_LOAD` program header segment."""

    vaddr: int
    file_offset: int
    file_size: int
    mem_size: int
    flags: int
    sha256: str


@dataclass(frozen=True, slots=True)
class LiftedProcessGround:
    """Versioned Ground representation of a lifted ELF64, WASM, or VM process image."""

    format_kind: str  # "elf64", "wasm", or "vm_bytecode"
    source_sha256: str
    entry_point: int | str
    architecture: str
    ground_spec: VirtualGroundSpec
    segments: tuple[ElfSegmentDescriptor, ...] = ()
    exports: Mapping[str, Any] = ()  # type: ignore[assignment]
    parent_lineage_digest: str | None = None

    @property
    def lineage_digest(self) -> str:
        return domain_digest(
            "vld.lifted-process-ground.v1",
            {
                "architecture": self.architecture,
                "entry_point": self.entry_point,
                "exports": dict(self.exports) if isinstance(self.exports, Mapping) else {},
                "format_kind": self.format_kind,
                "ground_fingerprint": self.ground_spec.fingerprint,
                "parent_lineage_digest": self.parent_lineage_digest,
                "segments": [
                    {
                        "file_offset": s.file_offset,
                        "file_size": s.file_size,
                        "flags": s.flags,
                        "mem_size": s.mem_size,
                        "sha256": s.sha256,
                        "vaddr": s.vaddr,
                    }
                    for s in self.segments
                ],
                "source_sha256": self.source_sha256,
            },
        )


def build_minimal_elf64_image(
    segments: Sequence[tuple[int, int, bytes, int]],
    *,
    entry: int = 0x401000,
    e_type: int = ET_EXEC,
    machine: int = 62,
) -> bytes:
    """Construct a valid little-endian ELF64 binary image from `(vaddr, flags, data, mem_size)` tuples."""
    ehdr_size = 64
    phent_size = 56
    phnum = len(segments)
    data_cursor = ehdr_size + phnum * phent_size
    # Pad cursor to page alignment 0x1000
    if data_cursor % 4096 != 0:
        data_cursor = ((data_cursor // 4096) + 1) * 4096

    phdrs = bytearray()
    payloads = bytearray()
    cur_off = data_cursor
    for vaddr, flags, raw, mem_sz in segments:
        if mem_sz < len(raw):
            raise LiftingError("mem_size must be >= len(raw).")
        pad = (4096 - ((cur_off - (vaddr % 4096)) % 4096)) % 4096
        payloads.extend(b"\x00" * pad)
        cur_off += pad
        phdrs.extend(
            struct.pack(
                "<IIQQQQQQ",
                PT_LOAD,
                flags,
                cur_off,
                vaddr,
                vaddr,
                len(raw),
                mem_sz,
                4096,
            )
        )
        payloads.extend(raw)
        cur_off += len(raw)

    e_ident = ELF_MAGIC + bytes([2, 1, 1, 0]) + bytes(8)  # ELFCLASS64, ELFDATA2LSB, EV_CURRENT
    ehdr = e_ident + struct.pack(
        "<HHIQQQIHHHHHH",
        e_type,
        machine,
        1,
        entry,
        ehdr_size,
        0,
        0,
        ehdr_size,
        phent_size,
        phnum,
        0,
        0,
        0,
    )
    header_pad = b"\x00" * (data_cursor - len(ehdr) - len(phdrs))
    return bytes(ehdr) + bytes(phdrs) + header_pad + bytes(payloads)


def lift_elf64_image(
    raw_elf: bytes,
    *,
    load_bias: int = 0,
    enforce_w_xor_x: bool = True,
    page_size: int = 4096,
    max_segment_bytes: int = 64 * 1024 * 1024,
) -> tuple[LiftedProcessGround, VirtualPageSpace]:
    """Validate an ELF64 binary image and stage its `PT_LOAD` segments into a `VirtualPageSpace`."""
    if not isinstance(raw_elf, bytes) or len(raw_elf) < 64 or raw_elf[:4] != ELF_MAGIC:
        raise LiftingError("Invalid ELF64 magic or truncated ELF header.")
    if raw_elf[4] != 2 or raw_elf[5] != 1 or raw_elf[6] != 1:
        raise LiftingError("Only little-endian 64-bit ELFv1 images are supported.")

    (
        e_type,
        e_machine,
        e_version,
        e_entry,
        e_phoff,
        _e_shoff,
        _e_flags,
        e_ehsize,
        e_phentsize,
        e_phnum,
        *_rest,
    ) = struct.unpack("<HHIQQQIHHHHHH", raw_elf[16:64])

    if e_version != 1 or e_ehsize < 64:
        raise LiftingError("Invalid ELF64 header version or ehsize.")
    if e_type not in (ET_EXEC, ET_DYN):
        raise LiftingError(f"Unsupported ELF e_type {e_type}; expected ET_EXEC or ET_DYN.")
    if e_machine not in ELF_MACHINES:
        raise LiftingError(f"Unsupported ELF machine {e_machine}.")
    if e_type == ET_EXEC and load_bias != 0:
        raise LiftingError("Non-zero load_bias is only valid for PIE (ET_DYN) ELF images.")
    if load_bias < 0:
        raise LiftingError("load_bias must be non-negative.")
    if e_phnum == 0 or e_phnum > 1024 or e_phentsize < 56:
        raise LiftingError("Invalid ELF program header count or entry size.")
    if e_phoff + e_phnum * e_phentsize > len(raw_elf):
        raise LiftingError("ELF program header table exceeds file bounds.")

    try:
        vspace = VirtualPageSpace(page_size=page_size)
    except VMError as exc:
        raise LiftingError(str(exc)) from exc
    seg_descs: list[ElfSegmentDescriptor] = []

    for idx in range(e_phnum):
        off = e_phoff + idx * e_phentsize
        p_type, p_flags, p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, _p_align = struct.unpack(
            "<IIQQQQQQ", raw_elf[off : off + 56]
        )
        if p_type != PT_LOAD:
            continue
        if p_memsz == 0:
            continue
        if p_memsz > max_segment_bytes:
            raise LiftingError(f"ELF PT_LOAD segment memsz {p_memsz} exceeds max_segment_bytes ({max_segment_bytes}).")
        if p_filesz > p_memsz or p_offset + p_filesz > len(raw_elf):
            raise LiftingError("ELF PT_LOAD segment bounds invalid.")
        if enforce_w_xor_x and (p_flags & PF_W) and (p_flags & PF_X):
            raise LiftingError("W^X policy violation: ELF segment is simultaneously writable and executable.")

        mapped_vaddr = p_vaddr + load_bias
        seg_bytes = raw_elf[p_offset : p_offset + p_filesz]
        perm = p_flags & (PF_R | PF_W | PF_X)
        try:
            vspace.map_region(mapped_vaddr, p_memsz, flags=perm, initial_bytes=seg_bytes, allow_overwrite=False)
        except VMError as exc:
            raise LiftingError(f"ELF segment mapping failed: {exc}") from exc
        seg_descs.append(
            ElfSegmentDescriptor(
                vaddr=mapped_vaddr,
                file_offset=p_offset,
                file_size=p_filesz,
                mem_size=p_memsz,
                flags=perm,
                sha256=hashlib.sha256(seg_bytes).hexdigest(),
            )
        )

    if not seg_descs:
        raise LiftingError("ELF64 image contains zero non-empty PT_LOAD segments.")

    src_sha = hashlib.sha256(raw_elf).hexdigest()
    arch_name = ELF_MACHINES[e_machine]
    entry_vaddr = e_entry + load_bias
    if not any((s.vaddr <= entry_vaddr < s.vaddr + s.mem_size) and bool(s.flags & PF_X) for s in seg_descs):
        raise LiftingError(f"ELF64 entrypoint {entry_vaddr:#x} does not reside inside an executable PT_LOAD segment.")
    spec = VirtualGroundSpec(
        ground_id=f"vld.elf64.{src_sha[:16]}",
        version="1.0.0",
        language=f"elf64.{arch_name}",
        state_domain="VirtualPageSpace",
        transitions=("locate", "restore", "continue", "transform"),
        invariants=("non_overlapping_pt_load", "w_xor_x" if enforce_w_xor_x else "rwx_declared"),
        observers=("virtual_memory_read", "entry_point"),
    )
    ground = LiftedProcessGround(
        format_kind="elf64",
        source_sha256=src_sha,
        entry_point=entry_vaddr,
        architecture=arch_name,
        ground_spec=spec,
        segments=tuple(seg_descs),
        exports={"entry": entry_vaddr},
    )
    return ground, vspace


# ============================================================================
# 2. Binary WebAssembly 1.0 Lifter & Deterministic Stack/Memory Realizer
# ============================================================================


def _encode_u_leb128(val: int) -> bytes:
    if val < 0:
        raise ValueError("Unsigned LEB128 requires non-negative integer.")
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if val:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _encode_s_leb128(val: int) -> bytes:
    out = bytearray()
    while True:
        b = val & 0x7F
        val >>= 7
        if (val == 0 and (b & 0x40) == 0) or (val == -1 and (b & 0x40) != 0):
            out.append(b)
            return bytes(out)
        out.append(b | 0x80)


def _read_u_leb128(buf: bytes, pos: int, max_bits: int = 32, limit: int | None = None) -> tuple[int, int]:
    bound = len(buf) if limit is None else min(len(buf), limit)
    result = 0
    shift = 0
    while True:
        if pos >= bound:
            raise LiftingError("Truncated unsigned LEB128 in WASM binary.")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            if result >= (1 << max_bits):
                raise LiftingError("Unsigned LEB128 exceeds bit width.")
            return result, pos
        shift += 7
        if shift >= max_bits + 7:
            raise LiftingError("Overlong unsigned LEB128 in WASM binary.")


def _read_s_leb128(buf: bytes, pos: int, bits: int = 64, limit: int | None = None) -> tuple[int, int]:
    bound = len(buf) if limit is None else min(len(buf), limit)
    result = 0
    shift = 0
    while True:
        if pos >= bound:
            raise LiftingError("Truncated signed LEB128 in WASM binary.")
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if (b & 0x80) == 0:
            if shift < bits and (b & 0x40):
                result |= -(1 << shift)
            return result, pos
        if shift >= bits + 7:
            raise LiftingError("Overlong signed LEB128 in WASM binary.")


def _to_s32(u: int) -> int:
    v = u & 0xFFFFFFFF
    return v - 0x100000000 if (v & 0x80000000) else v


def _to_s64(u: int) -> int:
    v = u & 0xFFFFFFFFFFFFFFFF
    return v - 0x10000000000000000 if (v & 0x8000000000000000) else v


def _wasm_div_s(a_s: int, b_s: int, min_signed: int) -> int:
    if b_s == 0:
        raise LiftingError("WASM trap: integer divide by zero.")
    if a_s == min_signed and b_s == -1:
        raise LiftingError("WASM trap: signed integer division overflow.")
    q = abs(a_s) // abs(b_s)
    return -q if ((a_s < 0) ^ (b_s < 0)) else q


def _wasm_rem_s(a_s: int, b_s: int) -> int:
    if b_s == 0:
        raise LiftingError("WASM trap: integer remainder by zero.")
    q = abs(a_s) // abs(b_s)
    if (a_s < 0) ^ (b_s < 0):
        q = -q
    return a_s - q * b_s


@dataclass(frozen=True, slots=True)
class WasmDecodedInstr:
    opcode: int
    imm: tuple[int, ...] = ()
    else_pc: int | None = None
    end_pc: int | None = None


@dataclass(frozen=True, slots=True)
class WasmFunctionBody:
    param_types: tuple[int, ...]
    result_types: tuple[int, ...]
    local_types: tuple[int, ...]
    instructions: tuple[WasmDecodedInstr, ...]


@dataclass(frozen=True, slots=True)
class LiftedWasmModule:
    """Validated binary WebAssembly 1.0 module lifted into Virtual Layer Ground form."""

    ground: LiftedProcessGround
    functions: tuple[WasmFunctionBody, ...]
    exports: Mapping[str, tuple[int, int]]  # name -> (kind, index)
    min_memory_pages: int
    data_segments: tuple[tuple[int, bytes], ...]


def _parse_wasm_expr(code: bytes) -> tuple[WasmDecodedInstr, ...]:
    pos = 0
    raw_ins: list[list[Any]] = []  # [opcode, imm, else_pc, end_pc]
    ctrl_stack: list[tuple[int, int]] = []  # (opcode, instr_index)

    while pos < len(code):
        op = code[pos]
        pos += 1
        idx = len(raw_ins)
        if op in (0x02, 0x03, 0x04):  # block, loop, if
            if pos >= len(code):
                raise LiftingError("Truncated WASM blocktype.")
            bt = code[pos]
            if bt not in (0x40, 0x7F, 0x7E):
                raise LiftingError(f"Unsupported WASM blocktype {bt:#04x}.")
            pos += 1
            raw_ins.append([op, (bt,), None, None])
            ctrl_stack.append((op, idx))
        elif op == 0x05:  # else
            if not ctrl_stack or ctrl_stack[-1][0] != 0x04:
                raise LiftingError("Misplaced WASM else opcode.")
            _, if_idx = ctrl_stack[-1]
            if raw_ins[if_idx][2] is not None:
                raise LiftingError("Duplicate WASM else opcode in same if block.")
            raw_ins.append([op, (), None, None])
            raw_ins[if_idx][2] = idx
        elif op == 0x0B:  # end
            raw_ins.append([op, (), None, None])
            if not ctrl_stack:
                if pos != len(code):
                    raise LiftingError("Trailing bytes after function-closing WASM end opcode.")
                break
            blk_op, blk_idx = ctrl_stack.pop()
            raw_ins[blk_idx][3] = idx
            if blk_op == 0x04:
                if raw_ins[blk_idx][2] is not None:
                    else_idx = raw_ins[blk_idx][2]
                    raw_ins[else_idx][3] = idx
                elif raw_ins[blk_idx][1][0] != 0x40:
                    raise LiftingError("Valued WASM if block requires an explicit else branch.")
        elif op == 0x0E:  # br_table
            t_cnt, pos = _read_u_leb128(code, pos, 32)
            if t_cnt > 4096:
                raise LiftingError("WASM br_table target count exceeds safety cap.")
            targets: list[int] = []
            for _ in range(t_cnt + 1):
                tv, pos = _read_u_leb128(code, pos, 32)
                targets.append(tv)
            raw_ins.append([op, tuple(targets), None, None])
        elif op in (0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22):  # br, br_if, call, local.get/set/tee
            v, pos = _read_u_leb128(code, pos, 32)
            raw_ins.append([op, (v,), None, None])
        elif op in (0x28, 0x36):  # i32.load, i32.store
            align, pos = _read_u_leb128(code, pos, 32)
            offset, pos = _read_u_leb128(code, pos, 32)
            raw_ins.append([op, (align, offset), None, None])
        elif op == 0x41:  # i32.const
            v, pos = _read_s_leb128(code, pos, 32)
            raw_ins.append([op, (v & 0xFFFFFFFF,), None, None])
        elif op == 0x42:  # i64.const
            v, pos = _read_s_leb128(code, pos, 64)
            raw_ins.append([op, (v & 0xFFFFFFFFFFFFFFFF,), None, None])
        elif op in (
            0x00, 0x01, 0x0F, 0x1A, 0x1B,
            0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,
            0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A,
            0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76,
            0x7C, 0x7D, 0x7E, 0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88,
        ):
            raw_ins.append([op, (), None, None])
        else:
            raise LiftingError(f"Unsupported WASM opcode {op:#04x}.")

    if ctrl_stack or not raw_ins or raw_ins[-1][0] != 0x0B:
        raise LiftingError("Unclosed WASM structured control block.")
    return tuple(WasmDecodedInstr(r[0], r[1], r[2], r[3]) for r in raw_ins)


def lift_wasm_module(
    raw_wasm: bytes,
    *,
    max_memory_pages: int = 256,
    max_locals_per_func: int = 4096,
) -> LiftedWasmModule:
    """Parse and validate a binary WebAssembly 1.0 module into a `LiftedWasmModule`."""
    if not isinstance(raw_wasm, bytes) or len(raw_wasm) < 8 or raw_wasm[:8] != WASM_MAGIC:
        raise LiftingError("Invalid WebAssembly binary magic or version.")

    pos = 8
    func_types: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    func_type_indices: list[int] = []
    min_mem_pages = 0
    has_memory = False
    exports: dict[str, tuple[int, int]] = {}
    code_bodies: list[tuple[tuple[int, ...], tuple[WasmDecodedInstr, ...]]] = []
    data_segs: list[tuple[int, bytes]] = []

    last_sec_id = 0
    while pos < len(raw_wasm):
        sec_id = raw_wasm[pos]
        pos += 1
        sec_len, pos = _read_u_leb128(raw_wasm, pos, 32)
        if pos + sec_len > len(raw_wasm):
            raise LiftingError("Truncated WASM section.")
        sec_end = pos + sec_len
        if sec_id != 0:
            if sec_id <= last_sec_id:
                raise LiftingError("Out-of-order or duplicate WASM section.")
            last_sec_id = sec_id

        if sec_id == 0:  # Custom section
            pos = sec_end
        elif sec_id == 1:  # Type section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            for _ in range(cnt):
                if pos >= sec_end or raw_wasm[pos] != 0x60:
                    raise LiftingError("Expected 0x60 func type tag.")
                pos += 1
                p_cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if pos + p_cnt > sec_end:
                    raise LiftingError("Truncated WASM param types.")
                params = tuple(raw_wasm[pos : pos + p_cnt])
                pos += p_cnt
                r_cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if pos + r_cnt > sec_end:
                    raise LiftingError("Truncated WASM result types.")
                results = tuple(raw_wasm[pos : pos + r_cnt])
                pos += r_cnt
                if any(vt not in (0x7F, 0x7E) for vt in (*params, *results)):
                    raise LiftingError("Only i32/i64 WASM value types are supported.")
                func_types.append((params, results))
        elif sec_id == 3:  # Function section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            for _ in range(cnt):
                t_idx, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if t_idx >= len(func_types):
                    raise LiftingError("Function references out-of-bounds type index.")
                func_type_indices.append(t_idx)
        elif sec_id == 5:  # Memory section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            if cnt > 1:
                raise LiftingError("WASM 1.0 allows at most 1 linear memory.")
            if cnt == 1:
                if pos >= sec_end:
                    raise LiftingError("Truncated WASM memory limits.")
                lim_flag = raw_wasm[pos]
                if lim_flag not in (0, 1):
                    raise LiftingError("Invalid WASM memory limit flag.")
                pos += 1
                min_mem_pages, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if min_mem_pages > max_memory_pages:
                    raise LiftingError(f"WASM min_memory_pages {min_mem_pages} exceeds max_memory_pages ({max_memory_pages}).")
                if lim_flag == 1:
                    max_p, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                    if max_p < min_mem_pages:
                        raise LiftingError("WASM memory max pages < min pages.")
                has_memory = True
        elif sec_id == 7:  # Export section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            for _ in range(cnt):
                n_len, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if pos + n_len >= sec_end:
                    raise LiftingError("Truncated WASM export entry.")
                try:
                    name = raw_wasm[pos : pos + n_len].decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise LiftingError(f"Invalid UTF-8 WASM export name: {exc}") from exc
                pos += n_len
                kind = raw_wasm[pos]
                pos += 1
                e_idx, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if name in exports:
                    raise LiftingError(f"Duplicate WASM export {name!r}.")
                exports[name] = (kind, e_idx)
        elif sec_id == 10:  # Code section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            for _ in range(cnt):
                b_size, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                b_end = pos + b_size
                if b_end > sec_end:
                    raise LiftingError("WASM function body exceeds Code section bounds.")
                loc_decls, pos = _read_u_leb128(raw_wasm, pos, 32, b_end)
                locals_flat: list[int] = []
                for _ in range(loc_decls):
                    l_cnt, pos = _read_u_leb128(raw_wasm, pos, 32, b_end)
                    if pos >= b_end:
                        raise LiftingError("Truncated WASM local declaration.")
                    l_type = raw_wasm[pos]
                    if l_type not in (0x7F, 0x7E):
                        raise LiftingError(f"Unsupported WASM local value type {l_type:#04x}.")
                    pos += 1
                    if len(locals_flat) + l_cnt > max_locals_per_func:
                        raise LiftingError(f"WASM function locals exceed max_locals_per_func ({max_locals_per_func}).")
                    locals_flat.extend([l_type] * l_cnt)
                expr_ins = _parse_wasm_expr(raw_wasm[pos:b_end])
                pos = b_end
                code_bodies.append((tuple(locals_flat), expr_ins))
        elif sec_id == 11:  # Data section
            cnt, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
            for _ in range(cnt):
                if not has_memory:
                    raise LiftingError("WASM Data section requires a declared Memory section.")
                mem_idx, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if mem_idx != 0 or pos >= sec_end or raw_wasm[pos] != 0x41:
                    raise LiftingError("Only active i32.const WASM data segments supported.")
                pos += 1
                d_off, pos = _read_s_leb128(raw_wasm, pos, 32, sec_end)
                if pos >= sec_end or raw_wasm[pos] != 0x0B:
                    raise LiftingError("Expected 0x0B end after data offset expr.")
                pos += 1
                d_len, pos = _read_u_leb128(raw_wasm, pos, 32, sec_end)
                if pos + d_len > sec_end:
                    raise LiftingError("Truncated WASM data segment payload.")
                d_bytes = raw_wasm[pos : pos + d_len]
                pos += d_len
                data_segs.append((d_off, d_bytes))
        else:
            pos = sec_end

        if pos != sec_end:
            raise LiftingError(f"Section {sec_id} byte length mismatch.")

    if len(func_type_indices) != len(code_bodies):
        raise LiftingError("WASM Function section count != Code section count.")

    funcs = tuple(
        WasmFunctionBody(
            param_types=func_types[t_idx][0],
            result_types=func_types[t_idx][1],
            local_types=locs,
            instructions=ins,
        )
        for t_idx, (locs, ins) in zip(func_type_indices, code_bodies)
    )
    for kind, e_idx in exports.values():
        if kind == 0 and e_idx >= len(funcs):
            raise LiftingError("WASM function export references out-of-bounds function index.")
    for fn in funcs:
        total_locs = len(fn.param_types) + len(fn.local_types)
        for ins in fn.instructions:
            if ins.opcode in (0x20, 0x21, 0x22) and ins.imm[0] >= total_locs:
                raise LiftingError("WASM local index out of bounds.")
            if ins.opcode == 0x10 and ins.imm[0] >= len(funcs):
                raise LiftingError("WASM call references out-of-bounds function index.")
            if ins.opcode in (0x28, 0x36) and not has_memory:
                raise LiftingError("WASM memory instruction requires a declared Memory section.")

    src_sha = hashlib.sha256(raw_wasm).hexdigest()
    entry_export = next((k for k, (kind, _) in sorted(exports.items()) if kind == 0), "<none>")
    spec = VirtualGroundSpec(
        ground_id=f"vld.wasm.{src_sha[:16]}",
        version="1.0.0",
        language="wasm.v1",
        state_domain="WasmStackAndLinearMemory",
        transitions=("invoke_export", "locate", "transform"),
        invariants=("structured_control_flow", "bounded_linear_memory"),
        observers=("return_values", "linear_memory"),
    )
    ground = LiftedProcessGround(
        format_kind="wasm",
        source_sha256=src_sha,
        entry_point=entry_export,
        architecture="wasm32",
        ground_spec=spec,
        exports={k: {"index": idx, "kind": kind} for k, (kind, idx) in sorted(exports.items())},
    )
    return LiftedWasmModule(
        ground=ground,
        functions=funcs,
        exports=exports,
        min_memory_pages=min_mem_pages,
        data_segments=tuple(data_segs),
    )


class WasmRealizer:
    """Deterministic stack + linear-memory realizer for `LiftedWasmModule`."""

    def __init__(self, module: LiftedWasmModule, *, max_fuel: int = 200_000, max_memory_pages: int = 256) -> None:
        if module.min_memory_pages > max_memory_pages:
            raise LiftingError(f"Module min_memory_pages {module.min_memory_pages} exceeds {max_memory_pages}.")
        self.module = module
        self.max_fuel = max_fuel
        self.memory = bytearray(max(1, module.min_memory_pages) * 65_536)
        for off, seg in module.data_segments:
            if off < 0 or off + len(seg) > len(self.memory):
                raise LiftingError("WASM data segment exceeds linear memory bounds.")
            self.memory[off : off + len(seg)] = seg

    def invoke(self, export_name: str, args: Sequence[int] = ()) -> tuple[int, ...]:
        exp = self.module.exports.get(export_name)
        if exp is None or exp[0] != 0:
            raise LiftingError(f"Exported WASM function {export_name!r} not found.")
        return self._call_func(exp[1], args, fuel_budget=[self.max_fuel])

    def _call_func(self, fn_idx: int, args: Sequence[int], *, fuel_budget: list[int]) -> tuple[int, ...]:
        fn = self.module.functions[fn_idx]
        if len(args) != len(fn.param_types):
            raise LiftingError(f"Expected {len(fn.param_types)} WASM args, got {len(args)}.")
        locals_arr = [
            (int(v) & (0xFFFFFFFF if pt == 0x7F else 0xFFFFFFFFFFFFFFFF))
            for v, pt in zip(args, fn.param_types)
        ] + [0] * len(fn.local_types)

        stack: list[int] = []
        # Label stack: (block_opcode, start_pc, end_pc, arity, stack_height)
        labels: list[tuple[int, int, int, int, int]] = []
        ins_list = fn.instructions
        pc = 0
        M32 = 0xFFFFFFFF
        M64 = 0xFFFFFFFFFFFFFFFF
        MIN_S32 = -(1 << 31)
        MIN_S64 = -(1 << 63)

        def _pop() -> int:
            if not stack:
                raise LiftingError("WASM operand stack underflow.")
            return stack.pop()

        def _execute_branch(depth: int) -> int | None:
            if depth > len(labels):
                raise LiftingError(f"WASM branch depth {depth} exceeds label stack height {len(labels)}.")
            if depth == len(labels):
                return None  # Branch out of implicit function body block!
            target_idx = len(labels) - 1 - depth
            blk_op, start_pc, end_pc, arity, base_h = labels[target_idx]
            if blk_op == 0x03:  # loop re-enters at start_pc (loop params = 0 in WASM 1.0 MVP)
                del stack[base_h:]
                del labels[target_idx + 1 :]
                return start_pc
            if arity > 0:
                if len(stack) - base_h < arity:
                    raise LiftingError("WASM stack underflow carrying block result on branch.")
                kept = stack[-arity:]
            else:
                kept = []
            del stack[base_h:]
            stack.extend(kept)
            del labels[target_idx:]
            return end_pc + 1

        while pc < len(ins_list):
            if fuel_budget[0] <= 0:
                raise LiftingError("WASM deterministic fuel budget exhausted.")
            fuel_budget[0] -= 1

            ins = ins_list[pc]
            op = ins.opcode

            if op == 0x00:  # unreachable
                raise LiftingError("WASM trap: unreachable executed.")
            elif op == 0x01:  # nop
                pc += 1
            elif op in (0x02, 0x03):  # block, loop
                bt = ins.imm[0]
                arity = 0 if bt == 0x40 else 1
                assert ins.end_pc is not None
                labels.append((op, pc + 1, ins.end_pc, arity, len(stack)))
                pc += 1
            elif op == 0x04:  # if
                cond = _pop()
                bt = ins.imm[0]
                arity = 0 if bt == 0x40 else 1
                assert ins.end_pc is not None
                labels.append((op, pc + 1, ins.end_pc, arity, len(stack)))
                if cond != 0:
                    pc += 1
                elif ins.else_pc is not None:
                    pc = ins.else_pc + 1
                else:
                    labels.pop()
                    pc = ins.end_pc + 1
            elif op == 0x05:  # else (reached from end of true-branch -> unwind to block result & jump past end!)
                assert ins.end_pc is not None
                _, _, _, arity, base_h = labels.pop()
                kept = stack[-arity:] if arity > 0 else []
                del stack[base_h:]
                stack.extend(kept)
                pc = ins.end_pc + 1
            elif op == 0x0B:  # end
                if labels:
                    _, _, _, arity, base_h = labels.pop()
                    kept = stack[-arity:] if arity > 0 else []
                    del stack[base_h:]
                    stack.extend(kept)
                    pc += 1
                else:
                    break
            elif op in (0x0C, 0x0D):  # br, br_if
                depth = ins.imm[0]
                if op == 0x0D and _pop() == 0:
                    pc += 1
                    continue
                next_pc = _execute_branch(depth)
                if next_pc is None:
                    break
                pc = next_pc
            elif op == 0x0E:  # br_table
                idx_u32 = _pop() & M32
                table = ins.imm
                chosen_depth = table[idx_u32] if idx_u32 < len(table) - 1 else table[-1]
                next_pc = _execute_branch(chosen_depth)
                if next_pc is None:
                    break
                pc = next_pc
            elif op == 0x0F:  # return
                break
            elif op == 0x10:  # call
                callee_idx = ins.imm[0]
                callee = self.module.functions[callee_idx]
                p_len = len(callee.param_types)
                if len(stack) < p_len:
                    raise LiftingError("WASM operand stack underflow on call.")
                call_args = stack[-p_len:] if p_len > 0 else []
                if p_len > 0:
                    del stack[-p_len:]
                rets = self._call_func(callee_idx, call_args, fuel_budget=fuel_budget)
                stack.extend(rets)
                pc += 1
            elif op == 0x1A:  # drop
                _pop()
                pc += 1
            elif op == 0x1B:  # select
                c = _pop()
                v2 = _pop()
                v1 = _pop()
                stack.append(v1 if c != 0 else v2)
                pc += 1
            elif op == 0x20:  # local.get
                stack.append(locals_arr[ins.imm[0]])
                pc += 1
            elif op == 0x21:  # local.set
                locals_arr[ins.imm[0]] = _pop()
                pc += 1
            elif op == 0x22:  # local.tee
                if not stack:
                    raise LiftingError("WASM stack underflow on local.tee.")
                locals_arr[ins.imm[0]] = stack[-1]
                pc += 1
            elif op == 0x28:  # i32.load (WASM 1.0 spec §4.4.7: effective address in Z_>=0 WITHOUT mod 2^32 wrap!)
                base_u32 = _pop() & M32
                ea = base_u32 + ins.imm[1]
                if ea + 4 > len(self.memory):
                    raise LiftingError("WASM out-of-bounds i32.load.")
                stack.append(struct.unpack("<I", self.memory[ea : ea + 4])[0])
                pc += 1
            elif op == 0x36:  # i32.store (WASM 1.0 spec §4.4.7: effective address in Z_>=0 WITHOUT mod 2^32 wrap!)
                val = _pop() & M32
                base_u32 = _pop() & M32
                ea = base_u32 + ins.imm[1]
                if ea + 4 > len(self.memory):
                    raise LiftingError("WASM out-of-bounds i32.store.")
                self.memory[ea : ea + 4] = struct.pack("<I", val)
                pc += 1
            elif op in (0x41, 0x42):  # i32.const, i64.const
                stack.append(ins.imm[0])
                pc += 1
            elif op == 0x45:  # i32.eqz
                stack.append(1 if (_pop() & M32) == 0 else 0)
                pc += 1
            elif op == 0x50:  # i64.eqz
                stack.append(1 if (_pop() & M64) == 0 else 0)
                pc += 1
            elif op in (0x46, 0x51):  # eq
                b, a = _pop(), _pop()
                m = M32 if op == 0x46 else M64
                stack.append(1 if (a & m) == (b & m) else 0)
                pc += 1
            elif op in (0x47, 0x52):  # ne
                b, a = _pop(), _pop()
                m = M32 if op == 0x47 else M64
                stack.append(1 if (a & m) != (b & m) else 0)
                pc += 1
            elif op == 0x48:  # i32.lt_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s32(a) < _to_s32(b) else 0)
                pc += 1
            elif op == 0x49:  # i32.lt_u
                b, a = _pop() & M32, _pop() & M32
                stack.append(1 if a < b else 0)
                pc += 1
            elif op == 0x4A:  # i32.gt_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s32(a) > _to_s32(b) else 0)
                pc += 1
            elif op == 0x4B:  # i32.gt_u
                b, a = _pop() & M32, _pop() & M32
                stack.append(1 if a > b else 0)
                pc += 1
            elif op == 0x4C:  # i32.le_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s32(a) <= _to_s32(b) else 0)
                pc += 1
            elif op == 0x4D:  # i32.le_u
                b, a = _pop() & M32, _pop() & M32
                stack.append(1 if a <= b else 0)
                pc += 1
            elif op == 0x4E:  # i32.ge_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s32(a) >= _to_s32(b) else 0)
                pc += 1
            elif op == 0x4F:  # i32.ge_u
                b, a = _pop() & M32, _pop() & M32
                stack.append(1 if a >= b else 0)
                pc += 1
            elif op == 0x53:  # i64.lt_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s64(a) < _to_s64(b) else 0)
                pc += 1
            elif op == 0x54:  # i64.lt_u
                b, a = _pop() & M64, _pop() & M64
                stack.append(1 if a < b else 0)
                pc += 1
            elif op == 0x55:  # i64.gt_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s64(a) > _to_s64(b) else 0)
                pc += 1
            elif op == 0x56:  # i64.gt_u
                b, a = _pop() & M64, _pop() & M64
                stack.append(1 if a > b else 0)
                pc += 1
            elif op == 0x57:  # i64.le_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s64(a) <= _to_s64(b) else 0)
                pc += 1
            elif op == 0x58:  # i64.le_u
                b, a = _pop() & M64, _pop() & M64
                stack.append(1 if a <= b else 0)
                pc += 1
            elif op == 0x59:  # i64.ge_s
                b, a = _pop(), _pop()
                stack.append(1 if _to_s64(a) >= _to_s64(b) else 0)
                pc += 1
            elif op == 0x5A:  # i64.ge_u
                b, a = _pop() & M64, _pop() & M64
                stack.append(1 if a >= b else 0)
                pc += 1
            elif op == 0x6A:  # i32.add
                b, a = _pop(), _pop()
                stack.append((a + b) & M32)
                pc += 1
            elif op == 0x6B:  # i32.sub
                b, a = _pop(), _pop()
                stack.append((a - b) & M32)
                pc += 1
            elif op == 0x6C:  # i32.mul
                b, a = _pop(), _pop()
                stack.append((a * b) & M32)
                pc += 1
            elif op == 0x6D:  # i32.div_s
                b, a = _pop(), _pop()
                stack.append(_wasm_div_s(_to_s32(a), _to_s32(b), MIN_S32) & M32)
                pc += 1
            elif op == 0x6E:  # i32.div_u
                b, a = _pop() & M32, _pop() & M32
                if b == 0:
                    raise LiftingError("WASM trap: i32.div_u by zero.")
                stack.append((a // b) & M32)
                pc += 1
            elif op == 0x6F:  # i32.rem_s
                b, a = _pop(), _pop()
                stack.append(_wasm_rem_s(_to_s32(a), _to_s32(b)) & M32)
                pc += 1
            elif op == 0x70:  # i32.rem_u
                b, a = _pop() & M32, _pop() & M32
                if b == 0:
                    raise LiftingError("WASM trap: i32.rem_u by zero.")
                stack.append((a % b) & M32)
                pc += 1
            elif op == 0x71:  # i32.and
                b, a = _pop(), _pop()
                stack.append((a & b) & M32)
                pc += 1
            elif op == 0x72:  # i32.or
                b, a = _pop(), _pop()
                stack.append((a | b) & M32)
                pc += 1
            elif op == 0x73:  # i32.xor
                b, a = _pop(), _pop()
                stack.append((a ^ b) & M32)
                pc += 1
            elif op == 0x74:  # i32.shl
                b, a = _pop(), _pop()
                stack.append((a << (b & 31)) & M32)
                pc += 1
            elif op == 0x75:  # i32.shr_s
                b, a = _pop(), _pop()
                stack.append((_to_s32(a) >> (b & 31)) & M32)
                pc += 1
            elif op == 0x76:  # i32.shr_u
                b, a = _pop(), _pop()
                stack.append(((a & M32) >> (b & 31)) & M32)
                pc += 1
            elif op == 0x7C:  # i64.add
                b, a = _pop(), _pop()
                stack.append((a + b) & M64)
                pc += 1
            elif op == 0x7D:  # i64.sub
                b, a = _pop(), _pop()
                stack.append((a - b) & M64)
                pc += 1
            elif op == 0x7E:  # i64.mul
                b, a = _pop(), _pop()
                stack.append((a * b) & M64)
                pc += 1
            elif op == 0x7F:  # i64.div_s
                b, a = _pop(), _pop()
                stack.append(_wasm_div_s(_to_s64(a), _to_s64(b), MIN_S64) & M64)
                pc += 1
            elif op == 0x80:  # i64.div_u
                b, a = _pop() & M64, _pop() & M64
                if b == 0:
                    raise LiftingError("WASM trap: i64.div_u by zero.")
                stack.append((a // b) & M64)
                pc += 1
            elif op == 0x81:  # i64.rem_s
                b, a = _pop(), _pop()
                stack.append(_wasm_rem_s(_to_s64(a), _to_s64(b)) & M64)
                pc += 1
            elif op == 0x82:  # i64.rem_u
                b, a = _pop() & M64, _pop() & M64
                if b == 0:
                    raise LiftingError("WASM trap: i64.rem_u by zero.")
                stack.append((a % b) & M64)
                pc += 1
            elif op == 0x83:  # i64.and
                b, a = _pop(), _pop()
                stack.append((a & b) & M64)
                pc += 1
            elif op == 0x84:  # i64.or
                b, a = _pop(), _pop()
                stack.append((a | b) & M64)
                pc += 1
            elif op == 0x85:  # i64.xor
                b, a = _pop(), _pop()
                stack.append((a ^ b) & M64)
                pc += 1
            elif op == 0x86:  # i64.shl
                b, a = _pop(), _pop()
                stack.append((a << (b & 63)) & M64)
                pc += 1
            elif op == 0x87:  # i64.shr_s
                b, a = _pop(), _pop()
                stack.append((_to_s64(a) >> (b & 63)) & M64)
                pc += 1
            elif op == 0x88:  # i64.shr_u
                b, a = _pop(), _pop()
                stack.append(((a & M64) >> (b & 63)) & M64)
                pc += 1
            else:
                raise LiftingError(f"Unhandled WASM opcode {op:#04x}.")

        r_len = len(fn.result_types)
        if len(stack) < r_len:
            raise LiftingError("WASM stack underflow at function return.")
        return tuple(
            (v & (M32 if rt == 0x7F else M64))
            for v, rt in zip(stack[-r_len:], fn.result_types)
        ) if r_len > 0 else ()


def assemble_wasm_affine_orbit_module() -> bytes:
    r"""Assemble a standard WebAssembly 1.0 binary module exporting `affine_jump(a, c, m, seed, n) -> i64`.

    Computes $x_n = (A x_0 + B) \bmod m$ via $O(\log n)$ binary affine monoid exponentiation in `i64`:
      locals: 0=a, 1=c, 2=m, 3=seed, 4=n, 5=p(1), 6=q(0)
    """
    def _sec(sec_id: int, payload: bytes) -> bytes:
        return bytes([sec_id]) + _encode_u_leb128(len(payload)) + payload

    # Type 0: (i64, i64, i64, i64, i64) -> (i64)
    type_sec = _sec(1, b"\x01\x60\x05\x7e\x7e\x7e\x7e\x7e\x01\x7e")
    func_sec = _sec(3, b"\x01\x00")
    mem_sec = _sec(5, b"\x01\x00\x01")
    name = b"affine_jump"
    exp_sec = _sec(7, b"\x01" + _encode_u_leb128(len(name)) + name + b"\x00\x00")

    # Body: 2 extra i64 locals (5: p, 6: q)
    ins = bytearray()
    # p = 1; q = 0
    ins += b"\x42" + _encode_s_leb128(1) + b"\x21\x05"
    ins += b"\x42" + _encode_s_leb128(0) + b"\x21\x06"
    # block $exit (0x02 0x40), loop $top (0x03 0x40)
    ins += b"\x02\x40\x03\x40"
    # if n == 0: br 1 ($exit)
    ins += b"\x20\x04\x50\x0d\x01"
    # if (n & 1) != 0: p = (p * a) % m; q = (q * a + c) % m
    ins += b"\x20\x04\x42\x01\x83\x42\x00\x52\x04\x40"
    ins += b"\x20\x05\x20\x00\x7e\x20\x02\x82\x21\x05"
    ins += b"\x20\x06\x20\x00\x7e\x20\x01\x7c\x20\x02\x82\x21\x06"
    ins += b"\x0b"
    # c = (c * (a + 1)) % m; a = (a * a) % m; n = n >> 1; br 0 ($top)
    ins += b"\x20\x01\x20\x00\x42\x01\x7c\x7e\x20\x02\x82\x21\x01"
    ins += b"\x20\x00\x20\x00\x7e\x20\x02\x82\x21\x00"
    ins += b"\x20\x04\x42\x01\x88\x21\x04"
    ins += b"\x0c\x00"
    # end loop, end block
    ins += b"\x0b\x0b"
    # return (p * seed + q) % m
    ins += b"\x20\x05\x20\x03\x7e\x20\x06\x7c\x20\x02\x82\x0b"

    locals_decl = b"\x01\x02\x7e"  # 1 group of 2 x i64
    body = locals_decl + bytes(ins)
    code_sec = _sec(10, b"\x01" + _encode_u_leb128(len(body)) + body)
    return WASM_MAGIC + type_sec + func_sec + mem_sec + exp_sec + code_sec


# ============================================================================
# 3. Canonical Process Ground Operations: Locate, Restore, Continue, Transform
# ============================================================================


def locate_in_process(
    ground: LiftedProcessGround,
    vspace: VirtualPageSpace | None,
    *,
    symbol: str | None = None,
    byte_pattern: bytes | None = None,
) -> list[int | dict[str, Any]]:
    """Locate exported entrypoints or byte signatures across a lifted process Ground (`Locate`)."""
    matches: list[int | dict[str, Any]] = []
    if symbol is not None and isinstance(ground.exports, Mapping) and symbol in ground.exports:
        matches.append(ground.exports[symbol])
    if byte_pattern is not None and vspace is not None:
        if not byte_pattern:
            raise LiftingError("byte_pattern cannot be empty.")
        for seg in ground.segments:
            if not (seg.flags & PF_R):
                continue
            buf = vspace.read_bytes(seg.vaddr, seg.mem_size)
            start = 0
            while True:
                idx = buf.find(byte_pattern, start)
                if idx == -1:
                    break
                matches.append(seg.vaddr + idx)
                start = idx + 1
    return matches


def restore_process(
    kernel: DeterministicVMKernel,
    snapshot_manifest: PagedByteManifest,
    store: PagedObjectStore,
) -> DeterministicVMProcess:
    """Restore a content-addressed M1 process snapshot into `DeterministicVMKernel` (`Restore`)."""
    return kernel.restore_process_from_m1(snapshot_manifest, store)


def continue_process(
    kernel: DeterministicVMKernel,
    proc: DeterministicVMProcess,
    *,
    max_ticks: int = 10_000,
) -> dict[str, Any]:
    """Continue execution of a lifted VM process and emit a deterministic transition receipt (`Continue`)."""
    pre_digest = proc.state_digest
    steps = kernel.run(proc, max_ticks=max_ticks)
    post_digest = proc.state_digest
    body = {
        "halted": proc.halted,
        "pid": proc.pid,
        "post_state_digest": post_digest,
        "pre_state_digest": pre_digest,
        "steps_executed": steps,
        "ticks_total": proc.ticks,
    }
    return {**body, "receipt_digest": domain_digest("vld.process-continue.v1", body)}


def transform_process(
    ground: LiftedProcessGround,
    proc: DeterministicVMProcess,
    *,
    morphism_name: str,
    transform_fn: Callable[[ tuple[VMInstruction, ...], dict[str, int] ], tuple[tuple[VMInstruction, ...], dict[str, int]]],
) -> LiftedProcessGround:
    """Apply a verified program/register morphism to a VM process, tracking parent-child Ground lineage (`Transform`)."""
    if not morphism_name.strip():
        raise LiftingError("morphism_name must be non-empty.")
    new_prog, new_regs = transform_fn(proc.program, dict(proc.registers))
    proc.program = tuple(new_prog)
    proc.registers = {k: int(v) & DeterministicVMKernel.MASK64 for k, v in new_regs.items()}
    child_spec = VirtualGroundSpec(
        ground_id=f"{ground.ground_spec.ground_id}.{morphism_name}",
        version=ground.ground_spec.version,
        language=ground.ground_spec.language,
        state_domain=ground.ground_spec.state_domain,
        transitions=ground.ground_spec.transitions,
        invariants=ground.ground_spec.invariants,
        observers=ground.ground_spec.observers,
    )
    return LiftedProcessGround(
        format_kind=ground.format_kind,
        source_sha256=proc.state_digest,
        entry_point=proc.pc,
        architecture=ground.architecture,
        ground_spec=child_spec,
        segments=ground.segments,
        exports=ground.exports,
        parent_lineage_digest=ground.lineage_digest,
    )


__all__ = [
    "ELF_MACHINES",
    "ET_DYN",
    "ET_EXEC",
    "ElfSegmentDescriptor",
    "LiftedProcessGround",
    "LiftedWasmModule",
    "LiftingError",
    "PT_LOAD",
    "WASM_MAGIC",
    "WasmDecodedInstr",
    "WasmFunctionBody",
    "WasmRealizer",
    "assemble_wasm_affine_orbit_module",
    "build_minimal_elf64_image",
    "continue_process",
    "lift_elf64_image",
    "lift_wasm_module",
    "locate_in_process",
    "restore_process",
    "transform_process",
]
