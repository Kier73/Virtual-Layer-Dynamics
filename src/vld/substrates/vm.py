r"""Phase 5 — Deterministic VM Kernel, Copy-on-Write & Structural-Memory Virtual Pages (`vld.substrates.vm`, `G09`).

Governing equations:
1. **Composable Virtual Page Space ($P = 2^k$ bytes):**
   Each virtual page index $p = \lfloor \text{vaddr} / P \rfloor$ resolves through one of three sources:
     - **Shared Immutable / CoW Buffer:** `fork_cow()` duplicates descriptor references in $O(M_{\text{desc}})$
       with **zero** page byte copies; a write to page $p$ materializes a private `bytearray(P)` for page $p$
       alone (`Copy-on-Write` isolation).
     - **Structural-Memory Demand Page:** Page $p$ evaluates a certified `StructuralMemoryRegistry` law
       $B[k] = f_{\text{law}}(\text{base\_coord} + k) \bmod 256$ on demand in $O(P)$ when touched, consuming
       **zero** resident page memory for untouched regions.
2. **Deterministic VM Execution & M1 Snapshot Continuity:**
   `DeterministicVMKernel` steps bounded register-memory programs with deterministic tick accounting,
   capability-gated syscalls, resident page quotas, and content-addressed M1 snapshot/restore.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import canonical_bytes, domain_digest, parse_strict_json
from vld.memory import PagedByteManifest, PagedObjectStore, StructuralMemoryRegistry

PF_X = 0x1
PF_W = 0x2
PF_R = 0x4


class VMError(ValueError):
    """Base error for deterministic VM configuration, instruction decoding, or execution."""


class PageFaultError(VMError):
    """Raised on unmapped virtual address access or R/W/X page permission violation."""


class ResidentQuotaExceededError(VMError):
    """Raised when materializing a virtual page would exceed `max_resident_pages`."""


class SyscallRefusedError(PermissionError):
    """Raised when a VM process invokes an unpermitted or unknown syscall."""


@dataclass(frozen=True, slots=True)
class StructuralPageBinding:
    """Binds a virtual page to a certified `StructuralMemoryRegistry` byte law."""

    registry: StructuralMemoryRegistry
    memory_id: str
    base_coordinate: int


@dataclass
class _PageDescriptor:
    flags: int
    immutable_data: bytes | None = None
    private_data: bytearray | None = None
    structural: StructuralPageBinding | None = None

    def clone_cow(self) -> _PageDescriptor:
        # Demote any private mutable page into shared immutable bytes so parent & child share CoW!
        if self.private_data is not None:
            frozen = bytes(self.private_data)
            self.immutable_data = frozen
            self.private_data = None
            return _PageDescriptor(flags=self.flags, immutable_data=frozen, structural=self.structural)
        return _PageDescriptor(
            flags=self.flags,
            immutable_data=self.immutable_data,
            private_data=None,
            structural=self.structural,
        )


class VirtualPageSpace:
    r"""Paged virtual address space supporting R/W/X flags, $O(1)$ Copy-on-Write forks, and Structural Memory demand paging."""

    def __init__(
        self,
        page_size: int = 4096,
        *,
        max_resident_pages: int | None = None,
    ) -> None:
        if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 64 or (page_size & (page_size - 1)) != 0:
            raise VMError("page_size must be a power of two >= 64.")
        if max_resident_pages is not None and (isinstance(max_resident_pages, bool) or max_resident_pages <= 0):
            raise VMError("max_resident_pages must be positive when supplied.")
        self.page_size = page_size
        self.max_resident_pages = max_resident_pages
        self._pages: dict[int, _PageDescriptor] = {}
        self.cow_faults = 0
        self.structural_syntheses = 0

    @property
    def resident_private_pages(self) -> int:
        return sum(1 for d in self._pages.values() if d.private_data is not None)

    @property
    def mapped_page_count(self) -> int:
        return len(self._pages)

    def _check_private_quota(self, additional: int = 1) -> None:
        if self.max_resident_pages is not None:
            if self.resident_private_pages + additional > self.max_resident_pages:
                raise ResidentQuotaExceededError(
                    f"Resident private page quota ({self.max_resident_pages}) exceeded."
                )

    def map_region(
        self,
        vaddr: int,
        size: int,
        *,
        flags: int = PF_R | PF_W,
        initial_bytes: bytes = b"",
        allow_overwrite: bool = False,
    ) -> None:
        """Map `[vaddr, vaddr + size)` with R/W/X permission `flags` and optional initial bytes."""
        if vaddr < 0 or size <= 0:
            raise VMError("vaddr must be >= 0 and size > 0.")
        if len(initial_bytes) > size:
            raise VMError("initial_bytes exceeds mapped region size.")
        start_page = vaddr // self.page_size
        end_page = (vaddr + size - 1) // self.page_size
        if end_page - start_page + 1 > 262_144:
            raise VMError("Mapped region exceeds maximum page count budget (262,144 pages).")
        if not allow_overwrite:
            for p_idx in range(start_page, end_page + 1):
                if p_idx in self._pages:
                    raise PageFaultError(f"Overlapping virtual page mapping at page {p_idx:#x}.")

        zero_page = bytes(self.page_size)
        for p_idx in range(start_page, end_page + 1):
            self._pages[p_idx] = _PageDescriptor(flags=flags, immutable_data=zero_page)

        if initial_bytes:
            self._write_raw_unchecked(vaddr, initial_bytes)
            # Freeze initial staged bytes as shared immutable pages so CoW starts clean!
            for p_idx in range(start_page, end_page + 1):
                desc = self._pages[p_idx]
                if desc.private_data is not None:
                    desc.immutable_data = bytes(desc.private_data)
                    desc.private_data = None

    def map_structural_region(
        self,
        vaddr: int,
        page_count: int,
        *,
        registry: StructuralMemoryRegistry,
        memory_id: str,
        base_coordinate: int = 0,
        flags: int = PF_R,
    ) -> None:
        """Map `page_count` demand-synthesized Structural Memory pages with zero eager byte allocation."""
        if vaddr < 0 or vaddr % self.page_size != 0 or page_count <= 0:
            raise VMError("vaddr must be page-aligned >= 0 and page_count > 0.")
        start_page = vaddr // self.page_size
        for i in range(page_count):
            p_idx = start_page + i
            if p_idx in self._pages:
                raise PageFaultError(f"Virtual page {p_idx:#x} is already mapped.")
            self._pages[p_idx] = _PageDescriptor(
                flags=flags,
                structural=StructuralPageBinding(
                    registry=registry,
                    memory_id=memory_id,
                    base_coordinate=base_coordinate + i * self.page_size,
                ),
            )

    def fork_cow(self) -> VirtualPageSpace:
        r"""Create an $O(1)$-byte Copy-on-Write child address space sharing all immutable/structural pages."""
        child = VirtualPageSpace(self.page_size, max_resident_pages=self.max_resident_pages)
        for p_idx, desc in self._pages.items():
            child._pages[p_idx] = desc.clone_cow()
        return child

    def _resolve_read_page(self, p_idx: int, *, require_exec: bool = False) -> bytes | bytearray:
        desc = self._pages.get(p_idx)
        if desc is None:
            raise PageFaultError(f"Unmapped virtual page {p_idx:#x}.")
        req_flag = PF_X if require_exec else PF_R
        if not (desc.flags & req_flag):
            raise PageFaultError(f"Permission fault on page {p_idx:#x} (flags={desc.flags:#x}).")
        if desc.private_data is not None:
            return desc.private_data
        if desc.immutable_data is not None:
            return desc.immutable_data
        if desc.structural is not None:
            vals, _ = desc.structural.registry.read_range(
                desc.structural.memory_id,
                desc.structural.base_coordinate,
                self.page_size,
            )
            self.structural_syntheses += 1
            return bytes(int(v) & 0xFF for v in vals)
        raise PageFaultError(f"Corrupt descriptor on page {p_idx:#x}.")

    def _resolve_write_page(self, p_idx: int) -> bytearray:
        desc = self._pages.get(p_idx)
        if desc is None:
            raise PageFaultError(f"Unmapped virtual page {p_idx:#x}.")
        if not (desc.flags & PF_W):
            raise PageFaultError(f"Write permission fault on read-only page {p_idx:#x}.")
        if desc.private_data is not None:
            return desc.private_data

        # Copy-on-Write materialization of single page!
        self._check_private_quota(1)
        if desc.immutable_data is not None:
            desc.private_data = bytearray(desc.immutable_data)
            desc.immutable_data = None
        elif desc.structural is not None:
            vals, _ = desc.structural.registry.read_range(
                desc.structural.memory_id,
                desc.structural.base_coordinate,
                self.page_size,
            )
            self.structural_syntheses += 1
            desc.private_data = bytearray(int(v) & 0xFF for v in vals)
        else:
            desc.private_data = bytearray(self.page_size)
        self.cow_faults += 1
        return desc.private_data

    def _write_raw_unchecked(self, vaddr: int, data: bytes) -> None:
        pos = 0
        n = len(data)
        while pos < n:
            cur = vaddr + pos
            p_idx = cur // self.page_size
            off = cur % self.page_size
            step = min(self.page_size - off, n - pos)
            desc = self._pages[p_idx]
            if desc.private_data is None:
                base = desc.immutable_data if desc.immutable_data is not None else bytes(self.page_size)
                desc.private_data = bytearray(base)
                desc.immutable_data = None
            desc.private_data[off : off + step] = data[pos : pos + step]
            pos += step

    def read_bytes(self, vaddr: int, length: int, *, executable: bool = False) -> bytes:
        if vaddr < 0 or length < 0:
            raise PageFaultError("Negative virtual address or length.")
        if length == 0:
            return b""
        out = bytearray()
        pos = 0
        while pos < length:
            cur = vaddr + pos
            p_idx = cur // self.page_size
            off = cur % self.page_size
            step = min(self.page_size - off, length - pos)
            buf = self._resolve_read_page(p_idx, require_exec=executable)
            out.extend(buf[off : off + step])
            pos += step
        return bytes(out)

    def write_bytes(self, vaddr: int, data: bytes) -> None:
        if vaddr < 0:
            raise PageFaultError("Negative virtual address.")
        n = len(data)
        if n == 0:
            return
        # Pre-validate permissions and resident CoW quota across ALL touched pages for strict atomicity!
        first_p = vaddr // self.page_size
        last_p = (vaddr + n - 1) // self.page_size
        needed_cow = 0
        for p_idx in range(first_p, last_p + 1):
            desc = self._pages.get(p_idx)
            if desc is None:
                raise PageFaultError(f"Unmapped virtual page {p_idx:#x}.")
            if not (desc.flags & PF_W):
                raise PageFaultError(f"Write permission fault on read-only page {p_idx:#x}.")
            if desc.private_data is None:
                needed_cow += 1
        if needed_cow > 0:
            self._check_private_quota(needed_cow)

        pos = 0
        while pos < n:
            cur = vaddr + pos
            p_idx = cur // self.page_size
            off = cur % self.page_size
            step = min(self.page_size - off, n - pos)
            buf = self._resolve_write_page(p_idx)
            buf[off : off + step] = data[pos : pos + step]
            pos += step

    def read_u64_le(self, vaddr: int) -> int:
        return struct.unpack("<Q", self.read_bytes(vaddr, 8))[0]

    def write_u64_le(self, vaddr: int, value: int) -> None:
        self.write_bytes(vaddr, struct.pack("<Q", value & 0xFFFFFFFFFFFFFFFF))

    def content_digest(self) -> str:
        """Compute canonical digest of all mapped page permissions and byte contents."""
        entries: list[dict[str, Any]] = []
        for p_idx in sorted(self._pages.keys()):
            desc = self._pages[p_idx]
            raw = bytes(self._resolve_read_page(p_idx, require_exec=bool(desc.flags & PF_X) and not bool(desc.flags & PF_R)))
            entries.append(
                {
                    "flags": desc.flags,
                    "page_index": p_idx,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        return domain_digest("vld.virtual-page-space.v1", {"page_size": self.page_size, "pages": entries})


@dataclass(frozen=True, slots=True)
class VMInstruction:
    """Typed deterministic VM instruction."""

    op: str
    args: tuple[int | str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"args": list(self.args), "op": self.op}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> VMInstruction:
        return cls(op=str(data["op"]), args=tuple(data["args"]))


@dataclass
class DeterministicVMProcess:
    """Process state inside `DeterministicVMKernel`."""

    pid: int
    name: str
    program: tuple[VMInstruction, ...]
    memory: VirtualPageSpace
    capabilities: frozenset[str]
    registers: dict[str, int]
    pc: int = 0
    ticks: int = 0
    halted: bool = False
    stdout_events: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.stdout_events is None:
            self.stdout_events = []
        for i in range(16):
            self.registers.setdefault(f"r{i}", 0)

    @property
    def state_digest(self) -> str:
        return domain_digest(
            "vld.vm-process-state.v1",
            {
                "capabilities": sorted(self.capabilities),
                "halted": self.halted,
                "memory_digest": self.memory.content_digest(),
                "name": self.name,
                "pc": self.pc,
                "pid": self.pid,
                "program": [ins.to_dict() for ins in self.program],
                "registers": {k: int(v) & 0xFFFFFFFFFFFFFFFF for k, v in sorted(self.registers.items())},
                "stdout_events": list(self.stdout_events or []),
                "ticks": self.ticks,
            },
        )


class DeterministicVMKernel:
    """Deterministic register + paged-memory VM kernel with M1 snapshot/restore and capability syscalls."""

    MASK64 = 0xFFFFFFFFFFFFFFFF

    def __init__(
        self,
        syscall_handlers: Mapping[str, Callable[[DeterministicVMProcess, tuple[Any, ...]], int]] | None = None,
    ) -> None:
        self._processes: dict[int, DeterministicVMProcess] = {}
        self._next_pid = 1
        self._syscalls = dict(syscall_handlers or {})

    def spawn(
        self,
        name: str,
        program: Sequence[VMInstruction],
        *,
        memory: VirtualPageSpace | None = None,
        capabilities: Iterable_or_Set = frozenset({"emit"}),
        initial_registers: Mapping[str, int] | None = None,
    ) -> DeterministicVMProcess:
        pid = self._next_pid
        self._next_pid += 1
        mem = memory if memory is not None else VirtualPageSpace()
        if mem.mapped_page_count == 0:
            mem.map_region(0, mem.page_size * 2, flags=PF_R | PF_W)
        regs = {f"r{i}": 0 for i in range(16)}
        if initial_registers:
            for k, v in initial_registers.items():
                regs[str(k)] = int(v) & self.MASK64
        proc = DeterministicVMProcess(
            pid=pid,
            name=name,
            program=tuple(program),
            memory=mem,
            capabilities=frozenset(capabilities),
            registers=regs,
        )
        self._processes[pid] = proc
        return proc

    def fork_process_cow(self, parent_pid: int, *, child_name: str | None = None) -> DeterministicVMProcess:
        parent = self._processes[parent_pid]
        pid = self._next_pid
        self._next_pid += 1
        child = DeterministicVMProcess(
            pid=pid,
            name=child_name or f"{parent.name}.cow",
            program=parent.program,
            memory=parent.memory.fork_cow(),
            capabilities=parent.capabilities,
            registers=dict(parent.registers),
            pc=parent.pc,
            ticks=parent.ticks,
            halted=parent.halted,
            stdout_events=list(parent.stdout_events or []),
        )
        self._processes[pid] = child
        return child

    def step(self, proc: DeterministicVMProcess) -> bool:
        """Execute one deterministic instruction on `proc`. Returns False if already halted."""
        if proc.halted:
            return False
        if not (0 <= proc.pc < len(proc.program)):
            raise VMError(f"Program counter {proc.pc} out of bounds [0, {len(proc.program)}).")

        ins = proc.program[proc.pc]
        op = ins.op
        a = ins.args
        R = proc.registers
        M = self.MASK64
        next_pc = proc.pc + 1

        if op == "HALT":
            proc.halted = True
            next_pc = proc.pc
        elif op == "MOV_IMM":
            R[str(a[0])] = int(a[1]) & M
        elif op == "MOV_REG":
            R[str(a[0])] = R[str(a[1])] & M
        elif op == "ADD":
            R[str(a[0])] = (R[str(a[1])] + R[str(a[2])]) & M
        elif op == "SUB":
            R[str(a[0])] = (R[str(a[1])] - R[str(a[2])]) & M
        elif op == "MUL":
            R[str(a[0])] = (R[str(a[1])] * R[str(a[2])]) & M
        elif op == "MOD":
            divisor = R[str(a[2])]
            if divisor == 0:
                raise VMError("Division/modulo by zero in VM instruction MOD.")
            R[str(a[0])] = (R[str(a[1])] % divisor) & M
        elif op == "AND":
            R[str(a[0])] = (R[str(a[1])] & R[str(a[2])]) & M
        elif op == "XOR":
            R[str(a[0])] = (R[str(a[1])] ^ R[str(a[2])]) & M
        elif op == "SHR":
            R[str(a[0])] = (R[str(a[1])] >> (R[str(a[2])] & 63)) & M
        elif op == "CMP_EQ":
            R[str(a[0])] = 1 if R[str(a[1])] == R[str(a[2])] else 0
        elif op == "LOAD_U8":
            vaddr = R[str(a[1])] + (int(a[2]) if len(a) > 2 else 0)
            R[str(a[0])] = proc.memory.read_bytes(vaddr, 1)[0]
        elif op == "STORE_U8":
            vaddr = R[str(a[0])] + (int(a[2]) if len(a) > 2 else 0)
            proc.memory.write_bytes(vaddr, bytes([R[str(a[1])] & 0xFF]))
        elif op == "LOAD_U64":
            vaddr = R[str(a[1])] + (int(a[2]) if len(a) > 2 else 0)
            R[str(a[0])] = proc.memory.read_u64_le(vaddr)
        elif op == "STORE_U64":
            vaddr = R[str(a[0])] + (int(a[2]) if len(a) > 2 else 0)
            proc.memory.write_u64_le(vaddr, R[str(a[1])])
        elif op == "JMP":
            target = int(a[0])
            if not (0 <= target < len(proc.program)):
                raise VMError(f"JMP target {target} out of bounds.")
            next_pc = target
        elif op == "JZ":
            if R[str(a[0])] == 0:
                target = int(a[1])
                if not (0 <= target < len(proc.program)):
                    raise VMError(f"JZ target {target} out of bounds.")
                next_pc = target
        elif op == "JNZ":
            if R[str(a[0])] != 0:
                target = int(a[1])
                if not (0 <= target < len(proc.program)):
                    raise VMError(f"JNZ target {target} out of bounds.")
                next_pc = target
        elif op == "SYSCALL":
            sys_name = str(a[0])
            if sys_name not in proc.capabilities:
                raise SyscallRefusedError(f"Process {proc.pid} lacks capability for syscall {sys_name!r}.")
            if sys_name == "emit":
                val = R[str(a[1])]
                assert proc.stdout_events is not None
                proc.stdout_events.append(val)
            elif sys_name in self._syscalls:
                ret = self._syscalls[sys_name](proc, a[1:])
                R["r0"] = int(ret) & M
            else:
                raise SyscallRefusedError(f"Unknown syscall {sys_name!r}.")
        else:
            raise VMError(f"Unsupported VM opcode {op!r}.")

        proc.pc = next_pc
        proc.ticks += 1
        return True

    def run(self, proc: DeterministicVMProcess, *, max_ticks: int = 100_000) -> int:
        if max_ticks <= 0:
            raise VMError("max_ticks must be positive.")
        executed = 0
        while executed < max_ticks and not proc.halted:
            self.step(proc)
            executed += 1
        return executed

    def snapshot_process_to_m1(
        self,
        proc: DeterministicVMProcess,
        store: PagedObjectStore,
    ) -> PagedByteManifest:
        """Serialize complete process state & mapped pages into content-addressed M1 storage."""
        pages_ser: list[dict[str, Any]] = []
        for p_idx in sorted(proc.memory._pages.keys()):
            desc = proc.memory._pages[p_idx]
            raw = bytes(
                proc.memory._resolve_read_page(
                    p_idx,
                    require_exec=bool(desc.flags & PF_X) and not bool(desc.flags & PF_R),
                )
            )
            pm = store.put_bytes(raw, media_type="application/octet-stream")
            pages_ser.append(
                {
                    "flags": desc.flags,
                    "manifest": pm.to_dict(),
                    "page_index": p_idx,
                }
            )
        doc = {
            "capabilities": sorted(proc.capabilities),
            "halted": proc.halted,
            "max_resident_pages": proc.memory.max_resident_pages,
            "name": proc.name,
            "page_size": proc.memory.page_size,
            "pages": pages_ser,
            "pc": proc.pc,
            "pid": proc.pid,
            "program": [ins.to_dict() for ins in proc.program],
            "registers": {k: int(v) & self.MASK64 for k, v in sorted(proc.registers.items())},
            "schema": "vld.vm-process-snapshot.v1",
            "state_digest": proc.state_digest,
            "stdout_events": list(proc.stdout_events or []),
            "ticks": proc.ticks,
        }
        return store.put_bytes(canonical_bytes(doc), media_type="application/json")

    def restore_process_from_m1(
        self,
        snapshot_manifest: PagedByteManifest,
        store: PagedObjectStore,
    ) -> DeterministicVMProcess:
        """Rehydrate a `DeterministicVMProcess` from an M1 snapshot manifest and verify exact `state_digest`."""
        doc = parse_strict_json(store.read_bytes(snapshot_manifest))
        if not isinstance(doc, Mapping) or doc.get("schema") != "vld.vm-process-snapshot.v1":
            raise VMError("Invalid VM process snapshot schema.")
        mem = VirtualPageSpace(int(doc["page_size"]), max_resident_pages=doc["max_resident_pages"])
        for p_entry in doc["pages"]:
            pm = PagedByteManifest.from_dict(p_entry["manifest"])
            raw = store.read_bytes(pm)
            p_idx = int(p_entry["page_index"])
            mem._pages[p_idx] = _PageDescriptor(flags=int(p_entry["flags"]), immutable_data=raw)
        proc = DeterministicVMProcess(
            pid=int(doc["pid"]),
            name=str(doc["name"]),
            program=tuple(VMInstruction.from_dict(ins) for ins in doc["program"]),
            memory=mem,
            capabilities=frozenset(str(c) for c in doc["capabilities"]),
            registers={str(k): int(v) for k, v in doc["registers"].items()},
            pc=int(doc["pc"]),
            ticks=int(doc["ticks"]),
            halted=bool(doc["halted"]),
            stdout_events=list(doc["stdout_events"]),
        )
        if proc.state_digest != doc["state_digest"]:
            raise VMError("Restored VM process state_digest mismatch.")
        self._processes[proc.pid] = proc
        self._next_pid = max(self._next_pid, proc.pid + 1)
        return proc


Iterable_or_Set = Any

__all__ = [
    "DeterministicVMKernel",
    "DeterministicVMProcess",
    "PF_R",
    "PF_W",
    "PF_X",
    "PageFaultError",
    "ResidentQuotaExceededError",
    "StructuralPageBinding",
    "SyscallRefusedError",
    "VMError",
    "VMInstruction",
    "VirtualPageSpace",
]
