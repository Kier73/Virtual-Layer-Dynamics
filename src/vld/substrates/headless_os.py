r"""Bootable Deterministic Virtual Operating System (`HeadlessOS`), Linux `x86_64` POSIX ABI, Structural-Memory Virtual Files & Distributed CoW Continuity (`vld.substrates.headless_os`, `G27`, `G28`).

Governing mathematical contracts:
1. **POSIX Syscall Partition (`G27`):**
   Every `x86_64` syscall invocation $\Pi(\text{pid}, n, \text{args})$ evaluates atomically to the disjoint sum:
     $$\Pi(\text{pid}, n, \text{args}) \in \operatorname{Ok}(r, 0) \;\sqcup\; \operatorname{Err}(-e, e), \qquad e \in \operatorname{Errno}_{>0},$$
   guaranteeing **zero partial state mutation** when validation, capability, address, or image admission fails.
2. **Generative Structural-Memory Virtual Files:**
   An inode bound to `(memory_id, base_coordinate, virtual_size)` synthesizes `[offset, offset + size)` on demand
   via $B[k] = f_{\text{law}}(\text{base} + \text{offset} + k) \bmod 256$ in $O(\text{size} \cdot \log(\text{offset}))$,
   allowing processes to `lseek` and `read` petabyte-scale deterministic files with $O(1)$ resident storage.
3. **Distributed Copy-on-Write Process Continuity (`G28`):**
   `fork()` shares immutable `VirtualPageSpace` descriptors in $O(1)$ page copies; `snapshot_os_to_m1()`
   deduplicates shared immutable pages by content digest in M1 storage so restored parent/child processes on a
   replacement host retain shared page identities and diverge safely upon subsequent private writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import posixpath
import struct
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.journal import HashChainedJournal
from vld.memory import PagedByteManifest, PagedObjectStore, StructuralMemoryRegistry
from vld.substrates.lifting import LiftingError, lift_elf64_image
from vld.substrates.vm import (
    DeterministicVMKernel,
    DeterministicVMProcess,
    PF_R,
    PF_W,
    PF_X,
    PageFaultError,
    ResidentQuotaExceededError,
    VMError,
    VMInstruction,
    VirtualPageSpace,
    _PageDescriptor,
)


class Errno(IntEnum):
    EPERM = 1
    ENOENT = 2
    ESRCH = 3
    EBADF = 9
    ECHILD = 10
    EAGAIN = 11
    ENOMEM = 12
    EACCES = 13
    EFAULT = 14
    EEXIST = 17
    ENOTDIR = 20
    EISDIR = 21
    EINVAL = 22
    EMFILE = 24
    EPIPE = 32
    ENOSYS = 38


# Linux x86_64 stable open/mmap/fcntl/wait constants
O_RDONLY = 0
O_WRONLY = 1
O_RDWR = 2
O_CREAT = 0x40
O_EXCL = 0x80
O_TRUNC = 0x200
O_APPEND = 0x400
O_NONBLOCK = 0x800
O_CLOEXEC = 0x80000

PROT_READ = 0x1
PROT_WRITE = 0x2
PROT_EXEC = 0x4
MAP_PRIVATE = 0x02
MAP_FIXED = 0x10
MAP_ANONYMOUS = 0x20

F_GETFD = 1
F_SETFD = 2
F_GETFL = 3
F_SETFL = 4
FD_CLOEXEC = 1
WNOHANG = 1

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2


class PosixKernelError(Exception):
    """Typed internal kernel exception carrying exact POSIX `Errno`."""

    def __init__(self, errno: Errno, message: str) -> None:
        super().__init__(message)
        self.errno = errno


@dataclass(frozen=True, slots=True)
class PosixResponse:
    """Strict POSIX return value `Ok(result, 0) | Err(-errno, errno)`."""

    result: int
    errno: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.errno == 0

    @property
    def return_value(self) -> int:
        return self.result if self.errno == 0 else -int(self.errno)

    def to_dict(self) -> dict[str, Any]:
        return {"errno": int(self.errno), "error": self.error, "ok": self.ok, "result": self.result}


def normalize_vfs_path(path: str, cwd: str = "/") -> str:
    if not isinstance(path, str) or not path or "\x00" in path:
        raise PosixKernelError(Errno.EINVAL, "Invalid empty or NUL-containing VFS path.")
    joined = path if path.startswith("/") else posixpath.join(cwd, path)
    normed = posixpath.normpath(joined)
    if not normed.startswith("/"):
        normed = "/" + normed
    return normed


@dataclass
class _VFSInode:
    ino: int
    kind: str  # "dir", "file", "structural"
    mode: int
    data: bytearray = field(default_factory=bytearray)
    memory_id: str | None = None
    base_coordinate: int = 0
    virtual_size: int = 0

    @property
    def size(self) -> int:
        if self.kind == "structural":
            return self.virtual_size
        return len(self.data)


class VirtualFileSystem:
    """In-memory POSIX VFS supporting directories, mutable files, and infinite/petabyte Structural Memory files."""

    def __init__(self, structural_memory: StructuralMemoryRegistry | None = None) -> None:
        self.structural_memory = structural_memory
        self._next_ino = 2
        self._inodes: dict[str, _VFSInode] = {
            "/": _VFSInode(ino=1, kind="dir", mode=0o755),
        }

    def _alloc_ino(self) -> int:
        ino = self._next_ino
        self._next_ino += 1
        return ino

    def mkdir_p(self, path: str, mode: int = 0o755) -> str:
        norm = normalize_vfs_path(path)
        if norm == "/":
            return "/"
        parts = [p for p in norm.split("/") if p]
        cur = ""
        for p in parts:
            cur = f"{cur}/{p}"
            existing = self._inodes.get(cur)
            if existing is None:
                self._inodes[cur] = _VFSInode(ino=self._alloc_ino(), kind="dir", mode=mode)
            elif existing.kind != "dir":
                raise PosixKernelError(Errno.ENOTDIR, f"{cur!r} is not a directory.")
        return norm

    def write_regular_file(self, path: str, content: bytes, *, mode: int = 0o644) -> str:
        norm = normalize_vfs_path(path)
        parent = posixpath.dirname(norm)
        self.mkdir_p(parent)
        existing = self._inodes.get(norm)
        if existing is not None and existing.kind == "dir":
            raise PosixKernelError(Errno.EISDIR, f"{norm!r} is a directory.")
        ino = existing.ino if existing is not None else self._alloc_ino()
        self._inodes[norm] = _VFSInode(ino=ino, kind="file", mode=mode, data=bytearray(content))
        return norm

    def register_structural_file(
        self,
        path: str,
        *,
        memory_id: str,
        virtual_size: int,
        base_coordinate: int = 0,
        mode: int = 0o444,
    ) -> str:
        if self.structural_memory is None:
            raise PosixKernelError(Errno.EINVAL, "No StructuralMemoryRegistry bound to VirtualFileSystem.")
        if virtual_size < 0 or base_coordinate < 0:
            raise PosixKernelError(Errno.EINVAL, "virtual_size and base_coordinate must be >= 0.")
        norm = normalize_vfs_path(path)
        self.mkdir_p(posixpath.dirname(norm))
        self._inodes[norm] = _VFSInode(
            ino=self._alloc_ino(),
            kind="structural",
            mode=mode,
            memory_id=memory_id,
            base_coordinate=base_coordinate,
            virtual_size=virtual_size,
        )
        return norm

    def read_slice(self, inode: _VFSInode, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0:
            raise PosixKernelError(Errno.EINVAL, "Negative read offset or size.")
        if size == 0 or offset >= inode.size:
            return b""
        actual = min(size, inode.size - offset)
        if inode.kind == "file":
            return bytes(inode.data[offset : offset + actual])
        if inode.kind == "structural":
            if self.structural_memory is None or inode.memory_id is None:
                raise PosixKernelError(Errno.EINVAL, "Unbound StructuralMemoryRegistry on structural file.")
            vals, _ = self.structural_memory.read_range(inode.memory_id, inode.base_coordinate + offset, actual)
            return bytes(int(v) & 0xFF for v in vals)
        raise PosixKernelError(Errno.EISDIR, "Cannot read bytes from directory inode.")


@dataclass
class _KernelPipe:
    pipe_id: int
    buf: bytearray = field(default_factory=bytearray)
    readers: int = 1
    writers: int = 1
    capacity: int = 65536


@dataclass
class _OpenFileDescription:
    kind: str  # "vfs", "pipe_r", "pipe_w", "stdout", "stderr", "stdin"
    flags: int
    vfs_path: str | None = None
    offset: int = 0
    pipe_id: int | None = None


@dataclass
class _FDEntry:
    desc_id: int
    cloexec: bool = False


@dataclass
class _OSProcessRecord:
    pid: int
    parent_pid: int
    cwd: str
    argv: tuple[str, ...]
    env: dict[str, str]
    vm_proc: DeterministicVMProcess
    fds: dict[int, _FDEntry]
    exited: bool = False
    exit_code: int = 0
    reaped: bool = False
    stdout_bytes: bytearray = field(default_factory=bytearray)
    stderr_bytes: bytearray = field(default_factory=bytearray)


# Supported Linux x86_64 syscall numbers
X86_64_SYSCALL_NUMBERS: dict[int, str] = {
    0: "read",
    1: "write",
    2: "open",
    3: "close",
    5: "fstat",
    8: "lseek",
    9: "mmap",
    10: "mprotect",
    11: "munmap",
    22: "pipe",
    32: "dup",
    33: "dup2",
    39: "getpid",
    57: "fork",
    59: "execve",
    60: "exit",
    61: "wait4",
    72: "fcntl",
    79: "getcwd",
    80: "chdir",
}


class HeadlessOS:
    """Booted, deterministic Virtual OS integrating POSIX `x86_64` ABI, CoW VM, VFS, Pipes & Structural Memory."""

    MMAP_BASE_START = 0x10000000

    def __init__(
        self,
        *,
        page_size: int = 4096,
        max_resident_pages: int | None = None,
        journal: HashChainedJournal | None = None,
        structural_memory: StructuralMemoryRegistry | None = None,
    ) -> None:
        self.page_size = page_size
        self.max_resident_pages = max_resident_pages
        self.journal = journal if journal is not None else HashChainedJournal()
        self.structural_memory = structural_memory if structural_memory is not None else StructuralMemoryRegistry(self.journal)
        self.vfs = VirtualFileSystem(self.structural_memory)
        self.vfs.mkdir_p("/bin")
        self.vfs.mkdir_p("/tmp")
        self.vfs.mkdir_p("/proc")
        self.vm_kernel = DeterministicVMKernel(syscall_handlers={"posix": self._vm_posix_bridge})
        self._procs: dict[int, _OSProcessRecord] = {}
        self._open_descs: dict[int, _OpenFileDescription] = {}
        self._pipes: dict[int, _KernelPipe] = {}
        self._next_desc_id = 1
        self._next_pipe_id = 1
        self._next_mmap_addr = self.MMAP_BASE_START

    def _alloc_desc(self, desc: _OpenFileDescription) -> int:
        did = self._next_desc_id
        self._next_desc_id += 1
        self._open_descs[did] = desc
        return did

    def _default_stdio_fds(self) -> dict[int, _FDEntry]:
        d0 = self._alloc_desc(_OpenFileDescription(kind="stdin", flags=O_RDONLY))
        d1 = self._alloc_desc(_OpenFileDescription(kind="stdout", flags=O_WRONLY))
        d2 = self._alloc_desc(_OpenFileDescription(kind="stderr", flags=O_WRONLY))
        return {0: _FDEntry(d0), 1: _FDEntry(d1), 2: _FDEntry(d2)}

    def _alloc_fd_num(self, proc: _OSProcessRecord, start: int = 0) -> int:
        for fd in range(start, 256):
            if fd not in proc.fds:
                return fd
        raise PosixKernelError(Errno.EMFILE, "Per-process file descriptor table full.")

    def _require_proc(self, pid: int) -> _OSProcessRecord:
        if isinstance(pid, bool) or not isinstance(pid, int):
            raise PosixKernelError(Errno.ESRCH, "Invalid PID type.")
        rec = self._procs.get(pid)
        if rec is None or rec.exited:
            raise PosixKernelError(Errno.ESRCH, f"No active process {pid}.")
        return rec

    def install_vm_program(self, path: str, instructions: Sequence[VMInstruction], *, mode: int = 0o755) -> str:
        payload = canonical_bytes({"program": [ins.to_dict() for ins in instructions], "schema": "vld.os-vm-exec.v1"})
        return self.vfs.write_regular_file(path, payload, mode=mode)

    def install_elf_binary(self, path: str, elf_bytes: bytes, *, mode: int = 0o755) -> str:
        return self.vfs.write_regular_file(path, elf_bytes, mode=mode)

    def spawn_process(
        self,
        name: str,
        program: Sequence[VMInstruction],
        *,
        argv: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        cwd: str = "/",
        parent_pid: int = 0,
    ) -> int:
        norm_cwd = normalize_vfs_path(cwd)
        c_ino = self.vfs._inodes.get(norm_cwd)
        if c_ino is None or c_ino.kind != "dir":
            raise PosixKernelError(Errno.ENOENT, f"cwd {norm_cwd!r} does not exist.")
        mem = VirtualPageSpace(self.page_size, max_resident_pages=self.max_resident_pages)
        mem.map_region(0, self.page_size * 4, flags=PF_R | PF_W)
        vm_p = self.vm_kernel.spawn(name, program, memory=mem, capabilities=frozenset({"emit", "posix"}))
        rec = _OSProcessRecord(
            pid=vm_p.pid,
            parent_pid=parent_pid,
            cwd=norm_cwd,
            argv=tuple(str(x) for x in argv),
            env={str(k): str(v) for k, v in (env or {}).items()},
            vm_proc=vm_p,
            fds=self._default_stdio_fds(),
        )
        self._procs[rec.pid] = rec
        return rec.pid

    def _vm_posix_bridge(self, vm_proc: DeterministicVMProcess, raw_args: tuple[Any, ...]) -> int:
        # Allows VM programs to invoke `SYSCALL("posix", sys_no)` reading args from r1..r6!
        sys_no = int(raw_args[0])
        R = vm_proc.registers
        resp = self.posix_invoke_registers(vm_proc.pid, sys_no, R["r1"], R["r2"], R["r3"], R["r4"], R["r5"], R["r6"])
        return resp.return_value

    def run_process(self, pid: int, *, max_ticks: int = 50_000) -> int:
        rec = self._require_proc(pid)
        ticks = self.vm_kernel.run(rec.vm_proc, max_ticks=max_ticks)
        if rec.vm_proc.halted and not rec.exited:
            rec.exited = True
        return ticks

    def process_stdout(self, pid: int) -> bytes:
        return bytes(self._procs[pid].stdout_bytes)

    def _close_fd_internal(self, proc: _OSProcessRecord, fd: int) -> None:
        entry = proc.fds.pop(fd, None)
        if entry is None:
            raise PosixKernelError(Errno.EBADF, f"Bad file descriptor {fd}.")
        desc = self._open_descs.get(entry.desc_id)
        if desc is not None and desc.pipe_id is not None:
            pipe = self._pipes.get(desc.pipe_id)
            if pipe is not None:
                if desc.kind == "pipe_r":
                    pipe.readers = max(0, pipe.readers - 1)
                elif desc.kind == "pipe_w":
                    pipe.writers = max(0, pipe.writers - 1)
        still_referenced = any(
            fe.desc_id == entry.desc_id
            for p in self._procs.values()
            if not p.reaped
            for fe in p.fds.values()
        )
        if not still_referenced:
            self._open_descs.pop(entry.desc_id, None)

    def _read_c_string(self, mem: VirtualPageSpace, vaddr: int, max_len: int = 1024) -> str:
        out = bytearray()
        for i in range(max_len):
            try:
                b = mem.read_bytes(vaddr + i, 1)[0]
            except PageFaultError as exc:
                raise PosixKernelError(Errno.EFAULT, str(exc)) from exc
            if b == 0:
                try:
                    return out.decode("utf-8", errors="strict")
                except UnicodeDecodeError as exc:
                    raise PosixKernelError(Errno.EINVAL, str(exc)) from exc
            out.append(b)
        raise PosixKernelError(Errno.EINVAL, "Unterminated C-string in process memory.")

    def posix_invoke(self, pid: int, number: int, **kwargs: Any) -> PosixResponse:
        """Invoke Linux `x86_64` syscall `number` on `pid`, returning `PosixResponse` with zero partial mutation on failure."""
        if isinstance(number, bool) or not isinstance(number, int):
            return PosixResponse(-int(Errno.EINVAL), int(Errno.EINVAL), "syscall number must be int")
        name = X86_64_SYSCALL_NUMBERS.get(number)
        if name is None:
            return PosixResponse(-int(Errno.ENOSYS), int(Errno.ENOSYS), f"unsupported x86_64 syscall {number}")
        try:
            res = self._dispatch_named(pid, name, kwargs)
            return PosixResponse(int(res), 0, None)
        except PosixKernelError as exc:
            return PosixResponse(-int(exc.errno), int(exc.errno), str(exc))
        except PageFaultError as exc:
            return PosixResponse(-int(Errno.EFAULT), int(Errno.EFAULT), str(exc))
        except ResidentQuotaExceededError as exc:
            return PosixResponse(-int(Errno.ENOMEM), int(Errno.ENOMEM), str(exc))
        except (VMError, ValueError, TypeError) as exc:
            return PosixResponse(-int(Errno.EINVAL), int(Errno.EINVAL), str(exc))

    def posix_invoke_registers(
        self, pid: int, number: int, a1: int = 0, a2: int = 0, a3: int = 0, a4: int = 0, a5: int = 0, a6: int = 0
    ) -> PosixResponse:
        name = X86_64_SYSCALL_NUMBERS.get(number)
        if name is None:
            return PosixResponse(-int(Errno.ENOSYS), int(Errno.ENOSYS), f"unsupported syscall {number}")
        if name in ("read", "write"):
            return self.posix_invoke(pid, number, fd=a1, address=a2, size=a3)
        if name == "open":
            return self.posix_invoke(pid, number, path_address=a1, flags=a2, mode=a3)
        if name == "close":
            return self.posix_invoke(pid, number, fd=a1)
        if name == "lseek":
            return self.posix_invoke(pid, number, fd=a1, offset=a2, whence=a3)
        if name == "getpid":
            return self.posix_invoke(pid, number)
        if name == "fork":
            return self.posix_invoke(pid, number)
        if name == "exit":
            return self.posix_invoke(pid, number, code=a1)
        return PosixResponse(-int(Errno.ENOSYS), int(Errno.ENOSYS), f"register bridge unsupported for {name}")

    def _dispatch_named(self, pid: int, name: str, args: Mapping[str, Any]) -> int:
        proc = self._require_proc(pid)
        mem = proc.vm_proc.memory

        if name == "getpid":
            return proc.pid

        if name == "exit":
            code = int(args.get("code", 0)) & 0xFF
            for fd in list(proc.fds.keys()):
                self._close_fd_internal(proc, fd)
            proc.exited = True
            proc.exit_code = code
            proc.vm_proc.halted = True
            return 0

        if name == "fork":
            child_vm = self.vm_kernel.fork_process_cow(proc.pid, child_name=f"{proc.vm_proc.name}.fork")
            child_fds: dict[int, _FDEntry] = {}
            for fd_no, fe in proc.fds.items():
                child_fds[fd_no] = _FDEntry(desc_id=fe.desc_id, cloexec=fe.cloexec)
                desc = self._open_descs[fe.desc_id]
                if desc.pipe_id is not None:
                    pipe = self._pipes[desc.pipe_id]
                    if desc.kind == "pipe_r":
                        pipe.readers += 1
                    elif desc.kind == "pipe_w":
                        pipe.writers += 1
            child_rec = _OSProcessRecord(
                pid=child_vm.pid,
                parent_pid=proc.pid,
                cwd=proc.cwd,
                argv=proc.argv,
                env=dict(proc.env),
                vm_proc=child_vm,
                fds=child_fds,
                stdout_bytes=bytearray(proc.stdout_bytes),
                stderr_bytes=bytearray(proc.stderr_bytes),
            )
            child_vm.registers["r0"] = 0
            self._procs[child_rec.pid] = child_rec
            return child_rec.pid

        if name == "wait4":
            target_pid = int(args.get("child_pid", -1))
            options = int(args.get("options", 0))
            status_addr = int(args.get("status_address", 0))
            children = [
                c
                for c in self._procs.values()
                if c.parent_pid == proc.pid and not c.reaped and (target_pid == -1 or c.pid == target_pid)
            ]
            if not children:
                raise PosixKernelError(Errno.ECHILD, "No matching un-reaped child process.")
            exited_child = next((c for c in children if c.exited), None)
            if exited_child is None:
                if options & WNOHANG:
                    return 0
                raise PosixKernelError(Errno.EAGAIN, "Child still running.")
            if status_addr != 0:
                # Validate writeability BEFORE marking child reaped!
                wstatus = (exited_child.exit_code & 0xFF) << 8
                mem.write_bytes(status_addr, struct.pack("<I", wstatus))
            exited_child.reaped = True
            return exited_child.pid

        if name == "getcwd":
            addr = int(args["address"])
            sz = int(args["size"])
            raw = proc.cwd.encode("utf-8") + b"\x00"
            if sz < len(raw):
                raise PosixKernelError(Errno.EINVAL, "getcwd buffer too small.")
            mem.write_bytes(addr, raw)
            return len(raw)

        if name == "chdir":
            p_str = str(args["path"]) if "path" in args else self._read_c_string(mem, int(args["path_address"]))
            norm = normalize_vfs_path(p_str, proc.cwd)
            ino = self.vfs._inodes.get(norm)
            if ino is None:
                raise PosixKernelError(Errno.ENOENT, f"{norm!r} not found.")
            if ino.kind != "dir":
                raise PosixKernelError(Errno.ENOTDIR, f"{norm!r} is not a directory.")
            proc.cwd = norm
            return 0

        if name == "open":
            p_str = str(args["path"]) if "path" in args else self._read_c_string(mem, int(args["path_address"]))
            flags = int(args.get("flags", O_RDONLY))
            mode = int(args.get("mode", 0o644))
            acc = flags & 0x3
            if acc not in (O_RDONLY, O_WRONLY, O_RDWR):
                raise PosixKernelError(Errno.EINVAL, "Invalid open access mode.")
            norm = normalize_vfs_path(p_str, proc.cwd)
            ino = self.vfs._inodes.get(norm)
            if ino is None:
                if not (flags & O_CREAT):
                    raise PosixKernelError(Errno.ENOENT, f"{norm!r} does not exist.")
                parent = posixpath.dirname(norm)
                p_ino = self.vfs._inodes.get(parent)
                if p_ino is None or p_ino.kind != "dir":
                    raise PosixKernelError(Errno.ENOENT, f"Parent directory {parent!r} missing.")
                fd_num = self._alloc_fd_num(proc)
                self.vfs.write_regular_file(norm, b"", mode=mode)
            else:
                if (flags & O_CREAT) and (flags & O_EXCL):
                    raise PosixKernelError(Errno.EEXIST, f"{norm!r} already exists (O_EXCL).")
                if ino.kind == "dir" and (acc != O_RDONLY or (flags & O_TRUNC)):
                    raise PosixKernelError(Errno.EISDIR, "Cannot open directory for write.")
                if ino.kind == "structural" and acc != O_RDONLY:
                    raise PosixKernelError(Errno.EACCES, "Structural-Memory virtual files are immutable read-only.")
                fd_num = self._alloc_fd_num(proc)
                if (flags & O_TRUNC) and acc in (O_WRONLY, O_RDWR) and ino.kind == "file":
                    ino.data.clear()

            did = self._alloc_desc(_OpenFileDescription(kind="vfs", flags=flags, vfs_path=norm, offset=0))
            proc.fds[fd_num] = _FDEntry(desc_id=did, cloexec=bool(flags & O_CLOEXEC))
            return fd_num

        if name == "close":
            self._close_fd_internal(proc, int(args["fd"]))
            return 0

        if name == "dup":
            old_fd = int(args["fd"])
            fe = proc.fds.get(old_fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Invalid fd in dup.")
            new_fd = self._alloc_fd_num(proc)
            desc = self._open_descs[fe.desc_id]
            if desc.pipe_id is not None:
                pipe = self._pipes[desc.pipe_id]
                if desc.kind == "pipe_r":
                    pipe.readers += 1
                elif desc.kind == "pipe_w":
                    pipe.writers += 1
            proc.fds[new_fd] = _FDEntry(desc_id=fe.desc_id, cloexec=False)
            return new_fd

        if name == "dup2":
            old_fd = int(args["fd"])
            new_fd = int(args["new_fd"])
            if not (0 <= new_fd < 256):
                raise PosixKernelError(Errno.EBADF, "new_fd out of range.")
            fe = proc.fds.get(old_fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Invalid old_fd in dup2.")
            if old_fd == new_fd:
                return new_fd
            if new_fd in proc.fds:
                self._close_fd_internal(proc, new_fd)
            desc = self._open_descs[fe.desc_id]
            if desc.pipe_id is not None:
                pipe = self._pipes[desc.pipe_id]
                if desc.kind == "pipe_r":
                    pipe.readers += 1
                elif desc.kind == "pipe_w":
                    pipe.writers += 1
            proc.fds[new_fd] = _FDEntry(desc_id=fe.desc_id, cloexec=False)
            return new_fd

        if name == "fcntl":
            fd = int(args["fd"])
            cmd = int(args["command"])
            arg = int(args.get("arg", 0))
            fe = proc.fds.get(fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Bad fd in fcntl.")
            desc = self._open_descs[fe.desc_id]
            if cmd == F_GETFD:
                return FD_CLOEXEC if fe.cloexec else 0
            if cmd == F_SETFD:
                fe.cloexec = bool(arg & FD_CLOEXEC)
                return 0
            if cmd == F_GETFL:
                return desc.flags
            if cmd == F_SETFL:
                desc.flags = (desc.flags & 0x3) | (arg & (O_APPEND | O_NONBLOCK))
                return 0
            raise PosixKernelError(Errno.EINVAL, f"Unsupported fcntl command {cmd}.")

        if name == "pipe":
            addr = int(args.get("address", 0))
            fd_r = self._alloc_fd_num(proc)
            # Reserve slot temporarily so fd_w gets distinct number
            proc.fds[fd_r] = _FDEntry(-1)
            try:
                fd_w = self._alloc_fd_num(proc)
                if addr != 0:
                    mem.write_bytes(addr, struct.pack("<ii", fd_r, fd_w))
            except Exception:
                proc.fds.pop(fd_r, None)
                raise
            pid_no = self._next_pipe_id
            self._next_pipe_id += 1
            self._pipes[pid_no] = _KernelPipe(pipe_id=pid_no, readers=1, writers=1)
            dr = self._alloc_desc(_OpenFileDescription(kind="pipe_r", flags=O_RDONLY, pipe_id=pid_no))
            dw = self._alloc_desc(_OpenFileDescription(kind="pipe_w", flags=O_WRONLY, pipe_id=pid_no))
            proc.fds[fd_r] = _FDEntry(dr)
            proc.fds[fd_w] = _FDEntry(dw)
            return 0

        if name == "lseek":
            fd = int(args["fd"])
            off = int(args["offset"])
            whence = int(args["whence"])
            fe = proc.fds.get(fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Bad fd in lseek.")
            desc = self._open_descs[fe.desc_id]
            if desc.kind != "vfs" or desc.vfs_path is None:
                raise PosixKernelError(Errno.EINVAL, "Cannot lseek on pipe/stdio.")
            ino = self.vfs._inodes[desc.vfs_path]
            if whence == SEEK_SET:
                new_off = off
            elif whence == SEEK_CUR:
                new_off = desc.offset + off
            elif whence == SEEK_END:
                new_off = ino.size + off
            else:
                raise PosixKernelError(Errno.EINVAL, "Invalid whence in lseek.")
            if new_off < 0:
                raise PosixKernelError(Errno.EINVAL, "Negative resulting file offset.")
            desc.offset = new_off
            return new_off

        if name == "fstat":
            fd = int(args["fd"])
            stat_addr = int(args["stat_address"])
            fe = proc.fds.get(fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Bad fd in fstat.")
            desc = self._open_descs[fe.desc_id]
            if desc.kind == "vfs" and desc.vfs_path is not None:
                ino = self.vfs._inodes[desc.vfs_path]
                ino_num, mode_v, sz_v = ino.ino, ino.mode, ino.size
            else:
                ino_num, mode_v, sz_v = 0, 0o600, 0
            # Write deterministic 24-byte stat header (ino:u64, mode:u64, size:u64)
            mem.write_bytes(stat_addr, struct.pack("<QQQ", ino_num, mode_v, sz_v))
            return 0

        if name == "read":
            fd = int(args["fd"])
            addr = int(args["address"])
            sz = int(args["size"])
            if sz < 0:
                raise PosixKernelError(Errno.EINVAL, "Negative read size.")
            fe = proc.fds.get(fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Bad fd in read.")
            desc = self._open_descs[fe.desc_id]
            if (desc.flags & 0x3) == O_WRONLY:
                raise PosixKernelError(Errno.EBADF, "File descriptor not open for reading.")
            if sz == 0:
                return 0
            if desc.kind == "vfs" and desc.vfs_path is not None:
                ino = self.vfs._inodes[desc.vfs_path]
                chunk = self.vfs.read_slice(ino, desc.offset, sz)
                mem.write_bytes(addr, chunk)
                desc.offset += len(chunk)
                return len(chunk)
            if desc.kind == "pipe_r" and desc.pipe_id is not None:
                pipe = self._pipes[desc.pipe_id]
                if not pipe.buf:
                    if pipe.writers == 0:
                        return 0
                    raise PosixKernelError(Errno.EAGAIN, "Pipe buffer empty (would block).")
                take = min(sz, len(pipe.buf))
                chunk = bytes(pipe.buf[:take])
                mem.write_bytes(addr, chunk)
                del pipe.buf[:take]
                return take
            return 0

        if name == "write":
            fd = int(args["fd"])
            addr = int(args["address"])
            sz = int(args["size"])
            if sz < 0:
                raise PosixKernelError(Errno.EINVAL, "Negative write size.")
            fe = proc.fds.get(fd)
            if fe is None:
                raise PosixKernelError(Errno.EBADF, "Bad fd in write.")
            desc = self._open_descs[fe.desc_id]
            if (desc.flags & 0x3) == O_RDONLY:
                raise PosixKernelError(Errno.EBADF, "File descriptor not open for writing.")
            payload = mem.read_bytes(addr, sz)
            if desc.kind == "stdout":
                proc.stdout_bytes.extend(payload)
                return len(payload)
            if desc.kind == "stderr":
                proc.stderr_bytes.extend(payload)
                return len(payload)
            if desc.kind == "vfs" and desc.vfs_path is not None:
                ino = self.vfs._inodes[desc.vfs_path]
                if ino.kind != "file":
                    raise PosixKernelError(Errno.EACCES, "Target inode is not writable.")
                pos = len(ino.data) if (desc.flags & O_APPEND) else desc.offset
                if pos > len(ino.data):
                    ino.data.extend(b"\x00" * (pos - len(ino.data)))
                ino.data[pos : pos + len(payload)] = payload
                desc.offset = pos + len(payload)
                return len(payload)
            if desc.kind == "pipe_w" and desc.pipe_id is not None:
                pipe = self._pipes[desc.pipe_id]
                if pipe.readers == 0:
                    raise PosixKernelError(Errno.EPIPE, "Broken pipe (zero active readers).")
                if len(pipe.buf) + len(payload) > pipe.capacity:
                    raise PosixKernelError(Errno.EAGAIN, "Pipe buffer full.")
                pipe.buf.extend(payload)
                return len(payload)
            raise PosixKernelError(Errno.EBADF, "Unsupported write target.")

        if name == "mmap":
            length = int(args["length"])
            prot = int(args["prot"])
            flags = int(args["flags"])
            addr_hint = int(args.get("address", 0))
            if length <= 0:
                raise PosixKernelError(Errno.EINVAL, "mmap length must be > 0.")
            if not (flags & MAP_ANONYMOUS) or not (flags & MAP_PRIVATE):
                raise PosixKernelError(Errno.EINVAL, "Only MAP_PRIVATE | MAP_ANONYMOUS supported.")
            if (prot & PROT_WRITE) and (prot & PROT_EXEC):
                raise PosixKernelError(Errno.EACCES, "W^X violation in mmap.")
            vm_flags = (PF_R if (prot & PROT_READ) else 0) | (PF_W if (prot & PROT_WRITE) else 0) | (PF_X if (prot & PROT_EXEC) else 0)
            aligned_len = ((length + self.page_size - 1) // self.page_size) * self.page_size
            if flags & MAP_FIXED:
                if addr_hint <= 0 or addr_hint % self.page_size != 0:
                    raise PosixKernelError(Errno.EINVAL, "MAP_FIXED requires page-aligned positive address.")
                target_addr = addr_hint
            else:
                target_addr = self._next_mmap_addr
                self._next_mmap_addr += aligned_len + self.page_size
            mem.map_region(target_addr, aligned_len, flags=vm_flags, allow_overwrite=bool(flags & MAP_FIXED))
            return target_addr

        if name == "mprotect":
            addr = int(args["address"])
            length = int(args["length"])
            prot = int(args["prot"])
            if addr < 0 or addr % self.page_size != 0 or length <= 0:
                raise PosixKernelError(Errno.EINVAL, "Invalid mprotect alignment or length.")
            if (prot & PROT_WRITE) and (prot & PROT_EXEC):
                raise PosixKernelError(Errno.EACCES, "W^X violation in mprotect.")
            s_p = addr // self.page_size
            e_p = (addr + length - 1) // self.page_size
            for p_i in range(s_p, e_p + 1):
                if p_i not in mem._pages:
                    raise PosixKernelError(Errno.ENOMEM, "Unmapped page in mprotect range.")
            vm_flags = (PF_R if (prot & PROT_READ) else 0) | (PF_W if (prot & PROT_WRITE) else 0) | (PF_X if (prot & PROT_EXEC) else 0)
            for p_i in range(s_p, e_p + 1):
                mem._pages[p_i].flags = vm_flags
            return 0

        if name == "munmap":
            addr = int(args["address"])
            length = int(args["length"])
            if addr < 0 or addr % self.page_size != 0 or length <= 0:
                raise PosixKernelError(Errno.EINVAL, "Invalid munmap alignment or length.")
            s_p = addr // self.page_size
            e_p = (addr + length - 1) // self.page_size
            for p_i in range(s_p, e_p + 1):
                mem._pages.pop(p_i, None)
            return 0

        if name == "execve":
            p_str = str(args["path"]) if "path" in args else self._read_c_string(mem, int(args["path_address"]))
            norm = normalize_vfs_path(p_str, proc.cwd)
            ino = self.vfs._inodes.get(norm)
            if ino is None:
                raise PosixKernelError(Errno.ENOENT, f"Executable {norm!r} not found.")
            if ino.kind != "file" or not (ino.mode & 0o111):
                raise PosixKernelError(Errno.EACCES, f"{norm!r} is not executable.")
            raw_img = bytes(ino.data)

            # Transactional admission: build new memory & program completely BEFORE mutating `proc`!
            if raw_img.startswith(b"\x7fELF"):
                try:
                    lifted, new_mem = lift_elf64_image(raw_img, page_size=self.page_size)
                except LiftingError as exc:
                    raise PosixKernelError(Errno.EINVAL, f"ELF64 validation failed: {exc}") from exc
                # Map writable stack region above segments
                new_mem.map_region(0x70000000, self.page_size * 2, flags=PF_R | PF_W)
                new_prog = (
                    VMInstruction("MOV_IMM", ("r0", lifted.entry_point)),
                    VMInstruction("SYSCALL", ("emit", "r0")),
                    VMInstruction("HALT", ()),
                )
            else:
                try:
                    parsed = parse_strict_json(raw_img)
                    if not isinstance(parsed, dict) or parsed.get("schema") != "vld.os-vm-exec.v1":
                        raise ValueError("Unsupported executable payload.")
                    new_prog = tuple(VMInstruction.from_dict(ins) for ins in parsed["program"])
                except Exception as exc:
                    raise PosixKernelError(Errno.EINVAL, f"Invalid VM executable image: {exc}") from exc
                new_mem = VirtualPageSpace(self.page_size, max_resident_pages=self.max_resident_pages)
                new_mem.map_region(0, self.page_size * 4, flags=PF_R | PF_W)

            # Commit replacement atomically & close FD_CLOEXEC descriptors!
            for fd_no in [k for k, fe in proc.fds.items() if fe.cloexec]:
                self._close_fd_internal(proc, fd_no)
            proc.vm_proc.name = posixpath.basename(norm)
            proc.vm_proc.program = new_prog
            proc.vm_proc.memory = new_mem
            proc.vm_proc.pc = 0
            proc.vm_proc.halted = False
            proc.vm_proc.registers = {f"r{i}": 0 for i in range(16)}
            if "argv" in args:
                proc.argv = tuple(str(x) for x in args["argv"])
            if "env" in args and isinstance(args["env"], Mapping):
                proc.env = {str(k): str(v) for k, v in args["env"].items()}
            return 0

        raise PosixKernelError(Errno.ENOSYS, f"Unhandled syscall {name!r}.")

    # ========================================================================
    # Distributed Copy-on-Write OS Snapshot & Restore (`G28`)
    # ========================================================================

    def snapshot_os_to_m1(self, store: PagedObjectStore) -> PagedByteManifest:
        """Persist multi-process `HeadlessOS` to M1, deduplicating shared immutable CoW pages by SHA-256."""
        page_dedup: dict[str, PagedByteManifest] = {}
        procs_ser: list[dict[str, Any]] = []

        for pid in sorted(self._procs.keys()):
            rec = self._procs[pid]
            if rec.reaped:
                continue
            pages_list: list[dict[str, Any]] = []
            for p_idx in sorted(rec.vm_proc.memory._pages.keys()):
                desc = rec.vm_proc.memory._pages[p_idx]
                raw = bytes(
                    rec.vm_proc.memory._resolve_read_page(
                        p_idx,
                        require_exec=bool(desc.flags & PF_X) and not bool(desc.flags & PF_R),
                    )
                )
                sha = hashlib.sha256(raw).hexdigest()
                if sha not in page_dedup:
                    page_dedup[sha] = store.put_bytes(raw)
                pages_list.append(
                    {
                        "flags": desc.flags,
                        "is_shared_immutable": desc.private_data is None,
                        "manifest": page_dedup[sha].to_dict(),
                        "page_index": p_idx,
                    }
                )
            procs_ser.append(
                {
                    "argv": list(rec.argv),
                    "cwd": rec.cwd,
                    "env": rec.env,
                    "exit_code": rec.exit_code,
                    "exited": rec.exited,
                    "halted": rec.vm_proc.halted,
                    "name": rec.vm_proc.name,
                    "pages": pages_list,
                    "parent_pid": rec.parent_pid,
                    "pc": rec.vm_proc.pc,
                    "pid": rec.pid,
                    "program": [ins.to_dict() for ins in rec.vm_proc.program],
                    "registers": rec.vm_proc.registers,
                    "state_digest": rec.vm_proc.state_digest,
                    "stdout_hex": rec.stdout_bytes.hex(),
                    "ticks": rec.vm_proc.ticks,
                }
            )

        vfs_ser: list[dict[str, Any]] = []
        for path in sorted(self.vfs._inodes.keys()):
            ino = self.vfs._inodes[path]
            vfs_ser.append(
                {
                    "base_coordinate": ino.base_coordinate,
                    "data_hex": ino.data.hex() if ino.kind == "file" else "",
                    "ino": ino.ino,
                    "kind": ino.kind,
                    "memory_id": ino.memory_id,
                    "mode": ino.mode,
                    "path": path,
                    "virtual_size": ino.virtual_size,
                }
            )

        doc = {
            "deduplicated_unique_page_count": len(page_dedup),
            "max_resident_pages": self.max_resident_pages,
            "page_size": self.page_size,
            "processes": procs_ser,
            "schema": "vld.headless-os-snapshot.v1",
            "vfs": vfs_ser,
        }
        return store.put_bytes(canonical_bytes(doc), media_type="application/json")

    @classmethod
    def restore_os_from_m1(
        cls,
        manifest: PagedByteManifest,
        store: PagedObjectStore,
        *,
        structural_memory: StructuralMemoryRegistry | None = None,
    ) -> HeadlessOS:
        """Restore `HeadlessOS` on a fresh host, sharing identical `bytes` objects across CoW-shared pages (`G28`)."""
        doc = parse_strict_json(store.read_bytes(manifest))
        if not isinstance(doc, Mapping) or doc.get("schema") != "vld.headless-os-snapshot.v1":
            raise VMError("Invalid HeadlessOS M1 snapshot schema.")

        os_inst = cls(
            page_size=int(doc["page_size"]),
            max_resident_pages=doc["max_resident_pages"],
            structural_memory=structural_memory,
        )
        os_inst.vfs._inodes.clear()
        for v_item in doc["vfs"]:
            os_inst.vfs._inodes[str(v_item["path"])] = _VFSInode(
                ino=int(v_item["ino"]),
                kind=str(v_item["kind"]),
                mode=int(v_item["mode"]),
                data=bytearray.fromhex(str(v_item["data_hex"])),
                memory_id=v_item["memory_id"],
                base_coordinate=int(v_item["base_coordinate"]),
                virtual_size=int(v_item["virtual_size"]),
            )
            os_inst.vfs._next_ino = max(os_inst.vfs._next_ino, int(v_item["ino"]) + 1)

        shared_immutable_pool: dict[str, bytes] = {}
        for p_item in doc["processes"]:
            mem = VirtualPageSpace(os_inst.page_size, max_resident_pages=os_inst.max_resident_pages)
            for pg in p_item["pages"]:
                pm = PagedByteManifest.from_dict(pg["manifest"])
                if pm.content_digest not in shared_immutable_pool:
                    shared_immutable_pool[pm.content_digest] = store.read_bytes(pm)
                shared_buf = shared_immutable_pool[pm.content_digest]
                if bool(pg["is_shared_immutable"]):
                    mem._pages[int(pg["page_index"])] = _PageDescriptor(flags=int(pg["flags"]), immutable_data=shared_buf)
                else:
                    mem._pages[int(pg["page_index"])] = _PageDescriptor(
                        flags=int(pg["flags"]), private_data=bytearray(shared_buf)
                    )

            vm_p = DeterministicVMProcess(
                pid=int(p_item["pid"]),
                name=str(p_item["name"]),
                program=tuple(VMInstruction.from_dict(i) for i in p_item["program"]),
                memory=mem,
                capabilities=frozenset({"emit", "posix"}),
                registers={str(k): int(v) for k, v in p_item["registers"].items()},
                pc=int(p_item["pc"]),
                ticks=int(p_item["ticks"]),
                halted=bool(p_item["halted"]),
            )
            if vm_p.state_digest != str(p_item["state_digest"]):
                raise VMError(f"Restored process {vm_p.pid} state_digest mismatch.")
            os_inst.vm_kernel._processes[vm_p.pid] = vm_p
            os_inst.vm_kernel._next_pid = max(os_inst.vm_kernel._next_pid, vm_p.pid + 1)
            os_inst._procs[vm_p.pid] = _OSProcessRecord(
                pid=vm_p.pid,
                parent_pid=int(p_item["parent_pid"]),
                cwd=str(p_item["cwd"]),
                argv=tuple(str(x) for x in p_item["argv"]),
                env={str(k): str(v) for k, v in p_item["env"].items()},
                vm_proc=vm_p,
                fds=os_inst._default_stdio_fds(),
                exited=bool(p_item["exited"]),
                exit_code=int(p_item["exit_code"]),
                stdout_bytes=bytearray.fromhex(str(p_item["stdout_hex"])),
            )
        return os_inst


__all__ = [
    "Errno",
    "FD_CLOEXEC",
    "F_GETFD",
    "F_GETFL",
    "F_SETFD",
    "F_SETFL",
    "HeadlessOS",
    "MAP_ANONYMOUS",
    "MAP_FIXED",
    "MAP_PRIVATE",
    "O_APPEND",
    "O_CLOEXEC",
    "O_CREAT",
    "O_EXCL",
    "O_NONBLOCK",
    "O_RDONLY",
    "O_RDWR",
    "O_TRUNC",
    "O_WRONLY",
    "PROT_EXEC",
    "PROT_READ",
    "PROT_WRITE",
    "PosixKernelError",
    "PosixResponse",
    "SEEK_CUR",
    "SEEK_END",
    "SEEK_SET",
    "VirtualFileSystem",
    "WNOHANG",
    "X86_64_SYSCALL_NUMBERS",
    "normalize_vfs_path",
]
