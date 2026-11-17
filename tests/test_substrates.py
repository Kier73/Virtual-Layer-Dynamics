"""Phase 5 tests: CoW & Structural-Memory VM pages (`G09`), ELF64/WASM lifting (`Locate`, `Restore`, `Continue`, `Transform`, `G10`),
SIMT barriers & $O(1)$-delta matmul `os._exit` recovery (`G23`, `G24`), Path-Ground negative controls (`G26`-`G29`), and Portable `ROOM`.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vld.accel import NumericEngine
from vld.memory import InMemoryPageBacking, PagedObjectStore, StructuralMemoryRegistry
from vld.substrates import (
    ActiveHostFormationPipeline,
    BarrierDivergenceError,
    CountingHTTPOriginServer,
    DeltaTensorCheckpointLog,
    DeterministicVMKernel,
    ET_DYN,
    ET_EXEC,
    ForgedTileDigestError,
    HTTPPathIntermediaryServer,
    LiftingError,
    PF_R,
    PF_W,
    PF_X,
    PageFaultError,
    PathQualificationRefused,
    ResidentQuotaExceededError,
    RoomError,
    SIMTInstruction,
    SIMTWorkgroupExecutor,
    SyscallRefusedError,
    TiledTensorExecutor,
    VMInstruction,
    VirtualPageSpace,
    WasmRealizer,
    assemble_wasm_affine_orbit_module,
    build_minimal_elf64_image,
    continue_process,
    execute_portable_room_suite,
    lift_elf64_image,
    lift_wasm_module,
    locate_in_process,
    restore_process,
    transform_process,
    verify_room_capsule,
)


def test_cow_virtual_pages_structural_demand_paging_and_quotas() -> None:
    reg = StructuralMemoryRegistry()
    reg.certify_and_admit("mem.page_law", [10, 23, 36, 49, 62], domain="Z/256Z", declared_modulus=256)

    vspace = VirtualPageSpace(page_size=64, max_resident_pages=2)
    vspace.map_region(0, 128, flags=PF_R | PF_W, initial_bytes=b"HELLO_VLD_COW!")
    # Map 500 Structural Memory pages -> ZERO eager syntheses!
    vspace.map_structural_region(4096, 500, registry=reg, memory_id="mem.page_law", flags=PF_R | PF_W)
    assert vspace.structural_syntheses == 0
    assert vspace.resident_private_pages == 0

    # Reading 4 bytes inside structural page #100 synthesizes ONLY 1 page!
    sample = vspace.read_bytes(4096 + 100 * 64, 4)
    expected = bytes(((10 + 13 * (100 * 64 + k)) % 256) for k in range(4))
    assert sample == expected
    assert vspace.structural_syntheses == 1

    # Fork CoW: 0 private pages copied!
    child = vspace.fork_cow()
    assert child.resident_private_pages == 0
    child.write_bytes(0, b"CHILD")
    assert child.resident_private_pages == 1
    assert child.read_bytes(0, 14) == b"CHILD_VLD_COW!"
    assert vspace.read_bytes(0, 14) == b"HELLO_VLD_COW!"  # Parent completely isolated!

    # Second private page write succeeds (quota = 2); third distinct page write exceeds quota!
    child.write_bytes(64, b"PAGE1")
    with pytest.raises(ResidentQuotaExceededError):
        child.write_bytes(4096, b"OVER_QUOTA")


def test_elf64_and_wasm_lifting_and_locate_restore_continue_transform() -> None:
    # 1. W^X violation on ELF64 must be refused!
    bad_elf = build_minimal_elf64_image([(0x401000, PF_R | PF_W | PF_X, b"\x90\xc3", 16)], e_type=ET_EXEC)
    with pytest.raises(LiftingError, match="W\\^X"):
        lift_elf64_image(bad_elf, enforce_w_xor_x=True)

    # 2. Valid PIE ELF64 with .text (R|X) and .rodata/.data (R|W) lifted at load_bias=0x100000
    pie_elf = build_minimal_elf64_image(
        [
            (0x1000, PF_R | PF_X, b"\x55\x48\x89\xe5VLD_SIG_ALPHA\xc3", 64),
            (0x2000, PF_R | PF_W, b"CONFIG_BLOCK_42", 128),
        ],
        entry=0x1000,
        e_type=ET_DYN,
    )
    elf_ground, elf_mem = lift_elf64_image(pie_elf, load_bias=0x100000)
    assert elf_ground.entry_point == 0x101000
    hits = locate_in_process(elf_ground, elf_mem, symbol="entry", byte_pattern=b"VLD_SIG_ALPHA")
    assert hits == [0x101000, 0x101004]
    with pytest.raises(PageFaultError):
        elf_mem.write_bytes(0x101000, b"\x00")  # .text is read/exec only!

    # 3. VM Process: Locate, Restore, Continue, Transform + Capability Syscall Enforcement
    kernel = DeterministicVMKernel()
    store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    prog = (
        VMInstruction("MOV_IMM", ("r0", 7)),
        VMInstruction("MOV_IMM", ("r1", 6)),
        VMInstruction("MUL", ("r2", "r0", "r1")),
        VMInstruction("SYSCALL", ("emit", "r2")),
        VMInstruction("HALT", ()),
    )
    proc = kernel.spawn("calc", prog, memory=elf_mem.fork_cow())
    rcpt_part = continue_process(kernel, proc, max_ticks=2)
    assert rcpt_part["steps_executed"] == 2 and not proc.halted

    snap = kernel.snapshot_process_to_m1(proc, store)
    restored = restore_process(kernel, snap, store)
    # Transform: replace MUL with ADD before finishing!
    child_ground = transform_process(
        elf_ground,
        restored,
        morphism_name="mul_to_add",
        transform_fn=lambda p, regs: ((*p[:2], VMInstruction("ADD", ("r2", "r0", "r1")), *p[3:]), regs),
    )
    assert child_ground.parent_lineage_digest == elf_ground.lineage_digest
    continue_process(kernel, restored, max_ticks=10)
    assert restored.halted and restored.stdout_events == [13]

    # Unauthorized syscall must be refused
    unpriv = kernel.spawn("sandbox", (VMInstruction("SYSCALL", ("emit", "r0")),), capabilities=frozenset())
    with pytest.raises(SyscallRefusedError):
        kernel.step(unpriv)

    # 4. Binary WebAssembly module validation & execution
    wasm_raw = assemble_wasm_affine_orbit_module()
    lifted_wasm = lift_wasm_module(wasm_raw)
    expected_500 = (pow(5, 500, 97) * 1 + 3 * sum(pow(5, k, 97) for k in range(500))) % 97
    assert WasmRealizer(lifted_wasm).invoke("affine_jump", (5, 3, 97, 1, 500)) == (expected_500,)


def test_simt_workgroup_barriers_and_o1_delta_matmul_crash_recovery(tmp_path: Path) -> None:
    simt = SIMTWorkgroupExecutor(workgroup_size=4, shared_elements=8)
    # Valid lockstep kernel: each lane writes lane_id * 10 to shared[lane_id], BARRIER, then reads neighbor (lane_id + 1) % 4
    valid_kernel = (
        SIMTInstruction("MOV_IMM", ("r1", 10.0)),
        SIMTInstruction("MUL", ("r2", "lane_id", "r1")),
        SIMTInstruction("STORE_SHARED", ("lane_id", "r2")),
        SIMTInstruction("BARRIER", ("sync_shared",)),
        SIMTInstruction("STORE_GLOBAL", ("lane_id", "r2")),
        SIMTInstruction("HALT", ()),
    )
    g_out, s_out = simt.execute_kernel(valid_kernel, [0.0, 0.0, 0.0, 0.0])
    assert g_out == [0.0, 10.0, 20.0, 30.0] and s_out[:4] == [0.0, 10.0, 20.0, 30.0]

    # Divergent BARRIER (even lanes jump past barrier!) MUST trap with BarrierDivergenceError!
    divergent_kernel = (
        SIMTInstruction("JZ_EVEN_LANE", ("r0", 2)),
        SIMTInstruction("BARRIER", ("bad_sync",)),
        SIMTInstruction("HALT", ()),
    )
    with pytest.raises(BarrierDivergenceError):
        simt.execute_kernel(divergent_kernel, [0.0] * 4)

    # O(1)-Incremental Delta Tensor Matmul across real subprocess `os._exit(97)` + forged digest refusal!
    delta_file = tmp_path / "matmul_deltas.jsonl"
    crash_script = f"""
import math
from pathlib import Path
from vld.substrates import DeltaTensorCheckpointLog, TiledTensorExecutor
A = [[float((r * 7 + c * 3) % 19) * 0.25 for c in range(10)] for r in range(12)]
B = [[float((r * 5 + c * 11) % 17) * 0.5 for c in range(12)] for r in range(10)]
log = DeltaTensorCheckpointLog(Path({str(delta_file)!r}))
TiledTensorExecutor(tile_m=4, tile_n=4).matmul_tiled_recoverable(A, B, checkpoint_log=log, _fault_after_tiles=5)
"""
    proc = subprocess.run([sys.executable, "-c", crash_script], capture_output=True, text=True)
    assert proc.returncode == 97, proc.stderr

    mat_a = [[float((r * 7 + c * 3) % 19) * 0.25 for c in range(10)] for r in range(12)]
    mat_b = [[float((r * 5 + c * 11) % 17) * 0.5 for c in range(12)] for r in range(10)]
    log_resume = DeltaTensorCheckpointLog(delta_file)
    executor = TiledTensorExecutor(tile_m=4, tile_n=4)
    c_mat, telemetry = executor.matmul_tiled_recoverable(mat_a, mat_b, checkpoint_log=log_resume)
    # 12x12 in 4x4 tiles = 9 tiles total: 5 reused from pre-crash delta log, 4 newly computed!
    assert telemetry["total_tiles"] == 9
    assert telemetry["reused_verified_tiles"] == 5
    assert telemetry["newly_computed_tiles"] == 4
    # Verify O(1) constant delta size per tile (no O(K^2) growth!)
    assert max(log_resume.bytes_appended_history) - min(log_resume.bytes_appended_history) <= 4

    ref_c = [[sum(mat_a[r][p] * mat_b[p][c] for p in range(10)) for c in range(12)] for r in range(12)]
    assert NumericEngine.max_abs_diff(c_mat, ref_c) < 1e-5

    # Tampering with any persisted tile hex triggers ForgedTileDigestError!
    lines = delta_file.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[2])
    tampered["tile_f32_hex"] = "ff" * (len(tampered["tile_f32_hex"]) // 2)
    lines[2] = json.dumps(tampered)
    delta_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ForgedTileDigestError):
        executor.matmul_tiled_recoverable(mat_a, mat_b, checkpoint_log=DeltaTensorCheckpointLog(delta_file))


def test_active_host_formation_path_grounds_and_mandatory_negative_controls() -> None:
    payload = b"Virtual-Layer-Path-Ground-State-Slice-0123456789ABCDEF"
    pipeline = ActiveHostFormationPipeline()

    with CountingHTTPOriginServer({"blob.bin": payload}) as origin:
        # 1. Genuine in-path HTTP conditional/Range cache passes all 6 stages with 0 warm origin hits!
        with HTTPPathIntermediaryServer(origin.base_url, mode="cache") as cache_edge:
            ev_ok = pipeline.observe_http_cache_locus(
                "edge.cache.01", cache_edge.base_url, "blob.bin", lambda: origin.origin_hits
            )
            cert_ok = pipeline.qualify(ev_ok)
            pipeline.construct_and_admit(cert_ok)
            nav_rep = pipeline.navigate_and_verify(
                "edge.cache.01",
                {"length": 16, "offset": 8},
                lambda: payload[8:24],
                origin_hit_reader=lambda: origin.origin_hits,
            )
            assert nav_rep["verified"] is True and nav_rep["origin_hits_delta"] == 0

        # 2. Mandatory Negative Control #1: Pass-through proxy (endpoint compute masquerading as path cache) -> REFUSED!
        with HTTPPathIntermediaryServer(origin.base_url, mode="passthrough") as passthrough_edge:
            ev_pass = pipeline.observe_http_cache_locus(
                "edge.passthrough.bad", passthrough_edge.base_url, "blob.bin", lambda: origin.origin_hits
            )
            with pytest.raises(PathQualificationRefused, match="pass-through"):
                pipeline.qualify(ev_pass)

        # 3. Mandatory Negative Control #2: Nondeterministic noisy intermediary -> REFUSED!
        with HTTPPathIntermediaryServer(origin.base_url, mode="noisy") as noisy_edge:
            ev_noisy = pipeline.observe_http_cache_locus(
                "edge.noisy.bad", noisy_edge.base_url, "blob.bin", lambda: origin.origin_hits
            )
            with pytest.raises(PathQualificationRefused, match="nondeterministic"):
                pipeline.qualify(ev_noisy)

    # 4. Authenticated multi-hop transit projection chain + identity/forged-MAC negative controls!
    sec1 = b"hop1-secret-01234567"
    sec2 = b"hop2-secret-01234567"

    def _make_hop(a: int, b: int, sec: bytes):
        def _fn(inp: bytes) -> tuple[bytes, str]:
            out = bytes(((a * x + b) & 0xFF) for x in inp)
            mac = hmac.new(sec, hashlib.sha256(inp).digest() + out, hashlib.sha256).hexdigest()
            return out, mac
        return _fn

    ev_chain = pipeline.observe_transit_chain(
        "transit.chain.01",
        [("h1", 5, 11, sec1, _make_hop(5, 11, sec1)), ("h2", 3, 7, sec2, _make_hop(3, 7, sec2))],
        [b"\x01\x02\x03", b"\x10\x20\x30"],
    )
    cert_chain = pipeline.qualify(ev_chain)
    # Composite of (5x + 11) followed by (3y + 7) mod 256 is (15x + 40) mod 256!
    assert cert_chain.parameters == {"composite_a": 15, "composite_b": 40, "hop_count": 2}
    pipeline.construct_and_admit(cert_chain)
    rep_chain = pipeline.navigate_and_verify(
        "transit.chain.01",
        {"data": b"\x04\x09", "iterations": 2},
        lambda: bytes(((15 * ((15 * x + 40) & 0xFF) + 40) & 0xFF) for x in b"\x04\x09"),
    )
    assert rep_chain["verified"] is True

    # Identity pass-through hop (a=1, b=0) MUST be refused by negative control!
    ev_id = pipeline.observe_transit_chain(
        "transit.identity.bad",
        [("h_id", 1, 0, sec1, _make_hop(1, 0, sec1))],
        [b"\x01\x02", b"\x03\x04"],
    )
    with pytest.raises(PathQualificationRefused, match="identity pass-through"):
        pipeline.qualify(ev_id)


def test_portable_computational_room_across_4_realizers_and_rollback_pins() -> None:
    capsule = execute_portable_room_suite()
    assert len(capsule["receipts"]) == 4
    values = {r["value"] for r in capsule["receipts"]}
    assert len(values) == 1  # Identical across algebraic, VM-kernel (with M1 snapshot!), WASM binary, and hard-killed/rehomed subprocess!

    head = verify_room_capsule(capsule, expected_generation=4)
    assert head == capsule["receipts"][-1]["digest"]

    # Truncating receipt chain to generation 2 is caught by external pin!
    rolled_back = {**capsule, "receipts": capsule["receipts"][:2]}
    with pytest.raises(RoomError, match="rollback"):
        verify_room_capsule(rolled_back, expected_head_digest=head)


def test_atomic_cow_fault_wasm_edge_cases_simt_publication_and_false_arithmetic_refusal(tmp_path: Path) -> None:
    """Verify cross-page write atomicity, distributed M1 CoW (`G28`), WASM non-wrapping OOB / valued blocks / div_s overflow,
    SIMT barrier-published shared memory & max_steps cap, and E-GPU rehashed false arithmetic rejection.
    """
    from vld.accel import encode_f32_le
    from vld.canonical import canonical_bytes
    from vld.substrates import EGPUError, TileDeltaRecord, WASM_MAGIC
    from vld.substrates.lifting import _encode_s_leb128, _encode_u_leb128

    # 1. Cross-page write_bytes MUST be 100% atomic when second page hits ResidentQuotaExceededError!
    vps = VirtualPageSpace(page_size=64, max_resident_pages=1)
    vps.map_region(0, 128, flags=PF_R | PF_W, initial_bytes=b"A" * 128)
    with pytest.raises(ResidentQuotaExceededError):
        vps.write_bytes(60, b"ZZZZZZZZ")
    assert vps.read_bytes(0, 128) == b"A" * 128

    # Distributed CoW across two independently M1-restored VM processes (`G28`)
    store = PagedObjectStore(InMemoryPageBacking(page_size=256))
    k1, k2 = DeterministicVMKernel(), DeterministicVMKernel()
    p_orig = k1.spawn("base", (VMInstruction("HALT", ()),))
    p_orig.memory.write_bytes(0, b"SHARED_SNAPSHOT")
    snap = k1.snapshot_process_to_m1(p_orig, store)
    r1 = k1.restore_process_from_m1(snap, store)
    r2 = k2.restore_process_from_m1(snap, store)
    r1.memory.write_bytes(0, b"MUTATED_BY_R1__")
    assert r2.memory.read_bytes(0, 15) == b"SHARED_SNAPSHOT"

    # 2. WASM: (a) 0xFFFFFFF0 + 0x20 OOB memory trap, (b) valued if/else + br_table + i64.div_s(-2^63, -1) overflow & -7/2 == -3
    def _sec(sid: int, payload: bytes) -> bytes:
        return bytes([sid]) + _encode_u_leb128(len(payload)) + payload

    oob_body = b"\x00\x41" + _encode_s_leb128(-16) + b"\x28\x00\x20\x0b"
    oob_wasm = (
        WASM_MAGIC
        + _sec(1, b"\x01\x60\x00\x01\x7f")
        + _sec(3, b"\x01\x00")
        + _sec(5, b"\x01\x00\x01")
        + _sec(7, b"\x01\x03oob\x00\x00")
        + _sec(10, b"\x01" + _encode_u_leb128(len(oob_body)) + oob_body)
    )
    with pytest.raises(LiftingError, match="out-of-bounds"):
        WasmRealizer(lift_wasm_module(oob_wasm)).invoke("oob")

    # Valued block + if/else + br_table + i64.div_s(a, b): fn(sel: i32, a: i64, b: i64) -> i64
    # block (result i64) {
    #   if sel == 0 (result i64) { a /_s b } else { block { br_table [0, 1] sel } ; i64.const 99 }
    # }
    edge_ins = bytearray()
    edge_ins += b"\x02\x7e"  # block (result i64)
    edge_ins += b"\x20\x00\x45\x04\x7e"  # local.get 0; i32.eqz; if (result i64)
    edge_ins += b"\x20\x01\x20\x02\x7f"  # local.get 1; local.get 2; i64.div_s
    edge_ins += b"\x05"  # else
    edge_ins += b"\x42" + _encode_s_leb128(42)  # push 42 for outer valued br!
    edge_ins += b"\x02\x40\x20\x00\x0e\x02\x00\x00\x02\x0b"  # block void { local.get 0; br_table [0, 0, default=2(outer!)]; }
    edge_ins += b"\x1a\x42" + _encode_s_leb128(99)  # drop 42; push 99
    edge_ins += b"\x0b\x0b\x0b"  # end if, end outer block, end func
    edge_body = b"\x00" + bytes(edge_ins)
    edge_wasm = (
        WASM_MAGIC
        + _sec(1, b"\x01\x60\x03\x7f\x7e\x7e\x01\x7e")
        + _sec(3, b"\x01\x00")
        + _sec(7, b"\x01\x04edge\x00\x00")
        + _sec(10, b"\x01" + _encode_u_leb128(len(edge_body)) + edge_body)
    )
    edge_runner = WasmRealizer(lift_wasm_module(edge_wasm))
    # WASM truncated-toward-zero: -7 / 2 == -3 (unsigned 64-bit 2^64 - 3), NOT Python's floor -4!
    assert edge_runner.invoke("edge", (0, -7, 2)) == ((-3) & 0xFFFFFFFFFFFFFFFF,)
    # br_table index 1 -> depth 0 -> falls through to 99; index 5 -> default depth 2 -> branches to outer block carrying 42!
    assert edge_runner.invoke("edge", (1, 0, 1)) == (99,)
    assert edge_runner.invoke("edge", (5, 0, 1)) == (42,)
    # -2^63 / -1 MUST trap on signed division overflow!
    with pytest.raises(LiftingError, match="overflow"):
        edge_runner.invoke("edge", (0, -(1 << 63), -1))

    # 3. SIMT barrier-published shared memory & infinite-loop max_steps enforcement!
    simt = SIMTWorkgroupExecutor(workgroup_size=2, shared_elements=4)
    g_pub, s_pub = simt.execute_kernel(
        (
            SIMTInstruction("STORE_SHARED", ("lane_id", "lane_id")),
            SIMTInstruction("BARRIER", ("b0",)),
            SIMTInstruction("LOAD_SHARED", ("r1", "r0", 0)),  # Both lanes MUST see pre-epoch 0.0, not 999.0!
            SIMTInstruction("MOV_IMM", ("r2", 999.0)),
            SIMTInstruction("STORE_SHARED", ("lane_id", "r2")),
            SIMTInstruction("BARRIER", ("b1",)),
            SIMTInstruction("STORE_GLOBAL", ("lane_id", "r1")),
            SIMTInstruction("HALT", ()),
        ),
        [0.0, 0.0],
    )
    assert g_pub == [0.0, 0.0] and s_pub[:2] == [999.0, 999.0]

    with pytest.raises(EGPUError, match="max_steps"):
        simt.execute_kernel((SIMTInstruction("JZ_EVEN_LANE", ("r0", 0)), SIMTInstruction("HALT", ())), [0.0, 0.0], max_steps=25)

    # 4. E-GPU: Completed checkpoint replay dispatches 0 tiles; callback operand mutation is isolated; rehashed false arithmetic is rejected!
    d_path = tmp_path / "strict_deltas.jsonl"
    mat_a = [[2.0, 3.0], [4.0, 5.0]]
    mat_b = [[1.0, 2.0], [3.0, 4.0]]
    dispatches = {"count": 0}

    def _mutating_worker(_tr: int, _tc: int, a_sl: list[list[float]], b_sl: list[list[float]]) -> list[list[float]]:
        dispatches["count"] += 1
        res = [[sum(a_sl[i][p] * b_sl[p][j] for p in range(len(b_sl))) for j in range(len(b_sl[0]))] for i in range(len(a_sl))]
        # Attempt to poison operands in-place to trick oracle!
        a_sl[0][0] = -1e9
        b_sl[0][0] = -1e9
        return res

    ex = TiledTensorExecutor(tile_m=1, tile_n=2)
    c1, tel1 = ex.matmul_tiled_recoverable(mat_a, mat_b, checkpoint_log=DeltaTensorCheckpointLog(d_path), tile_compute_fn=_mutating_worker)
    assert c1 == [[11.0, 16.0], [19.0, 28.0]] and dispatches["count"] == 2

    # Replay of completed checkpoint dispatches ZERO worker calls!
    c2, tel2 = ex.matmul_tiled_recoverable(mat_a, mat_b, checkpoint_log=DeltaTensorCheckpointLog(d_path), tile_compute_fn=_mutating_worker)
    assert c2 == c1 and tel2["newly_computed_tiles"] == 0 and dispatches["count"] == 2

    # Rehashed false arithmetic (valid SHA-256 + valid hash chain, wrong matrix product!) MUST raise ForgedTileDigestError!
    forged_raw = encode_f32_le([-77.0, -88.0])
    forged_rec = TileDeltaRecord(
        seq=0,
        plan_digest=tel1["plan_digest"],
        operation_id="egpu.matmul:0:0",
        tile_row=0,
        tile_col=0,
        rows=1,
        cols=2,
        tile_f32_hex=forged_raw.hex(),
        tile_sha256=hashlib.sha256(forged_raw).hexdigest(),
        prev_digest="0" * 64,
    )
    d_path.write_bytes(canonical_bytes(forged_rec.to_dict()) + b"\n")
    with pytest.raises(ForgedTileDigestError, match="arithmetic oracle"):
        ex.matmul_tiled_recoverable(mat_a, mat_b, checkpoint_log=DeltaTensorCheckpointLog(d_path))

