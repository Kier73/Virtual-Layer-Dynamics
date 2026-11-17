r"""Phase 5 — Portable Computational Rooms Across Heterogeneous Substrates (`vld.substrates.rooms`, `ROOM`).

Mathematical contract:
A **Portable Computational Room Capsule** (`vld.room-capsule.v1`) binds one finite modular affine
intention $\mathcal{I} = (a, c, m, x_0, n)$ ($2 \le m \le 2^{31}-1,\; 0 \le n \le 10^{15}$) to a
monotone hash-chained receipt sequence realized across four non-isomorphic execution substrates:
  1. `algebraic-orbit`: Dual-derivation `AffineOrbitLaw` monoid/matrix jump.
  2. `vm-kernel`: Register-machine binary exponentiation inside `DeterministicVMKernel` (with optional
     mid-orbit M1 snapshot + rehydration).
  3. `wasm-binary`: Compiled WebAssembly 1.0 binary module executed inside `WasmRealizer`.
  4. `subprocess-rehomed`: Isolated OS process hard-killed (`os._exit`) on attempt 1 and completed on
     a fresh rehomed worker process.

`verify_room_capsule` independently validates every receipt's semantic output, realizer progression,
and hash chain, and enforces external `(expected_intention_digest, expected_head_digest, expected_generation)`
pins to defeat rollback and substituted-root attacks.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest, parse_strict_json
from vld.ground import AffineOrbitLaw
from vld.journal import GENESIS_PREV_DIGEST
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.substrates.lifting import WasmRealizer, assemble_wasm_affine_orbit_module, lift_wasm_module
from vld.substrates.vm import DeterministicVMKernel, VMInstruction

ALLOWED_REALIZERS = ("algebraic-orbit", "vm-kernel", "wasm-binary", "subprocess-rehomed")
MAX_ROOM_MODULUS = (1 << 31) - 1
MAX_ROOM_COORDINATE = 10**15


class RoomError(ValueError):
    """Raised when a portable room intention, realizer output, receipt chain, or rollback pin fails."""


def build_room_intention(
    *,
    request_id: str = "room001-portable-intention",
    a: int = 17,
    c: int = 29,
    modulus: int = 65521,
    seed: int = 5,
    coordinate: int = 1_000_000_000_137,
) -> dict[str, Any]:
    intention = {
        "coordinate": coordinate,
        "ground": {
            "a": a,
            "c": c,
            "family": "affine_mod",
            "modulus": modulus,
            "seed": seed,
        },
        "projection": "value",
        "request_id": request_id,
        "schema": "vld.room-intention.v1",
    }
    validate_room_intention(intention)
    return intention


def validate_room_intention(intention: Mapping[str, Any]) -> None:
    req = {"coordinate", "ground", "projection", "request_id", "schema"}
    if not isinstance(intention, Mapping) or set(intention.keys()) != req:
        raise RoomError("Invalid room intention keyset.")
    if intention["schema"] != "vld.room-intention.v1" or intention["projection"] != "value":
        raise RoomError("Unsupported room intention schema or projection.")
    if not isinstance(intention["request_id"], str) or not intention["request_id"].strip():
        raise RoomError("Invalid room request_id.")
    coord = intention["coordinate"]
    if isinstance(coord, bool) or not isinstance(coord, int) or not (0 <= coord <= MAX_ROOM_COORDINATE):
        raise RoomError("Room coordinate out of bounds.")
    g = intention["ground"]
    if not isinstance(g, Mapping) or set(g.keys()) != {"a", "c", "family", "modulus", "seed"} or g["family"] != "affine_mod":
        raise RoomError("Invalid room ground descriptor.")
    m = g["modulus"]
    if isinstance(m, bool) or not isinstance(m, int) or not (2 <= m <= MAX_ROOM_MODULUS):
        raise RoomError("Room modulus out of bounds [2, 2^31 - 1].")
    for k in ("a", "c", "seed"):
        v = g[k]
        if isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < m):
            raise RoomError(f"Room parameter {k!r} out of bounds [0, modulus).")


def build_vm_affine_program() -> tuple[VMInstruction, ...]:
    r"""24-instruction `DeterministicVMKernel` program computing $(A x_0 + B) \bmod m$ in $O(\log n)$ steps."""
    # Note: docstring above uses \bmod -> make raw string!
    return (
        VMInstruction("MOV_IMM", ("r5", 1)),          # 0: p = 1
        VMInstruction("MOV_IMM", ("r6", 0)),          # 1: q = 0
        VMInstruction("MOV_IMM", ("r7", 1)),          # 2: const 1
        VMInstruction("JZ", ("r4", 19)),              # 3: if n == 0 -> pc 19
        VMInstruction("AND", ("r8", "r4", "r7")),     # 4: r8 = n & 1
        VMInstruction("JZ", ("r8", 11)),              # 5: if (n & 1) == 0 -> pc 11
        VMInstruction("MUL", ("r5", "r5", "r0")),     # 6: p = (p * a) % m
        VMInstruction("MOD", ("r5", "r5", "r2")),     # 7
        VMInstruction("MUL", ("r6", "r6", "r0")),     # 8: q = (q * a + c) % m
        VMInstruction("ADD", ("r6", "r6", "r1")),     # 9
        VMInstruction("MOD", ("r6", "r6", "r2")),     # 10
        VMInstruction("ADD", ("r8", "r0", "r7")),     # 11: r8 = a + 1
        VMInstruction("MUL", ("r1", "r1", "r8")),     # 12: c = (c * (a + 1)) % m
        VMInstruction("MOD", ("r1", "r1", "r2")),     # 13
        VMInstruction("MUL", ("r0", "r0", "r0")),     # 14: a = (a * a) % m
        VMInstruction("MOD", ("r0", "r0", "r2")),     # 15
        VMInstruction("SHR", ("r4", "r4", "r7")),     # 16: n >>= 1
        VMInstruction("JMP", (3,)),                   # 17: loop
        VMInstruction("MOV_IMM", ("r8", 0)),          # 18: pad
        VMInstruction("MUL", ("r0", "r5", "r3")),     # 19: r0 = (p * seed + q) % m
        VMInstruction("ADD", ("r0", "r0", "r6")),     # 20
        VMInstruction("MOD", ("r0", "r0", "r2")),     # 21
        VMInstruction("SYSCALL", ("emit", "r0")),     # 22
        VMInstruction("HALT", ()),                    # 23
    )


def execute_room_on_realizer(intention: Mapping[str, Any], realizer: str) -> int:
    """Execute `intention` on one of the four heterogeneous realizers."""
    validate_room_intention(intention)
    g = intention["ground"]
    a, c, m, seed = int(g["a"]), int(g["c"]), int(g["modulus"]), int(g["seed"])
    n = int(intention["coordinate"])

    if realizer == "algebraic-orbit":
        return AffineOrbitLaw(multiplier=a, increment=c, modulus=m).navigate(seed, n).value

    if realizer == "vm-kernel":
        kernel = DeterministicVMKernel()
        store = PagedObjectStore(InMemoryPageBacking(page_size=256))
        proc = kernel.spawn(
            "room.affine",
            build_vm_affine_program(),
            initial_registers={"r0": a, "r1": c, "r2": m, "r3": seed, "r4": n},
        )
        # Step halfway, snapshot to M1, restore on a fresh kernel, and finish!
        kernel.run(proc, max_ticks=40)
        snap = kernel.snapshot_process_to_m1(proc, store)
        kernel2 = DeterministicVMKernel()
        restored = kernel2.restore_process_from_m1(snap, store)
        kernel2.run(restored, max_ticks=5_000)
        if not restored.halted or not restored.stdout_events:
            raise RoomError("VM realizer did not halt with emitted value.")
        return int(restored.stdout_events[-1])

    if realizer == "wasm-binary":
        wasm_bytes = assemble_wasm_affine_orbit_module()
        lifted = lift_wasm_module(wasm_bytes)
        runner = WasmRealizer(lifted)
        (val,) = runner.invoke("affine_jump", (a, c, m, seed, n))
        return int(val)

    if realizer == "subprocess-rehomed":
        with tempfile.TemporaryDirectory() as td:
            state_file = Path(td) / "room_rehome.json"
            # Host 1 starts execution, checkpoints intermediate coordinate n // 2, then suffers hard `os._exit(95)`!
            half = n // 2
            rem = n - half
            code_kill = f"""
import json, os
from pathlib import Path
from vld.ground import AffineOrbitLaw
law = AffineOrbitLaw({a}, {c}, {m})
mid = law.navigate({seed}, {half}).value
Path({str(state_file)!r}).write_text(json.dumps({{"mid": mid, "done": {half}}}), encoding="utf-8")
os._exit(95)
"""
            p1 = subprocess.run([sys.executable, "-c", code_kill], capture_output=True, text=True)
            if p1.returncode != 95:
                raise RoomError(f"Expected simulated hard-kill exit 95, got {p1.returncode}: {p1.stderr}")

            # Host 2 rehomes from saved intermediate orbit point and finishes remaining `rem` steps!
            code_resume = f"""
import json
from pathlib import Path
from vld.ground import AffineOrbitLaw
st = json.loads(Path({str(state_file)!r}).read_text(encoding="utf-8"))
law = AffineOrbitLaw({a}, {c}, {m})
final_v = law.navigate(int(st["mid"]), {rem}).value
print(json.dumps({{"value": final_v}}))
"""
            p2 = subprocess.run([sys.executable, "-c", code_resume], capture_output=True, text=True, check=True)
            return int(parse_strict_json(p2.stdout.strip())["value"])

    raise RoomError(f"Unsupported room realizer {realizer!r}.")


def execute_portable_room_suite(
    intention: Mapping[str, Any] | None = None,
    *,
    realizers: Sequence[str] = ALLOWED_REALIZERS,
) -> dict[str, Any]:
    """Run `intention` across all requested realizers and build a monotone hash-chained `vld.room-capsule.v1`."""
    intent = dict(intention) if intention is not None else build_room_intention()
    validate_room_intention(intent)
    intent_digest = domain_digest("vld.room-intention.v1", intent)

    receipts: list[dict[str, Any]] = []
    prev = GENESIS_PREV_DIGEST
    for idx, realizer in enumerate(realizers):
        if realizer not in ALLOWED_REALIZERS:
            raise RoomError(f"Unknown realizer {realizer!r}.")
        val = execute_room_on_realizer(intent, realizer)
        body = {
            "generation": idx + 1,
            "intention_digest": intent_digest,
            "predecessor": prev,
            "realizer": realizer,
            "schema": "vld.room-receipt.v1",
            "status": "finite-ground-checked",
            "value": val,
        }
        rdigest = domain_digest("vld.room-receipt.v1", body)
        rec = {**body, "digest": rdigest}
        receipts.append(rec)
        prev = rdigest

    capsule = {
        "intention": intent,
        "receipts": receipts,
        "schema": "vld.room-capsule.v1",
    }
    verify_room_capsule(capsule)
    return capsule


def verify_room_capsule(
    capsule: Mapping[str, Any],
    *,
    expected_intention_digest: str | None = None,
    expected_head_digest: str | None = None,
    expected_generation: int | None = None,
) -> str:
    """Independently verify a `vld.room-capsule.v1` receipt chain and external anti-rollback pins."""
    if not isinstance(capsule, Mapping) or set(capsule.keys()) != {"intention", "receipts", "schema"}:
        raise RoomError("Invalid room capsule fields.")
    if capsule["schema"] != "vld.room-capsule.v1":
        raise RoomError("Unsupported room capsule schema.")
    intent = capsule["intention"]
    validate_room_intention(intent)
    intent_digest = domain_digest("vld.room-intention.v1", intent)
    if expected_intention_digest is not None and intent_digest != expected_intention_digest:
        raise RoomError("External intention pin mismatch (substituted root detected).")

    g = intent["ground"]
    oracle_value = AffineOrbitLaw(int(g["a"]), int(g["c"]), int(g["modulus"])).navigate(
        int(g["seed"]), int(intent["coordinate"])
    ).value

    receipts = capsule["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise RoomError("Room capsule receipts must be a non-empty list.")

    prev = GENESIS_PREV_DIGEST
    for idx, rec in enumerate(receipts):
        req = {"digest", "generation", "intention_digest", "predecessor", "realizer", "schema", "status", "value"}
        if not isinstance(rec, Mapping) or set(rec.keys()) != req or rec["schema"] != "vld.room-receipt.v1":
            raise RoomError("Invalid room receipt structure.")
        if rec["generation"] != idx + 1 or rec["predecessor"] != prev or rec["intention_digest"] != intent_digest:
            raise RoomError("Broken room receipt lineage or generation sequence.")
        if rec["realizer"] not in ALLOWED_REALIZERS or rec["status"] != "finite-ground-checked":
            raise RoomError("Unrecognized room realizer or status.")
        if type(rec["value"]) is not int or rec["value"] != oracle_value:
            raise RoomError("Room receipt value disagrees with mathematical Ground oracle.")
        body = {k: v for k, v in rec.items() if k != "digest"}
        if domain_digest("vld.room-receipt.v1", body) != rec["digest"]:
            raise RoomError("Room receipt digest mismatch.")
        prev = rec["digest"]

    if expected_generation is not None and len(receipts) != expected_generation:
        raise RoomError("External generation pin mismatch (rollback/truncation detected).")
    if expected_head_digest is not None and prev != expected_head_digest:
        raise RoomError("External head digest pin mismatch (rollback/fork detected).")
    return prev


__all__ = [
    "ALLOWED_REALIZERS",
    "RoomError",
    "build_room_intention",
    "build_vm_affine_program",
    "execute_portable_room_suite",
    "execute_room_on_realizer",
    "validate_room_intention",
    "verify_room_capsule",
]
