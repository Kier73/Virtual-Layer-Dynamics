r"""Exhaustive 4,772-case Signal8 invariant suite and multi-backend differential verification."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Any

from vld.signal.backends import (
    ExactRationalBackend,
    ModularRingBackend,
    SIGNAL8_OBSERVERS,
    Signal8Backend,
)
from vld.signal.compiler import (
    AddExpr,
    ConditionalExpr,
    IdentityGuard,
    InputExpr,
    OutputSpec,
    ReframeExpr,
    SubExpr,
    TaskCompiler,
    TaskSpec,
)
from vld.signal.ir import Instruction, Program
from vld.signal.machine import SIGNAL8_GROUND, VirtualGroundInterpreter


def verify_signal8_ground() -> dict[str, Any]:
    r"""Run the exhaustive 4,772-case finite $\mathbb{Z}/8\mathbb{Z}$ signal-algebra verification suite."""
    backend = Signal8Backend()
    interpreter = VirtualGroundInterpreter(SIGNAL8_GROUND, backend)
    counts = {
        "encode_observe": 0,
        "reframe_round_trip": 0,
        "binary_arithmetic": 0,
        "observer_naturality": 0,
        "two_operation_composition": 0,
        "conditional_paths": 0,
        "compiled_task_paths": 0,
    }
    errors: list[str] = []

    # 1. 4 * 8 = 32 encode/observe round-trips
    for u in SIGNAL8_OBSERVERS:
        for k in range(8):
            counts["encode_observe"] += 1
            if backend.observe_symbol(backend.encode_symbol(k, u)) != k:
                errors.append(f"encode_observe mismatch: u={u}, k={k}")

    # 2. 4 * 4 * 8 = 128 reframe round-trips
    for u, v, k in product(SIGNAL8_OBSERVERS, SIGNAL8_OBSERVERS, range(8)):
        counts["reframe_round_trip"] += 1
        val = backend.encode_symbol(k, u)
        rt = backend.reframe(backend.reframe(val, v), u)
        if backend.observe_symbol(rt) != k or rt.observer != u:
            errors.append(f"reframe_round_trip mismatch: {u}->{v}->{u}, k={k}")

    # 3. 4 * 8 * 8 * 2 = 512 binary arithmetic cases
    for u, a, b in product(SIGNAL8_OBSERVERS, range(8), range(8)):
        left = backend.encode_symbol(a, u)
        right = backend.encode_symbol(b, u)
        for op, expected in (("ADD", (a + b) % 8), ("SUB", (a - b) % 8)):
            counts["binary_arithmetic"] += 1
            out = backend.add(left, right) if op == "ADD" else backend.sub(left, right)
            if backend.observe_symbol(out) != expected:
                errors.append(f"{op} mismatch: u={u}, a={a}, b={b}")

    # 4. 4 * 4 * 8 * 8 * 2 = 2,048 observer naturality cases
    for u, v, a, b in product(SIGNAL8_OBSERVERS, SIGNAL8_OBSERVERS, range(8), range(8)):
        sa, sb = backend.encode_symbol(a, u), backend.encode_symbol(b, u)
        ma, mb = backend.reframe(sa, v), backend.reframe(sb, v)
        for op in ("ADD", "SUB"):
            counts["observer_naturality"] += 1
            before = backend.add(sa, sb) if op == "ADD" else backend.sub(sa, sb)
            after = backend.add(ma, mb) if op == "ADD" else backend.sub(ma, mb)
            if backend.observe_symbol(backend.reframe(before, v)) != backend.observe_symbol(after):
                errors.append(f"observer_naturality mismatch: {op}, {u}->{v}, a={a}, b={b}")

    # 5. 4 * 8^3 = 2,048 two-operation cross-observer compositions
    for op1, op2 in (("ADD", "ADD"), ("ADD", "SUB"), ("SUB", "ADD"), ("SUB", "SUB")):
        for a, b, c in product(range(8), repeat=3):
            ua, ub, uc = SIGNAL8_OBSERVERS[a % 4], SIGNAL8_OBSERVERS[b % 4], SIGNAL8_OBSERVERS[c % 4]
            va = backend.encode_symbol(a, ua)
            vb = backend.encode_symbol(b, ub)
            vc = backend.encode_symbol(c, uc)
            mid = backend.add(va, vb) if op1 == "ADD" else backend.sub(va, vb)
            mid = backend.reframe(mid, SIGNAL8_OBSERVERS[(a + b + c) % 4])
            out = backend.add(mid, vc) if op2 == "ADD" else backend.sub(mid, vc)
            partial = (a + b) % 8 if op1 == "ADD" else (a - b) % 8
            expected = (partial + c) % 8 if op2 == "ADD" else (partial - c) % 8
            counts["two_operation_composition"] += 1
            if backend.observe_symbol(out) != expected:
                errors.append(f"composition mismatch: {op1}->{op2}({a},{b},{c})")

    # 6. 2 conditional IR paths + 2 compiled task paths = 4 cases (total = 4,772)
    prog = Program(
        "conditional-select",
        (
            Instruction("LOAD", {"dst": "a", "address": "input.a"}),
            Instruction("LOAD", {"dst": "b", "address": "input.b"}),
            Instruction("ADD", {"dst": "sum", "left": "a", "right": "b"}),
            Instruction("BRANCH_IDENTITY", {"src": "sum", "target": 6, "tolerance_rad": 0.02}),
            Instruction("MOVE", {"dst": "selected", "src": "a"}),
            Instruction("JUMP", {"target": 7}),
            Instruction("MOVE", {"dst": "selected", "src": "b"}),
            Instruction("EMIT", {"src": "selected", "label": "selected"}),
            Instruction("HALT"),
        ),
    )
    cases = ((3, 5, 5), (3, 4, 3))
    for a, b, expected in cases:
        counts["conditional_paths"] += 1
        res = interpreter.execute(prog, {"input.a": backend.encode_symbol(a, 1), "input.b": backend.encode_symbol(b, 7)})
        if res.observed_symbols(backend).get("selected") != expected:
            errors.append(f"conditional_paths mismatch: a={a}, b={b}")

    in_a, in_b = InputExpr("input.a"), InputExpr("input.b")
    task = TaskSpec(
        "finite-task-compiler-check",
        ("input.a", "input.b"),
        (OutputSpec("selected", ConditionalExpr(IdentityGuard(AddExpr(in_a, in_b), 0.02), in_b, in_a)),),
    )
    compiled = TaskCompiler().compile(task)
    for a, b, expected in cases:
        counts["compiled_task_paths"] += 1
        res = interpreter.execute(
            compiled,
            {"input.a": backend.encode_symbol(a, SIGNAL8_OBSERVERS[0]), "input.b": backend.encode_symbol(b, SIGNAL8_OBSERVERS[3])},
        )
        if res.observed_symbols(backend).get("selected") != expected:
            errors.append(f"compiled_task_paths mismatch: a={a}, b={b}")

    return {
        "counts": counts,
        "errors": errors,
        "ground_fingerprint": SIGNAL8_GROUND.fingerprint,
        "ground_id": SIGNAL8_GROUND.ground_id,
        "status": "PASS" if not errors else "FAIL",
        "total_cases": sum(counts.values()),
    }


def verify_differential_backends() -> dict[str, Any]:
    r"""Execute an identical compiled `Program` across three non-isomorphic backends ($\mathbb{Z}/8\mathbb{Z}$ phasor, $\mathbb{Z}/8\mathbb{Z}$ ring, $\mathbb{Q}$)."""
    in_x = InputExpr("input.x")
    in_y = InputExpr("input.y")
    in_z = InputExpr("input.z")
    # Task: if (x - y == 0) emit reframe(x + z, obs_target) else emit (x - z)
    task = TaskSpec(
        "differential-tri-backend",
        ("input.x", "input.y", "input.z"),
        (
            OutputSpec(
                "out",
                ConditionalExpr(
                    IdentityGuard(SubExpr(in_x, in_y), 1e-6),
                    ReframeExpr(AddExpr(in_x, in_z), 3),
                    SubExpr(in_x, in_z),
                ),
            ),
        ),
    )
    prog_mod8 = TaskCompiler().compile(task)

    sig_backend = Signal8Backend()
    ring_backend = ModularRingBackend(bit_width=3, observer_profiles=(1, 3, 5, 7))
    rat_backend = ExactRationalBackend(observer_profiles=(1, 2, 3, 4))

    sig_interp = VirtualGroundInterpreter(backend=sig_backend)
    ring_interp = VirtualGroundInterpreter(backend=ring_backend)
    rat_interp = VirtualGroundInterpreter(backend=rat_backend)

    checked = 0
    mismatches: list[str] = []
    for x, y, z in product(range(8), repeat=3):
        checked += 1
        res_sig = sig_interp.execute(
            prog_mod8,
            {
                "input.x": sig_backend.encode_symbol(x, 1),
                "input.y": sig_backend.encode_symbol(y, 5),
                "input.z": sig_backend.encode_symbol(z, 7),
            },
        )
        res_ring = ring_interp.execute(
            prog_mod8,
            {
                "input.x": ring_backend.encode_symbol(x, 1),
                "input.y": ring_backend.encode_symbol(y, 5),
                "input.z": ring_backend.encode_symbol(z, 7),
            },
        )
        res_rat = rat_interp.execute(
            prog_mod8,
            {
                "input.x": rat_backend.encode_rational(x, 1),
                "input.y": rat_backend.encode_rational(y, 2),
                "input.z": rat_backend.encode_rational(z, 4),
            },
        )
        v_sig = res_sig.observed_symbols(sig_backend)["out"]
        v_ring = res_ring.observed_symbols(ring_backend)["out"]
        v_rat_raw = res_rat.observed_symbols(rat_backend)["out"]
        assert isinstance(v_rat_raw, Fraction) and v_rat_raw.denominator == 1
        v_rat_mod8 = int(v_rat_raw) % 8

        expected = (x + z) % 8 if x == y else (x - z) % 8
        if not (v_sig == v_ring == v_rat_mod8 == expected):
            mismatches.append(f"({x},{y},{z}): sig={v_sig}, ring={v_ring}, rat_mod8={v_rat_mod8}, exp={expected}")

    return {
        "backends": [sig_backend.domain_id, ring_backend.domain_id, rat_backend.domain_id],
        "cases_checked": checked,
        "mismatches": mismatches,
        "program_fingerprint": prog_mod8.fingerprint,
        "status": "PASS" if not mismatches else "FAIL",
    }
