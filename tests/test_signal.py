"""Phase 1 tests: Signal-State Machine, multi-domain backends, static footprints, and base-8 transition lowering."""

from __future__ import annotations

import cmath
from fractions import Fraction
import math

import pytest

from vld.signal import (
    AddExpr,
    AmbiguousBranchTrap,
    AmbiguousObservationError,
    AmbiguousTransitionError,
    ConditionalExpr,
    ExactRationalBackend,
    GroundTransitionSystem,
    IdentityGuard,
    InputExpr,
    Instruction,
    ModularRingBackend,
    MulExpr,
    OutputSpec,
    Program,
    Signal8Backend,
    SignalDomainError,
    SignalMachineError,
    SignalStateMachine,
    StaticFootprintError,
    TaskCompiler,
    TaskSpec,
    TransitionCompileError,
    TransitionObservation,
    TransitionOutcome,
    UnknownTransitionError,
    VirtualGroundInterpreter,
    analyze_program_footprints,
    compose_sequential_programs,
    differential_verify_transitions,
    verify_differential_backends,
    verify_signal8_ground,
)
from vld.triad import IndeterminateGuardTrap


def test_exhaustive_4772_signal8_suite_and_tri_backend_differential() -> None:
    report = verify_signal8_ground()
    assert report["status"] == "PASS"
    assert report["total_cases"] == 4772
    assert report["errors"] == []

    diff = verify_differential_backends()
    assert diff["status"] == "PASS"
    assert diff["cases_checked"] == 512
    assert diff["mismatches"] == []


def test_iq_calibration_and_ambiguous_observation_refusal() -> None:
    backend = Signal8Backend()
    pilot = (1.2 + 0.4j, -0.5 + 1.1j, 0.8 - 0.9j, -1.0 - 0.3j)
    sym_val = backend.encode_symbol(5, observer=3, phase_error_bound=0.05)
    iq = backend.emit_iq(sym_val, pilot)
    recovered = backend.ingest_iq(iq, pilot, observer=3, phase_error_bound=0.05)
    assert backend.observe_symbol(recovered) == 5

    # Phase error > pi/8 overlaps adjacent 8-PSK decision regions -> must refuse point decode
    noisy = backend.encode_symbol(5, observer=3, phase_error_bound=0.45)
    assert len(backend.possible_symbols(noisy)) >= 2
    with pytest.raises(AmbiguousObservationError):
        backend.observe_symbol(noisy)

    with pytest.raises(SignalDomainError):
        backend.ingest_iq((1.0 + 0.0j,), (0.0 + 0.0j,), observer=1)


def test_ring_and_rational_mul_capabilities_and_signal8_mul_refusal() -> None:
    sig = Signal8Backend()
    assert not sig.capabilities.supports_mul
    with pytest.raises(SignalDomainError):
        sig.mul(sig.encode_symbol(2, 1), sig.encode_symbol(3, 1))

    # ModularRingBackend supports observer-covariant MUL over Z/256Z
    ring = ModularRingBackend(bit_width=8, observer_profiles=(1, 3, 5, 7))
    interp_ring = VirtualGroundInterpreter(backend=ring)
    task = TaskSpec(
        "ring-poly",
        ("x", "y", "z"),
        (OutputSpec("poly", AddExpr(MulExpr(InputExpr("x"), InputExpr("y")), InputExpr("z"))),),
    )
    res_ring = interp_ring.interpret(
        task,
        {
            "x": ring.encode_symbol(19, observer=3),
            "y": ring.encode_symbol(23, observer=5),
            "z": ring.encode_symbol(77, observer=7),
        },
    )
    assert res_ring.observed_symbols(ring)["poly"] == (19 * 23 + 77) % 256

    # ExactRationalBackend supports exact Fraction MUL with zero rounding error
    rat = ExactRationalBackend(observer_profiles=(1, 2, 4, 8))
    interp_rat = VirtualGroundInterpreter(backend=rat)
    res_rat = interp_rat.interpret(
        task,
        {
            "x": rat.encode_rational("3/7", observer=2),
            "y": rat.encode_rational("14/9", observer=4),
            "z": rat.encode_rational("1/3", observer=8),
        },
    )
    assert res_rat.observed_symbols(rat)["poly"] == Fraction(3, 7) * Fraction(14, 9) + Fraction(1, 3) == Fraction(1, 1)


def test_indeterminate_branch_traps_atomically_without_state_mutation() -> None:
    backend = Signal8Backend()
    interp = VirtualGroundInterpreter(backend=backend)
    prog = Program(
        "trap-atomicity",
        (
            Instruction("LOAD", {"dst": "r0", "address": "in.a"}),
            Instruction("STORE", {"src": "r0", "address": "committed.cell"}),
            Instruction("BRANCH_IDENTITY", {"src": "r0", "target": 4, "tolerance_rad": 0.10}),
            Instruction("STORE", {"src": "r0", "address": "must.never.write"}),
            Instruction("HALT"),
        ),
    )
    # Distance = 0, uncertainty = 0.25 > tolerance = 0.10 => [0, 0.25] straddles 0.10 -> INDETERMINATE!
    uncertain = Signal8Backend().encode_symbol(0, observer=1, phase_error_bound=0.25)
    machine = interp.instantiate(prog, {"in.a": uncertain})
    machine.step()  # LOAD
    machine.step()  # STORE committed.cell
    snap_before = machine.snapshot()

    with pytest.raises(AmbiguousBranchTrap) as exc_info:
        machine.step()  # BRANCH_IDENTITY traps!
    assert isinstance(exc_info.value, IndeterminateGuardTrap)
    assert machine.pc == 2
    assert "must.never.write" not in machine.memory
    assert machine.snapshot() == snap_before


def test_static_footprints_undo_snapshot_and_sequential_composition() -> None:
    # 1. Branch-conditional register initialization must be rejected statically
    bad_reg_prog = Program(
        "uninitialized-on-false-path",
        (
            Instruction("LOAD", {"dst": "cond", "address": "in.x"}),
            Instruction("BRANCH_IDENTITY", {"src": "cond", "target": 3, "tolerance_rad": 0.01}),
            Instruction("JUMP", {"target": 4}),
            Instruction("LOAD", {"dst": "only_true", "address": "in.x"}),
            Instruction("EMIT", {"src": "only_true", "label": "bad"}),
            Instruction("HALT"),
        ),
    )
    with pytest.raises(StaticFootprintError, match="uninitialized register"):
        analyze_program_footprints(bad_reg_prog)

    # 2. Undeclared memory cell read rejected when declared_memory_cells provided
    mem_prog = Program(
        "undeclared-mem",
        (
            Instruction("LOAD", {"dst": "r0", "address": "secret.cell"}),
            Instruction("HALT"),
        ),
    )
    with pytest.raises(StaticFootprintError, match="undeclared memory cell"):
        analyze_program_footprints(mem_prog, declared_memory_cells=("in.x",))

    # 3. Unreachable instruction after HALT flagged when forbid_unreachable=True
    dead_prog = Program(
        "dead-tail",
        (
            Instruction("HALT"),
            Instruction("LOAD", {"dst": "r0", "address": "in.x"}),
            Instruction("HALT"),
        ),
    )
    with pytest.raises(StaticFootprintError, match="unreachable"):
        analyze_program_footprints(dead_prog, forbid_unreachable=True)

    # 4. Safe sequential composition: stage1 writes memory cell "mid", stage2 reads "mid" and emits
    stage1 = Program(
        "stage1",
        (
            Instruction("LOAD", {"dst": "a", "address": "in.a"}),
            Instruction("LOAD", {"dst": "b", "address": "in.b"}),
            Instruction("ADD", {"dst": "s", "left": "a", "right": "b"}),
            Instruction("STORE", {"src": "s", "address": "mid"}),
            Instruction("HALT"),
        ),
    )
    stage2 = Program(
        "stage2",
        (
            Instruction("LOAD", {"dst": "m", "address": "mid"}),
            Instruction("REFRAME", {"dst": "m3", "src": "m", "observer": 3}),
            Instruction("EMIT", {"src": "m3", "label": "result"}),
            Instruction("HALT"),
        ),
    )
    composed = compose_sequential_programs(stage1, stage2, declared_memory_cells=("in.a", "in.b"))
    backend = Signal8Backend()
    interp = VirtualGroundInterpreter(backend=backend)
    machine = interp.instantiate(composed, {"in.a": backend.encode_symbol(6, 1), "in.b": backend.encode_symbol(5, 7)})
    machine.step()
    machine.step()
    snap = machine.snapshot()
    # Undo and re-step
    machine.undo()
    assert machine.pc == 1
    restored = SignalStateMachine.restore(interp, composed, snap)
    res = restored.run()
    assert res.observed_symbols(backend)["result"] == (6 + 5) % 8

    # Tampered snapshot digest must fail restore
    tampered = dict(snap)
    tampered["pc"] = 0
    with pytest.raises(SignalMachineError, match="digest mismatch"):
        SignalStateMachine.restore(interp, composed, tampered)


def test_multi_digit_base8_transition_compiler_and_partial_refusals() -> None:
    # 12 states (> 8 -> 2 octal digits) and 9 actions (> 8 -> 2 octal digits)
    states = tuple(f"S{i}" for i in range(12))
    actions = tuple(f"A{j}" for j in range(9))
    observations = tuple(
        TransitionObservation(
            source=states[i],
            action=actions[j],
            outcomes=(TransitionOutcome(states[(i * 3 + j + 1) % 12], ((i + j) % 8, (2 * i + j) % 8)),),
            support_count=2,
            evidence=("unit-test",),
        )
        for i in range(12)
        for j in range(9)
    )
    system = GroundTransitionSystem(
        states=states,
        actions=actions,
        initial_state="S0",
        channels=("ch0", "ch1"),
        observations=observations,
    )
    assert system.complete
    sequences = [
        (),
        ("A0", "A8", "A4"),
        ("A7", "A1", "A5", "A2", "A8"),
    ]
    diff = differential_verify_transitions(system, sequences)
    assert diff["status"] == "PASS"
    assert diff["cases_checked"] == 3

    # Incomplete and ambiguous transition systems refuse compilation and runtime stepping
    partial = GroundTransitionSystem(
        states=("q0", "q1"),
        actions=("go", "ambig"),
        initial_state="q0",
        channels=("y",),
        observations=(
            TransitionObservation("q0", "go", (TransitionOutcome("q1", (3,)),)),
            TransitionObservation(
                "q0",
                "ambig",
                (TransitionOutcome("q0", (1,)), TransitionOutcome("q1", (2,))),
                support_count=2,
            ),
        ),
    )
    assert not partial.complete
    with pytest.raises(TransitionCompileError):
        differential_verify_transitions(partial, [("go",)])
    with pytest.raises(AmbiguousTransitionError):
        partial.step("q0", "ambig")
    with pytest.raises(UnknownTransitionError):
        partial.step("q1", "go")


def test_whole_program_atomic_trap_footprint_write_guards_and_rational_metrics() -> None:
    # 1. Multi-output TaskCompiler orders all guards before STORE/EMIT so trap leaves zero partial mutation
    b8 = Signal8Backend()
    interp = VirtualGroundInterpreter(backend=b8)
    task = TaskSpec(
        "atomic-multi-output",
        ("x", "y"),
        (
            OutputSpec("first", InputExpr("x")),
            OutputSpec(
                "second",
                ConditionalExpr(IdentityGuard(InputExpr("y"), 0.10), InputExpr("x"), InputExpr("y")),
            ),
        ),
    )
    compiled = TaskCompiler().compile(task)
    fp_report = analyze_program_footprints(
        compiled,
        declared_memory_cells=("x", "y"),
        declared_writable_cells=("output.first", "output.second"),
        forbid_side_effects_before_trap=True,
    )
    assert fp_report.side_effect_before_trap_pcs == ()

    machine = interp.instantiate(
        compiled,
        {"x": b8.encode_symbol(3, 1), "y": b8.encode_symbol(0, 1, phase_error_bound=0.25)},
    )
    with pytest.raises(AmbiguousBranchTrap):
        machine.run()
    assert set(machine.memory.keys()) == {"x", "y"}
    assert machine.outputs == []
    assert machine.trace == []

    # 2. Hand-written program with STORE before reachable trap is rejected when forbid_side_effects_before_trap=True
    impure_prog = Program(
        "impure-before-trap",
        (
            Instruction("LOAD", {"dst": "r0", "address": "x"}),
            Instruction("STORE", {"src": "r0", "address": "leaked"}),
            Instruction("BRANCH_IDENTITY", {"src": "r0", "target": 3, "tolerance_rad": 0.10}),
            Instruction("HALT"),
        ),
    )
    with pytest.raises(StaticFootprintError, match="before a reachable indeterminate guard trap"):
        analyze_program_footprints(impure_prog, forbid_side_effects_before_trap=True)

    with pytest.raises(StaticFootprintError, match="writes undeclared memory cell"):
        analyze_program_footprints(impure_prog, declared_writable_cells=("allowed",))

    # 3. Even on impure hand-written IR, machine.run(atomic_on_trap=True) rolls back whole run on trap!
    m_impure = interp.instantiate(impure_prog, {"x": b8.encode_symbol(0, 1, phase_error_bound=0.25)})
    with pytest.raises(AmbiguousBranchTrap):
        m_impure.run()
    assert "leaked" not in m_impure.memory

    # 4. ExactRationalBackend preserves true unbounded distance & positive GuardDecision.margin
    rat = ExactRationalBackend()
    dec = rat.compare_equal(rat.encode_rational(100, error_bound=50), rat.encode_rational(0), 1.0)
    assert dec.distance == 100.0
    assert dec.lower_bound == 50.0
    assert dec.margin > 0.0

    # Boolean observer / field smuggling in from_dict must be rejected
    with pytest.raises(SignalDomainError):
        b8.value_from_dict({"real": 1.0, "imag": 0.0, "observer": True, "phase_error_bound": 0.0})
    with pytest.raises(SignalDomainError):
        rat.value_from_dict({"val_num": 1, "val_den": 1, "observer": True})
