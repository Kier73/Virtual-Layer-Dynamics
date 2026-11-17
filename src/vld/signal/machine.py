r"""Backend-agnostic deterministic Signal-State Machine with atomic trap rollback and reversible trace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Mapping, TypeVar

from vld.canonical import canonical_json, domain_digest
from vld.signal.backends import (
    ComputationalBackend,
    GuardDecision,
    SIGNAL8_OBSERVERS,
    Signal8Backend,
    SignalDomainError,
)
from vld.signal.ir import (
    INSTRUCTION_SCHEMAS,
    Program,
    ProgramFootprintReport,
    analyze_program_footprints,
)
from vld.triad import IndeterminateGuardTrap, TruthDisposition

if TYPE_CHECKING:
    from vld.signal.compiler import TaskSpec

V = TypeVar("V")


class SignalMachineError(RuntimeError):
    """Raised on signal-machine specification, register, memory, or checkpoint violations."""


class MissingRegisterError(SignalMachineError):
    """Raised when an instruction reads an uninitialized register at runtime."""


class MissingMemoryCellError(SignalMachineError):
    """Raised when an instruction reads an absent memory cell at runtime."""


class StepLimitExceededError(SignalMachineError):
    """Raised when machine execution exceeds its explicit step budget without halting."""


class AmbiguousBranchTrap(IndeterminateGuardTrap):
    """Raised when a branch guard evaluates to `TruthDisposition.INDETERMINATE`; rolls back step atomically."""

    def __init__(self, pc: int, decision: GuardDecision) -> None:
        self.pc = pc
        self.decision = decision
        super().__init__(
            f"Indeterminate branch guard trapped atomically at pc={pc}: "
            f"distance={decision.distance:.6g}, uncertainty={decision.uncertainty:.6g}, "
            f"tolerance={decision.tolerance:.6g}."
        )


@dataclass(frozen=True, slots=True)
class SignalGroundSpec:
    """Executable Ground contract binding domain algebra, observer profiles, and allowed opcodes."""

    ground_id: str
    version: str
    domain_id: str
    observer_profiles: tuple[int, ...]
    instruction_set: tuple[str, ...]
    branch_ambiguity_policy: str = "trap"
    assumptions: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ground_id or not self.version or not self.domain_id:
            raise SignalMachineError("ground_id, version, and domain_id must be non-empty strings.")
        observers = tuple(self.observer_profiles)
        instructions = tuple(op.upper() for op in self.instruction_set)
        assumptions = tuple(self.assumptions)
        evidence = tuple(self.evidence)
        if not observers or len(set(observers)) != len(observers):
            raise SignalMachineError("observer_profiles must be non-empty and unique.")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in observers):
            raise SignalMachineError("observer_profiles must contain exact integers.")
        if not instructions or len(set(instructions)) != len(instructions):
            raise SignalMachineError("instruction_set must be non-empty and unique.")
        if any(op not in INSTRUCTION_SCHEMAS for op in instructions):
            raise SignalMachineError(f"Unsupported instruction in GroundSpec: {instructions}")
        if self.branch_ambiguity_policy != "trap":
            raise SignalMachineError("Only branch_ambiguity_policy='trap' is admitted.")
        object.__setattr__(self, "observer_profiles", observers)
        object.__setattr__(self, "instruction_set", instructions)
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "evidence", evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumptions": list(self.assumptions),
            "branch_ambiguity_policy": self.branch_ambiguity_policy,
            "domain_id": self.domain_id,
            "evidence": list(self.evidence),
            "ground_id": self.ground_id,
            "instruction_set": list(self.instruction_set),
            "observer_profiles": list(self.observer_profiles),
            "version": self.version,
        }

    @property
    def fingerprint(self) -> str:
        return domain_digest("vld.signal-ground.v1", self.to_dict())

    @classmethod
    def for_backend(
        cls,
        backend: ComputationalBackend[Any],
        *,
        ground_id: str | None = None,
        version: str = "0.1.0",
    ) -> SignalGroundSpec:
        base_ops = ["LOAD", "STORE", "MOVE", "ADD", "SUB", "BRANCH_EQ", "BRANCH_IDENTITY", "JUMP", "EMIT", "HALT"]
        if backend.capabilities.supports_reframe:
            base_ops.insert(3, "REFRAME")
        if backend.capabilities.supports_mul:
            base_ops.insert(base_ops.index("SUB") + 1, "MUL")
        return cls(
            ground_id=ground_id or f"vld.signal.{backend.domain_id}",
            version=version,
            domain_id=backend.domain_id,
            observer_profiles=backend.observer_profiles,
            instruction_set=tuple(base_ops),
            branch_ambiguity_policy="trap",
            assumptions=(f"Bound to backend domain {backend.domain_id}.",),
            evidence=("Runtime-checked against ComputationalBackend capability contract.",),
        )


SIGNAL8_GROUND = SignalGroundSpec(
    ground_id="vld.signal-state.z8",
    version="0.1.0",
    domain_id="signal.phase.cyclic.z8.v1",
    observer_profiles=SIGNAL8_OBSERVERS,
    instruction_set=(
        "LOAD",
        "STORE",
        "MOVE",
        "REFRAME",
        "ADD",
        "SUB",
        "BRANCH_EQ",
        "BRANCH_IDENTITY",
        "JUMP",
        "EMIT",
        "HALT",
    ),
    branch_ambiguity_policy="trap",
    assumptions=(
        "Value domain is the finite cyclic group Z/8Z encoded as unit complex phasors.",
        "Observer coordinates are the four odd-unit characters {1, 3, 5, 7}.",
        "Indeterminate interval guards trap atomically without mutating state.",
    ),
    evidence=(
        "4,772-case exhaustive Z/8Z algebra, observer-naturality, and conditional verification.",
    ),
)


@dataclass(frozen=True, slots=True)
class Emission(Generic[V]):
    """Labeled terminal emission produced by an `EMIT` instruction."""

    label: str
    value: V


@dataclass(frozen=True, slots=True)
class ExecutionEvent(Generic[V]):
    """Single deterministic execution trace record."""

    step: int
    pc_before: int
    opcode: str
    pc_after: int
    branch_disposition: TruthDisposition | None = None
    branch_distance: float | None = None
    branch_uncertainty: float | None = None
    branch_tolerance: float | None = None
    writes: tuple[str, ...] = ()
    write_values: tuple[tuple[str, V], ...] = ()

    def to_dict(self, backend: ComputationalBackend[V]) -> dict[str, Any]:
        return {
            "branch_disposition": None if self.branch_disposition is None else self.branch_disposition.value,
            "branch_distance": self.branch_distance,
            "branch_tolerance": self.branch_tolerance,
            "branch_uncertainty": self.branch_uncertainty,
            "opcode": self.opcode,
            "pc_after": self.pc_after,
            "pc_before": self.pc_before,
            "step": self.step,
            "write_values": [
                {"location": loc, "value": backend.value_to_dict(val)} for loc, val in self.write_values
            ],
            "writes": list(self.writes),
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult(Generic[V]):
    """Completed execution result bound to program and Ground fingerprints."""

    outputs: tuple[Emission[V], ...]
    trace: tuple[ExecutionEvent[V], ...]
    halted: bool
    final_pc: int
    program_fingerprint: str
    ground_fingerprint: str

    def observed_symbols(self, backend: Any | None = None) -> dict[str, Any]:
        obs_backend = backend or Signal8Backend()
        if hasattr(obs_backend, "observe_symbol"):
            return {item.label: obs_backend.observe_symbol(item.value) for item in self.outputs}
        if hasattr(obs_backend, "observe_rational"):
            return {item.label: obs_backend.observe_rational(item.value) for item in self.outputs}
        raise SignalMachineError(f"Backend {type(obs_backend).__name__} has no terminal observation decoder.")


class VirtualGroundInterpreter(Generic[V]):
    """Parameterized interpreter executing `Program` IR over any `ComputationalBackend[V]`."""

    def __init__(
        self,
        ground: SignalGroundSpec | None = None,
        backend: ComputationalBackend[V] | None = None,
        *,
        enforce_static_footprints: bool = True,
    ) -> None:
        resolved_backend: Any = backend if backend is not None else Signal8Backend()
        resolved_ground = ground if ground is not None else (
            SIGNAL8_GROUND if isinstance(resolved_backend, Signal8Backend) else SignalGroundSpec.for_backend(resolved_backend)
        )
        if resolved_backend.domain_id != resolved_ground.domain_id:
            raise SignalMachineError(
                f"Backend domain {resolved_backend.domain_id!r} != Ground domain {resolved_ground.domain_id!r}."
            )
        if tuple(resolved_backend.observer_profiles) != tuple(resolved_ground.observer_profiles):
            raise SignalMachineError("Backend observer_profiles do not match SignalGroundSpec.")
        if "MUL" in resolved_ground.instruction_set and not resolved_backend.capabilities.supports_mul:
            raise SignalMachineError(f"Ground declares MUL, but backend {resolved_backend.domain_id!r} does not support MUL.")
        if "REFRAME" in resolved_ground.instruction_set and not resolved_backend.capabilities.supports_reframe:
            raise SignalMachineError(f"Ground declares REFRAME, but backend {resolved_backend.domain_id!r} does not support REFRAME.")

        self.ground = resolved_ground
        self.backend: ComputationalBackend[V] = resolved_backend
        self.enforce_static_footprints = enforce_static_footprints

    def instantiate(
        self,
        program: Program,
        initial_memory: Mapping[str, V] | None = None,
    ) -> SignalStateMachine[V]:
        unsupported = program.instruction_set - set(self.ground.instruction_set)
        if unsupported:
            raise SignalMachineError(f"Program uses opcode(s) not admitted by Ground: {sorted(unsupported)}")

        memory = dict(initial_memory or {})
        for address, value in memory.items():
            if not isinstance(address, str) or not address:
                raise SignalMachineError("Memory cell addresses must be non-empty strings.")
            self.backend.validate_value(value)

        footprint_report: ProgramFootprintReport | None = None
        if self.enforce_static_footprints:
            footprint_report = analyze_program_footprints(
                program,
                declared_memory_cells=memory.keys() if initial_memory is not None else None,
            )

        return SignalStateMachine(self, program, memory, footprint_report=footprint_report)

    def compile_task(self, task: TaskSpec) -> Program:
        from vld.signal.compiler import TaskCompiler

        return TaskCompiler().compile(task)

    def interpret(
        self,
        task: TaskSpec,
        initial_memory: Mapping[str, V] | None = None,
        *,
        max_steps: int = 10_000,
    ) -> ExecutionResult[V]:
        prog = self.compile_task(task)
        return self.execute(prog, initial_memory, max_steps=max_steps)

    def execute(
        self,
        program: Program,
        initial_memory: Mapping[str, V] | None = None,
        *,
        max_steps: int = 10_000,
    ) -> ExecutionResult[V]:
        machine = self.instantiate(program, initial_memory)
        return machine.run(max_steps=max_steps)


class SignalStateMachine(Generic[V]):
    """Stateful reversible signal machine with atomic trap rollback and verified checkpoints."""

    SNAPSHOT_SCHEMA = "vld.signal-machine-snapshot.v1"

    def __init__(
        self,
        interpreter: VirtualGroundInterpreter[V],
        program: Program,
        initial_memory: Mapping[str, V],
        *,
        footprint_report: ProgramFootprintReport | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.program = program
        self.footprint_report = footprint_report
        self.initial_memory: dict[str, V] = dict(initial_memory)
        self.memory: dict[str, V] = dict(initial_memory)
        self.registers: dict[str, V] = {}
        self.pc: int = 0
        self.halted: bool = False
        self.outputs: list[Emission[V]] = []
        self.trace: list[ExecutionEvent[V]] = []
        self._undo_stack: list[dict[str, Any]] = []

    def _reg(self, name: str) -> V:
        if name not in self.registers:
            raise MissingRegisterError(f"Register {name!r} read before initialization.")
        return self.registers[name]

    def _capture_local_state(self) -> dict[str, Any]:
        return {
            "halted": self.halted,
            "memory": dict(self.memory),
            "outputs_len": len(self.outputs),
            "pc": self.pc,
            "registers": dict(self.registers),
            "trace_len": len(self.trace),
        }

    def _restore_local_state(self, state: Mapping[str, Any]) -> None:
        self.memory = dict(state["memory"])
        self.registers = dict(state["registers"])
        self.pc = int(state["pc"])
        self.halted = bool(state["halted"])
        del self.outputs[int(state["outputs_len"]) :]
        del self.trace[int(state["trace_len"]) :]

    def step(self) -> ExecutionEvent[V] | None:
        if self.halted:
            return None
        if not (0 <= self.pc < len(self.program.instructions)):
            raise SignalMachineError(f"Program counter {self.pc} escaped instruction bounds.")

        before = self._capture_local_state()
        pc_before = self.pc
        ins = self.program.instructions[self.pc]
        args = ins.args
        backend = self.interpreter.backend

        branch_disp: TruthDisposition | None = None
        branch_dist: float | None = None
        branch_unc: float | None = None
        branch_tol: float | None = None
        writes: list[str] = []
        write_values: list[tuple[str, V]] = []

        try:
            op = ins.opcode
            next_pc = self.pc + 1
            if op == "LOAD":
                addr, dst = args["address"], args["dst"]
                if addr not in self.memory:
                    raise MissingMemoryCellError(f"Memory cell {addr!r} does not exist.")
                val = self.memory[addr]
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "STORE":
                addr, src = args["address"], args["src"]
                val = self._reg(src)
                self.memory[addr] = val
                writes.append(f"memory:{addr}")
                write_values.append((f"memory:{addr}", val))
            elif op == "MOVE":
                dst, src = args["dst"], args["src"]
                val = self._reg(src)
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "REFRAME":
                dst, src = args["dst"], args["src"]
                val = backend.reframe(self._reg(src), int(args["observer"]))
                backend.validate_value(val)
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "ADD":
                dst = args["dst"]
                val = backend.add(self._reg(args["left"]), self._reg(args["right"]))
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "SUB":
                dst = args["dst"]
                val = backend.sub(self._reg(args["left"]), self._reg(args["right"]))
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "MUL":
                dst = args["dst"]
                val = backend.mul(self._reg(args["left"]), self._reg(args["right"]))
                self.registers[dst] = val
                writes.append(f"register:{dst}")
                write_values.append((f"register:{dst}", val))
            elif op == "BRANCH_EQ":
                decision = backend.compare_equal(
                    self._reg(args["left"]),
                    self._reg(args["right"]),
                    float(args["tolerance_rad"]),
                )
                if decision.disposition is TruthDisposition.INDETERMINATE:
                    raise AmbiguousBranchTrap(self.pc, decision)
                branch_disp = decision.disposition
                branch_dist = decision.distance
                branch_unc = decision.uncertainty
                branch_tol = decision.tolerance
                if decision.disposition is TruthDisposition.TRUE:
                    next_pc = int(args["target"])
            elif op == "BRANCH_IDENTITY":
                decision = backend.compare_identity(
                    self._reg(args["src"]),
                    float(args["tolerance_rad"]),
                )
                if decision.disposition is TruthDisposition.INDETERMINATE:
                    raise AmbiguousBranchTrap(self.pc, decision)
                branch_disp = decision.disposition
                branch_dist = decision.distance
                branch_unc = decision.uncertainty
                branch_tol = decision.tolerance
                if decision.disposition is TruthDisposition.TRUE:
                    next_pc = int(args["target"])
            elif op == "JUMP":
                next_pc = int(args["target"])
            elif op == "EMIT":
                src = args["src"]
                lbl = str(args.get("label", src))
                val = self._reg(src)
                self.outputs.append(Emission(lbl, val))
                writes.append(f"output:{lbl}")
                write_values.append((f"output:{lbl}", val))
            elif op == "HALT":
                self.halted = True
            else:
                raise SignalMachineError(f"Unsupported opcode {op!r} at runtime.")

            self.pc = next_pc
            event = ExecutionEvent(
                step=len(self.trace),
                pc_before=pc_before,
                opcode=op,
                pc_after=self.pc,
                branch_disposition=branch_disp,
                branch_distance=branch_dist,
                branch_uncertainty=branch_unc,
                branch_tolerance=branch_tol,
                writes=tuple(writes),
                write_values=tuple(write_values),
            )
            self.trace.append(event)
            self._undo_stack.append(before)
            return event
        except Exception:
            self._restore_local_state(before)
            raise

    def run(self, *, max_steps: int = 10_000, atomic_on_trap: bool = True) -> ExecutionResult[V]:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer.")
        run_checkpoint = self._capture_local_state()
        undo_depth = len(self._undo_stack)
        steps = 0
        try:
            while not self.halted and steps < max_steps:
                self.step()
                steps += 1
        except AmbiguousBranchTrap:
            if atomic_on_trap:
                self._restore_local_state(run_checkpoint)
                del self._undo_stack[undo_depth:]
            raise
        if not self.halted:
            raise StepLimitExceededError(f"Execution exceeded max_steps={max_steps} at pc={self.pc}.")
        return self.result()

    def undo(self) -> ExecutionEvent[V]:
        if not self._undo_stack:
            raise SignalMachineError("Undo stack is empty.")
        last_event = self.trace[-1]
        self._restore_local_state(self._undo_stack.pop())
        return last_event

    def reset(self) -> None:
        self.memory = dict(self.initial_memory)
        self.registers.clear()
        self.pc = 0
        self.halted = False
        self.outputs.clear()
        self.trace.clear()
        self._undo_stack.clear()

    def result(self) -> ExecutionResult[V]:
        return ExecutionResult(
            outputs=tuple(self.outputs),
            trace=tuple(self.trace),
            halted=self.halted,
            final_pc=self.pc,
            program_fingerprint=self.program.fingerprint,
            ground_fingerprint=self.interpreter.ground.fingerprint,
        )

    def snapshot(self) -> dict[str, Any]:
        backend = self.interpreter.backend
        body = {
            "domain_id": backend.domain_id,
            "ground_fingerprint": self.interpreter.ground.fingerprint,
            "halted": self.halted,
            "initial_memory": {k: backend.value_to_dict(v) for k, v in sorted(self.initial_memory.items())},
            "memory": {k: backend.value_to_dict(v) for k, v in sorted(self.memory.items())},
            "outputs": [{"label": em.label, "value": backend.value_to_dict(em.value)} for em in self.outputs],
            "pc": self.pc,
            "program_fingerprint": self.program.fingerprint,
            "registers": {k: backend.value_to_dict(v) for k, v in sorted(self.registers.items())},
            "schema": self.SNAPSHOT_SCHEMA,
            "trace": [ev.to_dict(backend) for ev in self.trace],
        }
        state_digest = domain_digest(self.SNAPSHOT_SCHEMA, body)
        return {**body, "state_digest": state_digest}

    @classmethod
    def restore(
        cls,
        interpreter: VirtualGroundInterpreter[V],
        program: Program,
        snapshot: Mapping[str, Any],
    ) -> SignalStateMachine[V]:
        if not isinstance(snapshot, Mapping) or snapshot.get("schema") != cls.SNAPSHOT_SCHEMA:
            raise SignalMachineError("Unsupported machine snapshot schema.")
        supplied_digest = snapshot.get("state_digest")
        body = {k: v for k, v in snapshot.items() if k != "state_digest"}
        expected_digest = domain_digest(cls.SNAPSHOT_SCHEMA, body)
        if supplied_digest != expected_digest:
            raise SignalMachineError("Machine snapshot state_digest mismatch.")
        if snapshot.get("program_fingerprint") != program.fingerprint:
            raise SignalMachineError("Machine snapshot belongs to a different Program.")
        if snapshot.get("ground_fingerprint") != interpreter.ground.fingerprint:
            raise SignalMachineError("Machine snapshot belongs to a different SignalGroundSpec.")
        if snapshot.get("domain_id") != interpreter.backend.domain_id:
            raise SignalMachineError("Machine snapshot belongs to a different backend domain.")

        backend = interpreter.backend
        try:
            init_mem = {k: backend.value_from_dict(v) for k, v in snapshot["initial_memory"].items()}
            mem = {k: backend.value_from_dict(v) for k, v in snapshot["memory"].items()}
            regs = {k: backend.value_from_dict(v) for k, v in snapshot["registers"].items()}
            outputs = [Emission(str(item["label"]), backend.value_from_dict(item["value"])) for item in snapshot["outputs"]]
            trace = [
                ExecutionEvent(
                    step=int(ev["step"]),
                    pc_before=int(ev["pc_before"]),
                    opcode=str(ev["opcode"]),
                    pc_after=int(ev["pc_after"]),
                    branch_disposition=(
                        None if ev.get("branch_disposition") is None else TruthDisposition(ev["branch_disposition"])
                    ),
                    branch_distance=ev.get("branch_distance"),
                    branch_uncertainty=ev.get("branch_uncertainty"),
                    branch_tolerance=ev.get("branch_tolerance"),
                    writes=tuple(ev.get("writes", ())),
                    write_values=tuple(
                        (str(w["location"]), backend.value_from_dict(w["value"])) for w in ev.get("write_values", ())
                    ),
                )
                for ev in snapshot["trace"]
            ]
            pc = int(snapshot["pc"])
            halted = bool(snapshot["halted"])
        except (KeyError, TypeError, ValueError, SignalDomainError) as exc:
            raise SignalMachineError(f"Malformed machine snapshot payload: {exc}") from exc

        if not (0 <= pc <= len(program.instructions)):
            raise SignalMachineError("Snapshot pc out of bounds.")
        machine = cls(interpreter, program, init_mem)
        machine.memory = mem
        machine.registers = regs
        machine.outputs = outputs
        machine.trace = trace
        machine.pc = pc
        machine.halted = halted
        _ = canonical_json(snapshot)
        return machine
