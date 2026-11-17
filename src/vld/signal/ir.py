r"""Typed signal-machine IR, instruction footprints, and static composition analysis.

Mathematical model:
Every instruction $\iota_k$ exposes an explicit footprint tuple:
  $$\operatorname{fp}(\iota_k) = (\mathcal{R}_{\text{reg}}, \mathcal{W}_{\text{reg}}, \mathcal{R}_{\text{mem}}, \mathcal{W}_{\text{mem}}, \mathcal{E}_{\text{out}}, \text{may\_trap}, \text{is\_reversible})$$

Over the control-flow graph $G = (V, E)$ of a `Program`, definite register initialization
satisfies the forward dataflow fixed-point equations:
  $$\mathcal{D}_{\text{in}}[0] = \emptyset, \qquad \mathcal{D}_{\text{in}}[k] = \bigcap_{p \in \operatorname{pred}(k)} \mathcal{D}_{\text{out}}[p] \quad (k > 0 \text{ reachable})$$
  $$\mathcal{D}_{\text{out}}[k] = \mathcal{D}_{\text{in}}[k] \cup \mathcal{W}_{\text{reg}}(\iota_k)$$

A program is **statically footprint-valid** iff for every reachable instruction $k$:
  1. $\mathcal{R}_{\text{reg}}(\iota_k) \subseteq \mathcal{D}_{\text{in}}[k]$ (no uninitialized register reads on any path),
  2. Every control fallthrough stays within $[0, |V| - 1]$ (no fallthrough past end of instruction stream),
  3. When `declared_memory_cells` is supplied, every memory read $m \in \mathcal{R}_{\text{mem}}(\iota_k)$ belongs
     either to `declared_memory_cells` or to cells definitely written along all incoming paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from vld.canonical import canonical_json, domain_digest, parse_strict_json


class ProgramError(ValueError):
    """Raised when an instruction or program violates IR schema invariants."""


class StaticFootprintError(ProgramError):
    """Raised when static control-flow or footprint composition analysis fails."""


INSTRUCTION_SCHEMAS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "LOAD": (frozenset({"dst", "address"}), frozenset()),
    "STORE": (frozenset({"src", "address"}), frozenset()),
    "MOVE": (frozenset({"dst", "src"}), frozenset()),
    "REFRAME": (frozenset({"dst", "src", "observer"}), frozenset()),
    "ADD": (frozenset({"dst", "left", "right"}), frozenset()),
    "SUB": (frozenset({"dst", "left", "right"}), frozenset()),
    "MUL": (frozenset({"dst", "left", "right"}), frozenset()),
    "BRANCH_EQ": (frozenset({"left", "right", "target", "tolerance_rad"}), frozenset()),
    "BRANCH_IDENTITY": (frozenset({"src", "target", "tolerance_rad"}), frozenset()),
    "JUMP": (frozenset({"target"}), frozenset()),
    "EMIT": (frozenset({"src"}), frozenset({"label"})),
    "HALT": (frozenset(), frozenset()),
}

_REGISTER_ARGS = frozenset({"dst", "src", "left", "right"})
_ADDRESS_ARGS = frozenset({"address"})


@dataclass(frozen=True, slots=True)
class InstructionFootprint:
    """Explicit read/write/emission/trap/reversibility contract of a single IR instruction."""

    reg_reads: frozenset[str]
    reg_writes: frozenset[str]
    mem_reads: frozenset[str]
    mem_writes: frozenset[str]
    emissions: frozenset[str]
    may_trap_indeterminate: bool
    is_step_reversible: bool


@dataclass(frozen=True, slots=True)
class Instruction:
    """Immutable, schema-validated IR instruction with statically derivable footprint."""

    opcode: str
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.opcode, str) or not self.opcode:
            raise ProgramError("opcode must be a non-empty string.")
        op = self.opcode.upper()
        object.__setattr__(self, "opcode", op)
        if op not in INSTRUCTION_SCHEMAS:
            raise ProgramError(f"Unsupported opcode: {op!r}.")
        required, optional = INSTRUCTION_SCHEMAS[op]
        keys = set(self.args.keys())
        missing = required - keys
        extra = keys - required - optional
        if missing:
            raise ProgramError(f"{op} missing required argument(s): {sorted(missing)}")
        if extra:
            raise ProgramError(f"{op} received unknown argument(s): {sorted(extra)}")

        clean = dict(self.args)
        for reg_key in _REGISTER_ARGS & keys:
            if not isinstance(clean[reg_key], str) or not clean[reg_key]:
                raise ProgramError(f"{op}.{reg_key} must be a non-empty register name.")
        for addr_key in _ADDRESS_ARGS & keys:
            if not isinstance(clean[addr_key], str) or not clean[addr_key]:
                raise ProgramError(f"{op}.{addr_key} must be a non-empty memory address.")
        if "target" in clean:
            if isinstance(clean["target"], bool) or not isinstance(clean["target"], int):
                raise ProgramError(f"{op}.target must be an integer instruction index.")
        if "observer" in clean:
            if isinstance(clean["observer"], bool) or not isinstance(clean["observer"], int):
                raise ProgramError("REFRAME.observer must be an integer observer coordinate.")
        if "tolerance_rad" in clean:
            tol = clean["tolerance_rad"]
            if isinstance(tol, bool) or not isinstance(tol, (int, float)):
                raise ProgramError(f"{op}.tolerance_rad must be numeric.")
            tol_f = float(tol)
            if not math.isfinite(tol_f) or not (0.0 <= tol_f <= math.pi):
                raise ProgramError(f"{op}.tolerance_rad must lie in [0, pi].")
            clean["tolerance_rad"] = tol_f
        if "label" in clean and (not isinstance(clean["label"], str) or not clean["label"]):
            raise ProgramError("EMIT.label must be a non-empty string.")
        object.__setattr__(self, "args", MappingProxyType(clean))

    @property
    def footprint(self) -> InstructionFootprint:
        op = self.opcode
        a = self.args
        if op == "LOAD":
            return InstructionFootprint(
                reg_reads=frozenset(),
                reg_writes=frozenset({a["dst"]}),
                mem_reads=frozenset({a["address"]}),
                mem_writes=frozenset(),
                emissions=frozenset(),
                may_trap_indeterminate=False,
                is_step_reversible=True,
            )
        if op == "STORE":
            return InstructionFootprint(
                reg_reads=frozenset({a["src"]}),
                reg_writes=frozenset(),
                mem_reads=frozenset(),
                mem_writes=frozenset({a["address"]}),
                emissions=frozenset(),
                may_trap_indeterminate=False,
                is_step_reversible=True,
            )
        if op in ("MOVE", "REFRAME"):
            return InstructionFootprint(
                reg_reads=frozenset({a["src"]}),
                reg_writes=frozenset({a["dst"]}),
                mem_reads=frozenset(),
                mem_writes=frozenset(),
                emissions=frozenset(),
                may_trap_indeterminate=False,
                is_step_reversible=True,
            )
        if op in ("ADD", "SUB", "MUL"):
            return InstructionFootprint(
                reg_reads=frozenset({a["left"], a["right"]}),
                reg_writes=frozenset({a["dst"]}),
                mem_reads=frozenset(),
                mem_writes=frozenset(),
                emissions=frozenset(),
                may_trap_indeterminate=False,
                is_step_reversible=True,
            )
        if op == "BRANCH_EQ":
            return InstructionFootprint(
                reg_reads=frozenset({a["left"], a["right"]}),
                reg_writes=frozenset(),
                mem_reads=frozenset(),
                mem_writes=frozenset(),
                emissions=frozenset(),
                may_trap_indeterminate=True,
                is_step_reversible=True,
            )
        if op == "BRANCH_IDENTITY":
            return InstructionFootprint(
                reg_reads=frozenset({a["src"]}),
                reg_writes=frozenset(),
                mem_reads=frozenset(),
                mem_writes=frozenset(),
                emissions=frozenset(),
                may_trap_indeterminate=True,
                is_step_reversible=True,
            )
        if op == "EMIT":
            lbl = str(a.get("label", a["src"]))
            return InstructionFootprint(
                reg_reads=frozenset({a["src"]}),
                reg_writes=frozenset(),
                mem_reads=frozenset(),
                mem_writes=frozenset(),
                emissions=frozenset({lbl}),
                may_trap_indeterminate=False,
                is_step_reversible=True,
            )
        # JUMP or HALT
        return InstructionFootprint(
            reg_reads=frozenset(),
            reg_writes=frozenset(),
            mem_reads=frozenset(),
            mem_writes=frozenset(),
            emissions=frozenset(),
            may_trap_indeterminate=False,
            is_step_reversible=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.opcode, **dict(self.args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Instruction:
        if not isinstance(data, Mapping) or "op" not in data:
            raise ProgramError("Instruction dictionary must contain 'op'.")
        return cls(str(data["op"]), {k: v for k, v in data.items() if k != "op"})


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_metadata(v) for v in value)
    return value


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw_metadata(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class ProgramFootprintReport:
    """Aggregate static footprint and control-flow analysis report for a `Program`."""

    reachable_pcs: tuple[int, ...]
    unreachable_pcs: tuple[int, ...]
    mem_reads: frozenset[str]
    mem_writes: frozenset[str]
    emissions: frozenset[str]
    may_trap_pcs: tuple[int, ...]
    definite_registers_at_halt: frozenset[str]


@dataclass(frozen=True, slots=True)
class Program:
    """Content-addressed signal-state IR program."""

    name: str
    instructions: tuple[Instruction, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProgramError("Program name must be a non-empty string.")
        insts = tuple(self.instructions)
        if not insts:
            raise ProgramError("Program must contain at least one instruction.")
        if any(not isinstance(ins, Instruction) for ins in insts):
            raise ProgramError("All program instructions must be Instruction instances.")
        if not isinstance(self.metadata, Mapping):
            raise ProgramError("Program metadata must be a mapping.")
        norm_meta = parse_strict_json(canonical_json(dict(self.metadata)))
        n = len(insts)
        for idx, ins in enumerate(insts):
            if "target" in ins.args and not (0 <= ins.args["target"] < n):
                raise ProgramError(f"Instruction {idx} target {ins.args['target']} outside [0, {n - 1}].")
        if not any(ins.opcode == "HALT" for ins in insts):
            raise ProgramError("Program must contain at least one HALT instruction.")
        object.__setattr__(self, "instructions", insts)
        object.__setattr__(self, "metadata", _freeze_metadata(norm_meta))

    @property
    def instruction_set(self) -> frozenset[str]:
        return frozenset(ins.opcode for ins in self.instructions)

    @property
    def fingerprint(self) -> str:
        return domain_digest("vld.signal-program.v1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "instructions": [ins.to_dict() for ins in self.instructions],
            "metadata": _thaw_metadata(self.metadata),
            "name": self.name,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Program:
        if not isinstance(data, Mapping) or "name" not in data or "instructions" not in data:
            raise ProgramError("Program dictionary requires 'name' and 'instructions'.")
        raw_insts = data["instructions"]
        if not isinstance(raw_insts, (list, tuple)):
            raise ProgramError("Program 'instructions' must be a list.")
        return cls(
            name=str(data["name"]),
            instructions=tuple(Instruction.from_dict(item) for item in raw_insts),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> Program:
        parsed = parse_strict_json(raw)
        if not isinstance(parsed, dict):
            raise ProgramError("Program JSON root must be an object.")
        return cls.from_dict(parsed)


def analyze_program_footprints(
    program: Program,
    *,
    declared_memory_cells: Iterable[str] | None = None,
    forbid_unreachable: bool = False,
) -> ProgramFootprintReport:
    """Statically verify control-flow safety, definite register initialization, and memory footprints."""
    n = len(program.instructions)
    succs: dict[int, tuple[int, ...]] = {}
    for pc, ins in enumerate(program.instructions):
        op = ins.opcode
        if op == "HALT":
            succs[pc] = ()
        elif op == "JUMP":
            succs[pc] = (int(ins.args["target"]),)
        elif op in ("BRANCH_EQ", "BRANCH_IDENTITY"):
            if pc + 1 >= n:
                raise StaticFootprintError(f"Branch instruction at pc={pc} falls through past end of program.")
            succs[pc] = tuple(dict.fromkeys((pc + 1, int(ins.args["target"]))))
        else:
            if pc + 1 >= n:
                raise StaticFootprintError(f"Instruction {op} at pc={pc} falls through past end of program without HALT.")
            succs[pc] = (pc + 1,)

    # 1. Reachability from pc = 0
    reachable: set[int] = {0}
    work = [0]
    while work:
        curr = work.pop()
        for nxt in succs[curr]:
            if nxt not in reachable:
                reachable.add(nxt)
                work.append(nxt)

    unreachable = tuple(pc for pc in range(n) if pc not in reachable)
    if forbid_unreachable and unreachable:
        raise StaticFootprintError(f"Program {program.name!r} contains unreachable instruction(s) at pc={unreachable}.")

    if not any(program.instructions[pc].opcode == "HALT" for pc in reachable):
        raise StaticFootprintError(f"Program {program.name!r} has no reachable HALT instruction.")

    preds: dict[int, list[int]] = {pc: [] for pc in reachable}
    for pc in reachable:
        for nxt in succs[pc]:
            if nxt in reachable:
                preds[nxt].append(pc)

    # 2. Forward dataflow fixed-point for definite registers & definite memory cells
    all_regs: frozenset[str] = frozenset(
        r for pc in reachable for r in (program.instructions[pc].footprint.reg_writes | program.instructions[pc].footprint.reg_reads)
    )
    seed_mem: frozenset[str] = frozenset(declared_memory_cells) if declared_memory_cells is not None else frozenset()
    all_mem: frozenset[str] = seed_mem | frozenset(
        m for pc in reachable for m in (program.instructions[pc].footprint.mem_writes | program.instructions[pc].footprint.mem_reads)
    )

    reg_in: dict[int, frozenset[str]] = {pc: (frozenset() if pc == 0 else all_regs) for pc in reachable}
    reg_out: dict[int, frozenset[str]] = {
        pc: ((frozenset() if pc == 0 else all_regs) | program.instructions[pc].footprint.reg_writes)
        for pc in reachable
    }
    mem_in: dict[int, frozenset[str]] = {pc: (seed_mem if pc == 0 else all_mem) for pc in reachable}
    mem_out: dict[int, frozenset[str]] = {
        pc: ((seed_mem if pc == 0 else all_mem) | program.instructions[pc].footprint.mem_writes)
        for pc in reachable
    }

    changed = True
    while changed:
        changed = False
        for pc in sorted(reachable):
            new_r_in = frozenset() if pc == 0 and not preds[0] else (
                (frozenset() if pc == 0 else all_regs).intersection(*(reg_out[p] for p in preds[pc]))
                if preds[pc]
                else (frozenset() if pc == 0 else all_regs)
            )
            if pc == 0 and preds[0]:
                # Entry pc=0 has an implicit external entry edge with empty registers!
                new_r_in = frozenset()
            new_r_out = new_r_in | program.instructions[pc].footprint.reg_writes

            if pc == 0:
                new_m_in = seed_mem
            else:
                new_m_in = all_mem.intersection(*(mem_out[p] for p in preds[pc]))
            new_m_out = new_m_in | program.instructions[pc].footprint.mem_writes

            if new_r_in != reg_in[pc] or new_r_out != reg_out[pc] or new_m_in != mem_in[pc] or new_m_out != mem_out[pc]:
                reg_in[pc] = new_r_in
                reg_out[pc] = new_r_out
                mem_in[pc] = new_m_in
                mem_out[pc] = new_m_out
                changed = True

    # 3. Validate preconditions at every reachable instruction
    mem_reads_agg: set[str] = set()
    mem_writes_agg: set[str] = set()
    emissions_agg: set[str] = set()
    may_trap_pcs: list[int] = []
    halt_regs: list[frozenset[str]] = []

    for pc in sorted(reachable):
        fp = program.instructions[pc].footprint
        missing_regs = fp.reg_reads - reg_in[pc]
        if missing_regs:
            raise StaticFootprintError(
                f"Instruction {program.instructions[pc].opcode} at pc={pc} reads uninitialized register(s): {sorted(missing_regs)}"
            )
        if declared_memory_cells is not None:
            missing_mem = fp.mem_reads - mem_in[pc]
            if missing_mem:
                raise StaticFootprintError(
                    f"Instruction {program.instructions[pc].opcode} at pc={pc} reads undeclared memory cell(s): {sorted(missing_mem)}"
                )
        mem_reads_agg.update(fp.mem_reads)
        mem_writes_agg.update(fp.mem_writes)
        emissions_agg.update(fp.emissions)
        if fp.may_trap_indeterminate:
            may_trap_pcs.append(pc)
        if program.instructions[pc].opcode == "HALT":
            halt_regs.append(reg_in[pc])

    definite_halt = all_regs.intersection(*halt_regs) if halt_regs else frozenset()
    return ProgramFootprintReport(
        reachable_pcs=tuple(sorted(reachable)),
        unreachable_pcs=unreachable,
        mem_reads=frozenset(mem_reads_agg),
        mem_writes=frozenset(mem_writes_agg),
        emissions=frozenset(emissions_agg),
        may_trap_pcs=tuple(may_trap_pcs),
        definite_registers_at_halt=definite_halt,
    )


def compose_sequential_programs(
    first: Program,
    second: Program,
    *,
    name: str | None = None,
    declared_memory_cells: Iterable[str] | None = None,
) -> Program:
    """Safely compose `first ; second` by rewriting `first`'s HALT sites to jump to `second`'s entry.

    Validates both component footprints and ensures `second` does not rely on leaked scratch registers
    unless every path initialized them, rewriting all internal branch/jump targets of `second` by `+offset`.
    """
    analyze_program_footprints(first, declared_memory_cells=declared_memory_cells)
    # Validate `second` standalone assuming memory cells written by `first` are available if declared_memory_cells given
    second_seed_mem: set[str] | None = None
    if declared_memory_cells is not None:
        first_report = analyze_program_footprints(first, declared_memory_cells=declared_memory_cells)
        second_seed_mem = set(declared_memory_cells) | set(first_report.mem_writes)
    analyze_program_footprints(second, declared_memory_cells=second_seed_mem)

    offset = len(first.instructions)
    combined: list[Instruction] = []
    for ins in first.instructions:
        if ins.opcode == "HALT":
            combined.append(Instruction("JUMP", {"target": offset}))
        else:
            combined.append(ins)

    for ins in second.instructions:
        if "target" in ins.args:
            patched = dict(ins.args)
            patched["target"] = int(patched["target"]) + offset
            combined.append(Instruction(ins.opcode, patched))
        else:
            combined.append(ins)

    composed = Program(
        name=name or f"{first.name}__then__{second.name}",
        instructions=tuple(combined),
        metadata={
            "composed_from": [first.fingerprint, second.fingerprint],
            "kind": "vld.sequential-composition.v1",
        },
    )
    analyze_program_footprints(composed, declared_memory_cells=declared_memory_cells)
    return composed
