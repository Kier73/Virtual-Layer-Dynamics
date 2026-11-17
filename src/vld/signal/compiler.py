r"""Declarative task-expression compiler lowering typed ASTs into `vld.signal` IR."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, TypeAlias

from vld.canonical import canonical_json, parse_strict_json
from vld.signal.ir import Instruction, Program


class TaskCompileError(ValueError):
    """Raised when a declarative `TaskSpec` violates typing or scoping rules."""


@dataclass(frozen=True, slots=True)
class InputExpr:
    address: str

    def __post_init__(self) -> None:
        if not isinstance(self.address, str) or not self.address:
            raise TaskCompileError("InputExpr address must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class AddExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class SubExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class MulExpr:
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class ReframeExpr:
    value: Expr
    observer: int

    def __post_init__(self) -> None:
        if isinstance(self.observer, bool) or not isinstance(self.observer, int):
            raise TaskCompileError("ReframeExpr observer must be an integer coordinate.")


@dataclass(frozen=True, slots=True)
class IdentityGuard:
    value: Expr
    tolerance_rad: float

    def __post_init__(self) -> None:
        if isinstance(self.tolerance_rad, bool) or not isinstance(self.tolerance_rad, (int, float)):
            raise TaskCompileError("IdentityGuard tolerance_rad must be numeric.")
        if not math.isfinite(float(self.tolerance_rad)) or not (0.0 <= float(self.tolerance_rad) <= math.pi):
            raise TaskCompileError("IdentityGuard tolerance_rad must lie in [0, pi].")


@dataclass(frozen=True, slots=True)
class EqualityGuard:
    left: Expr
    right: Expr
    tolerance_rad: float

    def __post_init__(self) -> None:
        if isinstance(self.tolerance_rad, bool) or not isinstance(self.tolerance_rad, (int, float)):
            raise TaskCompileError("EqualityGuard tolerance_rad must be numeric.")
        if not math.isfinite(float(self.tolerance_rad)) or not (0.0 <= float(self.tolerance_rad) <= math.pi):
            raise TaskCompileError("EqualityGuard tolerance_rad must lie in [0, pi].")


Guard: TypeAlias = IdentityGuard | EqualityGuard


@dataclass(frozen=True, slots=True)
class ConditionalExpr:
    guard: Guard
    when_true: Expr
    when_false: Expr


Expr: TypeAlias = InputExpr | AddExpr | SubExpr | MulExpr | ReframeExpr | ConditionalExpr


@dataclass(frozen=True, slots=True)
class OutputSpec:
    label: str
    expression: Expr

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise TaskCompileError("OutputSpec label must be a non-empty string.")


def _referenced_inputs(expr: Expr) -> set[str]:
    if isinstance(expr, InputExpr):
        return {expr.address}
    if isinstance(expr, (AddExpr, SubExpr, MulExpr)):
        return _referenced_inputs(expr.left) | _referenced_inputs(expr.right)
    if isinstance(expr, ReframeExpr):
        return _referenced_inputs(expr.value)
    if isinstance(expr, ConditionalExpr):
        guard_refs = (
            _referenced_inputs(expr.guard.value)
            if isinstance(expr.guard, IdentityGuard)
            else (_referenced_inputs(expr.guard.left) | _referenced_inputs(expr.guard.right))
        )
        return guard_refs | _referenced_inputs(expr.when_true) | _referenced_inputs(expr.when_false)
    raise TaskCompileError(f"Unsupported expression node: {type(expr).__name__}")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Typed declarative expression specification lowered into `Program` IR."""

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[OutputSpec, ...]

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        if not isinstance(self.name, str) or not self.name:
            raise TaskCompileError("TaskSpec name must be a non-empty string.")
        if any(not isinstance(x, str) or not x for x in inputs) or len(set(inputs)) != len(inputs):
            raise TaskCompileError("TaskSpec inputs must be unique non-empty strings.")
        if not outputs or any(not isinstance(o, OutputSpec) for o in outputs):
            raise TaskCompileError("TaskSpec requires at least one OutputSpec.")
        labels = [o.label for o in outputs]
        if len(set(labels)) != len(labels):
            raise TaskCompileError("OutputSpec labels must be unique.")
        collisions = {f"output.{lbl}" for lbl in labels} & set(inputs)
        if collisions:
            raise TaskCompileError(f"Input addresses collide with reserved output cells: {sorted(collisions)}")
        refs: set[str] = set()
        for out in outputs:
            refs.update(_referenced_inputs(out.expression))
        missing = refs - set(inputs)
        if missing:
            raise TaskCompileError(f"Expressions reference undeclared input(s): {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": list(self.inputs),
            "name": self.name,
            "outputs": [{"expression": _expr_to_dict(o.expression), "label": o.label} for o in self.outputs],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TaskSpec:
        if not isinstance(data, Mapping) or set(data.keys()) != {"name", "inputs", "outputs"}:
            raise TaskCompileError("TaskSpec requires exact keys {'name', 'inputs', 'outputs'}.")
        if not isinstance(data["inputs"], (list, tuple)) or not isinstance(data["outputs"], (list, tuple)):
            raise TaskCompileError("TaskSpec inputs and outputs must be lists.")
        outs: list[OutputSpec] = []
        for item in data["outputs"]:
            if not isinstance(item, Mapping) or set(item.keys()) != {"label", "expression"}:
                raise TaskCompileError("OutputSpec entry requires {'label', 'expression'}.")
            outs.append(OutputSpec(str(item["label"]), _parse_expr(item["expression"])))
        return cls(str(data["name"]), tuple(data["inputs"]), tuple(outs))

    @classmethod
    def from_json(cls, raw: str | bytes) -> TaskSpec:
        parsed = parse_strict_json(raw)
        if not isinstance(parsed, dict):
            raise TaskCompileError("TaskSpec JSON root must be an object.")
        return cls.from_dict(parsed)


def _expr_to_dict(expr: Expr) -> dict[str, Any]:
    if isinstance(expr, InputExpr):
        return {"address": expr.address, "type": "input"}
    if isinstance(expr, AddExpr):
        return {"left": _expr_to_dict(expr.left), "right": _expr_to_dict(expr.right), "type": "add"}
    if isinstance(expr, SubExpr):
        return {"left": _expr_to_dict(expr.left), "right": _expr_to_dict(expr.right), "type": "sub"}
    if isinstance(expr, MulExpr):
        return {"left": _expr_to_dict(expr.left), "right": _expr_to_dict(expr.right), "type": "mul"}
    if isinstance(expr, ReframeExpr):
        return {"observer": expr.observer, "type": "reframe", "value": _expr_to_dict(expr.value)}
    if isinstance(expr, ConditionalExpr):
        guard_dict = (
            {"tolerance_rad": float(expr.guard.tolerance_rad), "type": "is_identity", "value": _expr_to_dict(expr.guard.value)}
            if isinstance(expr.guard, IdentityGuard)
            else {
                "left": _expr_to_dict(expr.guard.left),
                "right": _expr_to_dict(expr.guard.right),
                "tolerance_rad": float(expr.guard.tolerance_rad),
                "type": "equal",
            }
        )
        return {
            "guard": guard_dict,
            "type": "if",
            "when_false": _expr_to_dict(expr.when_false),
            "when_true": _expr_to_dict(expr.when_true),
        }
    raise TaskCompileError(f"Unsupported expression: {type(expr).__name__}")


def _parse_expr(data: Any) -> Expr:
    if not isinstance(data, Mapping) or "type" not in data:
        raise TaskCompileError("Expression must be an object with 'type'.")
    kind = data["type"]
    if kind == "input" and set(data.keys()) == {"type", "address"}:
        return InputExpr(data["address"])
    if kind in ("add", "sub", "mul") and set(data.keys()) == {"type", "left", "right"}:
        cls = {"add": AddExpr, "sub": SubExpr, "mul": MulExpr}[kind]
        return cls(_parse_expr(data["left"]), _parse_expr(data["right"]))
    if kind == "reframe" and set(data.keys()) == {"type", "value", "observer"}:
        return ReframeExpr(_parse_expr(data["value"]), data["observer"])
    if kind == "if" and set(data.keys()) == {"type", "guard", "when_true", "when_false"}:
        g = data["guard"]
        if not isinstance(g, Mapping) or "type" not in g:
            raise TaskCompileError("Guard must be an object with 'type'.")
        if g["type"] == "is_identity" and set(g.keys()) == {"type", "value", "tolerance_rad"}:
            guard: Guard = IdentityGuard(_parse_expr(g["value"]), g["tolerance_rad"])
        elif g["type"] == "equal" and set(g.keys()) == {"type", "left", "right", "tolerance_rad"}:
            guard = EqualityGuard(_parse_expr(g["left"]), _parse_expr(g["right"]), g["tolerance_rad"])
        else:
            raise TaskCompileError(f"Invalid guard specification: {g!r}")
        return ConditionalExpr(guard, _parse_expr(data["when_true"]), _parse_expr(data["when_false"]))
    raise TaskCompileError(f"Invalid or unsupported expression specification: {data!r}")


class TaskCompiler:
    """Lower declarative `TaskSpec` ASTs into statically footprint-safe `Program` IR."""

    def compile(self, task: TaskSpec) -> Program:
        if not isinstance(task, TaskSpec):
            raise TaskCompileError("TaskCompiler.compile requires a TaskSpec.")
        self._insts: list[tuple[str, dict[str, Any]]] = []
        self._reg_counter = 0
        # Evaluate all expressions (and any indeterminate branch guards) BEFORE committing STORE / EMIT side effects!
        evaluated_outputs = [(out.label, self._lower_expr(out.expression)) for out in task.outputs]
        for label, reg in evaluated_outputs:
            self._emit("STORE", src=reg, address=f"output.{label}")
            self._emit("EMIT", src=reg, label=label)
        self._emit("HALT")
        return Program(
            name=task.name,
            instructions=tuple(Instruction(op, args) for op, args in self._insts),
            metadata={
                "declared_inputs": list(task.inputs),
                "declared_outputs": [o.label for o in task.outputs],
                "source": "vld.task-compiler.v1",
            },
        )

    def _fresh_reg(self) -> str:
        name = f"t{self._reg_counter}"
        self._reg_counter += 1
        return name

    def _emit(self, opcode: str, **args: Any) -> int:
        self._insts.append((opcode, args))
        return len(self._insts) - 1

    def _patch(self, idx: int, target: int) -> None:
        op, args = self._insts[idx]
        patched = dict(args)
        patched["target"] = target
        self._insts[idx] = (op, patched)

    def _lower_expr(self, expr: Expr) -> str:
        if isinstance(expr, InputExpr):
            dst = self._fresh_reg()
            self._emit("LOAD", dst=dst, address=expr.address)
            return dst
        if isinstance(expr, (AddExpr, SubExpr, MulExpr)):
            left = self._lower_expr(expr.left)
            right = self._lower_expr(expr.right)
            dst = self._fresh_reg()
            op = "ADD" if isinstance(expr, AddExpr) else ("SUB" if isinstance(expr, SubExpr) else "MUL")
            self._emit(op, dst=dst, left=left, right=right)
            return dst
        if isinstance(expr, ReframeExpr):
            src = self._lower_expr(expr.value)
            dst = self._fresh_reg()
            self._emit("REFRAME", dst=dst, src=src, observer=expr.observer)
            return dst
        if isinstance(expr, ConditionalExpr):
            if isinstance(expr.guard, IdentityGuard):
                g_reg = self._lower_expr(expr.guard.value)
                branch_idx = self._emit("BRANCH_IDENTITY", src=g_reg, target=-1, tolerance_rad=float(expr.guard.tolerance_rad))
            else:
                gl = self._lower_expr(expr.guard.left)
                gr = self._lower_expr(expr.guard.right)
                branch_idx = self._emit("BRANCH_EQ", left=gl, right=gr, target=-1, tolerance_rad=float(expr.guard.tolerance_rad))

            res_reg = self._fresh_reg()
            false_reg = self._lower_expr(expr.when_false)
            self._emit("MOVE", dst=res_reg, src=false_reg)
            jump_end = self._emit("JUMP", target=-1)

            true_pc = len(self._insts)
            true_reg = self._lower_expr(expr.when_true)
            self._emit("MOVE", dst=res_reg, src=true_reg)
            join_pc = len(self._insts)

            self._patch(branch_idx, true_pc)
            self._patch(jump_end, join_pc)
            return res_reg
        raise TaskCompileError(f"Unsupported expression node: {type(expr).__name__}")
