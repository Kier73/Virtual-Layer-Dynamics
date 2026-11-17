r"""Partial predictive Mealy transition Grounds and multi-digit base-8 Signal8 compiler.

Mathematical model:
For a complete deterministic finite transition Ground $\mathcal{T} = (\Sigma, A, s_0, C, \delta, \lambda)$
with $|\Sigma|$ states and $|A|$ actions, state and action indices are embedded into little-endian
base-8 coordinate vectors of widths:
  $$w_\Sigma = \max\!\left(1, \lceil \log_8 |\Sigma| \rceil\right), \qquad w_A = \max\!\left(1, \lceil \log_8 |A| \rceil\right)$$
so arbitrary finite cardinalities lower faithfully onto `Signal8Backend` ($\mathbb{Z}/8\mathbb{Z}$) without
mutating the underlying 8-PSK character algebra (`G01` resolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.signal.backends import Signal8Backend, SignalPhasorValue
from vld.signal.ir import Instruction, Program
from vld.signal.machine import VirtualGroundInterpreter


class TransitionSystemError(ValueError):
    """Raised on malformed or unresolved transition-system queries."""


class UnknownTransitionError(TransitionSystemError):
    def __init__(self, source: str, action: str) -> None:
        self.source = source
        self.action = action
        super().__init__(f"Transition is UNKNOWN for state={source!r}, action={action!r}.")


class AmbiguousTransitionError(TransitionSystemError):
    def __init__(self, source: str, action: str, outcomes: tuple[TransitionOutcome, ...]) -> None:
        self.source = source
        self.action = action
        self.outcomes = outcomes
        super().__init__(f"Transition is AMBIGUOUS for state={source!r}, action={action!r} ({len(outcomes)} outcomes).")


class TransitionCompileError(TransitionSystemError):
    """Raised when a partial or incompatible Ground cannot be lowered to deterministic IR."""


class TransitionExecutionError(RuntimeError):
    """Raised when a compiled transition program fails to produce its complete trace."""


@dataclass(frozen=True, slots=True)
class TransitionOutcome:
    """Single `(target_state, channel_outputs)` Mealy transition outcome."""

    target: str
    output: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target, str) or not self.target:
            raise TransitionSystemError("TransitionOutcome target must be a non-empty string.")
        out = tuple(self.output)
        for ch in out:
            if isinstance(ch, bool) or not isinstance(ch, int) or not (0 <= ch <= 7):
                raise TransitionSystemError("Each output channel value must be an integer in Z/8Z [0..7].")
        object.__setattr__(self, "output", out)


@dataclass(frozen=True, slots=True)
class TransitionObservation:
    """Observed outcomes for a `(source, action)` cell with support count and evidence tags."""

    source: str
    action: str
    outcomes: tuple[TransitionOutcome, ...]
    support_count: int = 1
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise TransitionSystemError("source must be a non-empty state name.")
        if not isinstance(self.action, str) or not self.action:
            raise TransitionSystemError("action must be a non-empty action name.")
        outs = tuple(self.outcomes)
        if not outs or any(not isinstance(o, TransitionOutcome) for o in outs):
            raise TransitionSystemError("outcomes must contain at least one TransitionOutcome.")
        if len(set(outs)) != len(outs):
            raise TransitionSystemError("Duplicate outcomes in TransitionObservation must be consolidated.")
        if isinstance(self.support_count, bool) or not isinstance(self.support_count, int) or self.support_count < len(outs):
            raise TransitionSystemError("support_count must be an int >= len(outcomes).")
        ev = tuple(self.evidence)
        if any(not isinstance(x, str) or not x for x in ev):
            raise TransitionSystemError("evidence entries must be non-empty strings.")
        object.__setattr__(self, "outcomes", outs)
        object.__setattr__(self, "evidence", ev)

    @property
    def status(self) -> str:
        return "known" if len(self.outcomes) == 1 else "ambiguous"


@dataclass(frozen=True, slots=True)
class TransitionStep:
    index: int
    source: str
    action: str
    target: str
    output: tuple[int, ...]
    support_count: int
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TransitionRun:
    initial_state: str
    final_state: str
    steps: tuple[TransitionStep, ...]

    @property
    def state_trace(self) -> tuple[str, ...]:
        return tuple(s.target for s in self.steps)

    @property
    def outputs(self) -> tuple[tuple[int, ...], ...]:
        return tuple(s.output for s in self.steps)


@dataclass(frozen=True, slots=True)
class GroundTransitionSystem:
    """Evidence-backed partial or complete predictive Mealy transition Ground."""

    states: tuple[str, ...]
    actions: tuple[str, ...]
    initial_state: str
    channels: tuple[str, ...]
    observations: tuple[TransitionObservation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _lookup: Mapping[tuple[str, str], TransitionObservation] = field(init=False, repr=False, compare=False)

    FORMAT = "vld.transition-system.v1"

    def __post_init__(self) -> None:
        states = tuple(self.states)
        actions = tuple(self.actions)
        channels = tuple(self.channels)
        observations = tuple(self.observations)
        if not states or any(not isinstance(s, str) or not s for s in states) or len(set(states)) != len(states):
            raise TransitionSystemError("states must be unique non-empty strings.")
        if not actions or any(not isinstance(a, str) or not a for a in actions) or len(set(actions)) != len(actions):
            raise TransitionSystemError("actions must be unique non-empty strings.")
        if any(not isinstance(c, str) or not c for c in channels) or len(set(channels)) != len(channels):
            raise TransitionSystemError("channels must be unique non-empty strings.")
        if self.initial_state not in states:
            raise TransitionSystemError("initial_state must belong to states.")
        lookup: dict[tuple[str, str], TransitionObservation] = {}
        for obs in observations:
            if not isinstance(obs, TransitionObservation):
                raise TransitionSystemError("observations must contain TransitionObservation instances.")
            if obs.source not in states or obs.action not in actions:
                raise TransitionSystemError(f"Observation {(obs.source, obs.action)!r} references undeclared state/action.")
            for out in obs.outcomes:
                if out.target not in states:
                    raise TransitionSystemError(f"Outcome target {out.target!r} is undeclared.")
                if len(out.output) != len(channels):
                    raise TransitionSystemError("Outcome output width disagrees with channels.")
            key = (obs.source, obs.action)
            if key in lookup:
                raise TransitionSystemError(f"Duplicate observation record for {key!r}.")
            lookup[key] = obs
        norm_meta = parse_strict_json(canonical_json(dict(self.metadata)))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "metadata", MappingProxyType(norm_meta))
        object.__setattr__(self, "_lookup", MappingProxyType(lookup))

    @property
    def complete(self) -> bool:
        return len(self._lookup) == len(self.states) * len(self.actions) and all(
            obs.status == "known" for obs in self.observations
        )

    @property
    def unknown_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((s, a) for s in self.states for a in self.actions if (s, a) not in self._lookup)

    @property
    def ambiguous_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple((obs.source, obs.action) for obs in self.observations if obs.status == "ambiguous")

    def cell_status(self, source: str, action: str) -> str:
        if source not in self.states or action not in self.actions:
            raise TransitionSystemError("Undeclared state or action.")
        obs = self._lookup.get((source, action))
        return "unknown" if obs is None else obs.status

    def get_observation(self, source: str, action: str) -> TransitionObservation | None:
        if source not in self.states or action not in self.actions:
            raise TransitionSystemError("Undeclared state or action.")
        return self._lookup.get((source, action))

    def with_resolved_cell(
        self,
        source: str,
        action: str,
        outcome: TransitionOutcome,
        *,
        evidence_tag: str = "probe_observation",
    ) -> GroundTransitionSystem:
        """Return a new versioned GroundTransitionSystem with `(source, action)` resolved to `outcome`."""
        if source not in self.states or action not in self.actions:
            raise TransitionSystemError("Undeclared state or action.")
        if outcome.target not in self.states or len(outcome.output) != len(self.channels):
            raise TransitionSystemError("Invalid outcome for GroundTransitionSystem.")
        new_obs: list[TransitionObservation] = []
        replaced = False
        for item in self.observations:
            if (item.source, item.action) == (source, action):
                new_obs.append(
                    TransitionObservation(
                        source=source,
                        action=action,
                        outcomes=(outcome,),
                        support_count=max(1, item.support_count),
                        evidence=tuple(dict.fromkeys((*item.evidence, evidence_tag))),
                    )
                )
                replaced = True
            else:
                new_obs.append(item)
        if not replaced:
            new_obs.append(
                TransitionObservation(
                    source=source,
                    action=action,
                    outcomes=(outcome,),
                    support_count=1,
                    evidence=(evidence_tag,),
                )
            )
        return GroundTransitionSystem(
            states=self.states,
            actions=self.actions,
            initial_state=self.initial_state,
            channels=self.channels,
            observations=tuple(new_obs),
            metadata=dict(self.metadata),
        )

    def step(self, source: str, action: str, *, index: int = 0) -> TransitionStep:
        if source not in self.states:
            raise TransitionSystemError(f"Unknown state {source!r}.")
        if action not in self.actions:
            raise TransitionSystemError(f"Unknown action {action!r}.")
        obs = self._lookup.get((source, action))
        if obs is None:
            raise UnknownTransitionError(source, action)
        if obs.status == "ambiguous":
            raise AmbiguousTransitionError(source, action, obs.outcomes)
        out = obs.outcomes[0]
        return TransitionStep(index, source, action, out.target, out.output, obs.support_count, obs.evidence)

    def run(self, actions: Sequence[str]) -> TransitionRun:
        if isinstance(actions, (str, bytes)):
            raise TransitionSystemError("actions must be a sequence of action names, not a string.")
        curr = self.initial_state
        steps: list[TransitionStep] = []
        for idx, act in enumerate(actions):
            st = self.step(curr, act, index=idx)
            steps.append(st)
            curr = st.target
        return TransitionRun(self.initial_state, curr, tuple(steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": list(self.actions),
            "channels": list(self.channels),
            "format": self.FORMAT,
            "initial_state": self.initial_state,
            "metadata": dict(self.metadata),
            "observations": [
                {
                    "action": obs.action,
                    "evidence": list(obs.evidence),
                    "outcomes": [{"output": list(o.output), "target": o.target} for o in obs.outcomes],
                    "source": obs.source,
                    "support_count": obs.support_count,
                }
                for obs in self.observations
            ],
            "states": list(self.states),
        }

    @property
    def fingerprint(self) -> str:
        return domain_digest(self.FORMAT, self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GroundTransitionSystem:
        if not isinstance(data, Mapping) or data.get("format") not in (cls.FORMAT, "VGI-TransitionSystem-1"):
            raise TransitionSystemError("Unsupported transition-system format.")
        obs_list = [
            TransitionObservation(
                source=str(row["source"]),
                action=str(row["action"]),
                outcomes=tuple(TransitionOutcome(str(o["target"]), tuple(o["output"])) for o in row["outcomes"]),
                support_count=int(row.get("support_count", 1)),
                evidence=tuple(row.get("evidence", ())),
            )
            for row in data.get("observations", ())
        ]
        return cls(
            states=tuple(data["states"]),
            actions=tuple(data["actions"]),
            initial_state=str(data["initial_state"]),
            channels=tuple(data.get("channels", ())),
            observations=tuple(obs_list),
            metadata=dict(data.get("metadata", {})),
        )


def _octal_width(cardinality: int) -> int:
    """Minimum positive base-8 digit count without floating-point logarithms."""
    return max(1, ((cardinality - 1).bit_length() + 2) // 3)


def _cell_name(base: str, digit: int, width: int) -> str:
    return base if width == 1 else f"{base}.digit.{digit}"


@dataclass(frozen=True, slots=True)
class CompiledTransitionRun:
    state_trace: tuple[str, ...]
    outputs: tuple[tuple[int, ...], ...]
    final_state: str
    system_fingerprint: str
    program_fingerprint: str
    ground_fingerprint: str


@dataclass(frozen=True, slots=True)
class CompiledTransitionProgram:
    """Compiled multi-digit base-8 Signal8 realization of a complete `GroundTransitionSystem`."""

    system_fingerprint: str
    program: Program
    initial_memory: Mapping[str, SignalPhasorValue]
    states: tuple[str, ...]
    actions: tuple[str, ...]
    channels: tuple[str, ...]
    action_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_memory", MappingProxyType(dict(self.initial_memory)))

    def execute(
        self,
        actions: Sequence[str],
        *,
        interpreter: VirtualGroundInterpreter[SignalPhasorValue] | None = None,
        max_steps: int = 100_000,
    ) -> CompiledTransitionRun:
        word = tuple(actions)
        if len(word) != self.action_count:
            raise TransitionExecutionError(f"Expected {self.action_count} action(s), got {len(word)}.")
        act_idx = {name: idx for idx, name in enumerate(self.actions)}
        mem = dict(self.initial_memory)
        runtime = interpreter or VirtualGroundInterpreter()
        backend = runtime.backend
        if not isinstance(backend, Signal8Backend):
            raise TransitionExecutionError("CompiledTransitionProgram requires Signal8Backend.")

        s_width = _octal_width(len(self.states))
        a_width = _octal_width(len(self.actions))
        for step_i, act in enumerate(word):
            if act not in act_idx:
                raise TransitionExecutionError(f"Action {act!r} is outside compiled Ground.")
            code = act_idx[act]
            for d in range(a_width):
                mem[_cell_name(f"input.action.{step_i}", d, a_width)] = backend.encode_symbol(
                    (code >> (3 * d)) & 7, observer=1
                )

        res = runtime.execute(self.program, mem, max_steps=max_steps)
        observed = res.observed_symbols(backend)
        required = {
            *(_cell_name(f"state_after_{i}", d, s_width) for i in range(self.action_count) for d in range(s_width)),
            *(f"output_{i}_{ch}" for i in range(self.action_count) for ch in range(len(self.channels))),
            *(_cell_name("final_state", d, s_width) for d in range(s_width)),
        }
        missing = required - set(observed.keys())
        if missing:
            raise TransitionExecutionError(f"Compiled dispatch halted early; missing emissions: {sorted(missing)}")

        def decode_state(prefix: str) -> str:
            idx = sum(int(observed[_cell_name(prefix, d, s_width)]) << (3 * d) for d in range(s_width))
            if not (0 <= idx < len(self.states)):
                raise TransitionExecutionError(f"Decoded state index {idx} outside Ground.")
            return self.states[idx]

        state_trace = tuple(decode_state(f"state_after_{i}") for i in range(self.action_count))
        out_trace = tuple(
            tuple(int(observed[f"output_{i}_{ch}"]) for ch in range(len(self.channels)))
            for i in range(self.action_count)
        )
        final_st = decode_state("final_state")
        return CompiledTransitionRun(
            state_trace=state_trace,
            outputs=out_trace,
            final_state=final_st,
            system_fingerprint=self.system_fingerprint,
            program_fingerprint=res.program_fingerprint,
            ground_fingerprint=res.ground_fingerprint,
        )


class TransitionCompiler:
    """Compile complete deterministic `GroundTransitionSystem` models to Signal8 IR."""

    def compile(
        self,
        system: GroundTransitionSystem,
        action_count: int,
        *,
        backend: Signal8Backend | None = None,
    ) -> CompiledTransitionProgram:
        if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 0:
            raise TransitionCompileError("action_count must be a non-negative integer.")
        if not system.complete:
            raise TransitionCompileError(
                f"Cannot compile incomplete/ambiguous Ground: unknown={system.unknown_pairs}, ambiguous={system.ambiguous_pairs}"
            )
        sig_backend = backend or Signal8Backend()
        s_code = {s: idx for idx, s in enumerate(system.states)}
        a_code = {a: idx for idx, a in enumerate(system.actions)}
        s_width = _octal_width(len(system.states))
        a_width = _octal_width(len(system.actions))
        mem: dict[str, SignalPhasorValue] = {}

        def put_octal(prefix: str, code: int, width: int) -> None:
            for d in range(width):
                mem[_cell_name(prefix, d, width)] = sig_backend.encode_symbol((code >> (3 * d)) & 7, observer=1)

        put_octal("state", s_code[system.initial_state], s_width)
        for sym in range(8):
            mem[f"const.symbol.{sym}"] = sig_backend.encode_symbol(sym, observer=1)
        for s_idx in s_code.values():
            put_octal(f"const.state.{s_idx}", s_idx, s_width)
        for a_idx in a_code.values():
            put_octal(f"const.action.{a_idx}", a_idx, a_width)
        for step_i in range(action_count):
            put_octal(f"input.action.{step_i}", 0, a_width)

        insts: list[Instruction] = []

        def emit(op: str, **kwargs: Any) -> int:
            insts.append(Instruction(op, kwargs))
            return len(insts) - 1

        def patch(idx: int, target: int) -> None:
            old = insts[idx]
            patched = dict(old.args)
            patched["target"] = target
            insts[idx] = Instruction(old.opcode, patched)

        edges = {(obs.source, obs.action): obs.outcomes[0] for obs in system.observations}
        for step_i in range(action_count):
            for d in range(s_width):
                emit("LOAD", dst=_cell_name("dispatch_state", d, s_width), address=_cell_name("state", d, s_width))
            for d in range(a_width):
                emit(
                    "LOAD",
                    dst=_cell_name("dispatch_action", d, a_width),
                    address=_cell_name(f"input.action.{step_i}", d, a_width),
                )
            done_jumps: list[int] = []
            for src in system.states:
                for act in system.actions:
                    outcome = edges[(src, act)]
                    miss_jumps: list[int] = []
                    for kind, code, width in (
                        ("state", s_code[src], s_width),
                        ("action", a_code[act], a_width),
                    ):
                        for d in range(width):
                            emit("LOAD", dst=f"exp_{kind}", address=_cell_name(f"const.{kind}.{code}", d, width))
                            br = emit(
                                "BRANCH_EQ",
                                left=_cell_name(f"dispatch_{kind}", d, width),
                                right=f"exp_{kind}",
                                target=-1,
                                tolerance_rad=1e-8,
                            )
                            miss_jumps.append(emit("JUMP", target=-1))
                            patch(br, len(insts))

                    target_code = s_code[outcome.target]
                    for d in range(s_width):
                        emit("LOAD", dst="next_state", address=_cell_name(f"const.state.{target_code}", d, s_width))
                        emit("STORE", src="next_state", address=_cell_name("state", d, s_width))
                        emit("EMIT", src="next_state", label=_cell_name(f"state_after_{step_i}", d, s_width))
                    for ch_idx, ch_val in enumerate(outcome.output):
                        emit("LOAD", dst="ch_out", address=f"const.symbol.{ch_val}")
                        emit("EMIT", src="ch_out", label=f"output_{step_i}_{ch_idx}")
                    done_jumps.append(emit("JUMP", target=-1))
                    next_candidate_pc = len(insts)
                    for mj in miss_jumps:
                        patch(mj, next_candidate_pc)

            emit("HALT")
            after_step_pc = len(insts)
            for dj in done_jumps:
                patch(dj, after_step_pc)

        for d in range(s_width):
            emit("LOAD", dst="final_state_reg", address=_cell_name("state", d, s_width))
            emit("EMIT", src="final_state_reg", label=_cell_name("final_state", d, s_width))
        emit("HALT")

        prog = Program(
            name=f"ground-transition-{system.fingerprint[:12]}",
            instructions=tuple(insts),
            metadata={
                "action_count": action_count,
                "action_width": a_width,
                "source": "vld.transition-compiler.v2",
                "state_width": s_width,
                "system_fingerprint": system.fingerprint,
            },
        )
        return CompiledTransitionProgram(
            system_fingerprint=system.fingerprint,
            program=prog,
            initial_memory=mem,
            states=system.states,
            actions=system.actions,
            channels=system.channels,
            action_count=action_count,
        )


def differential_verify_transitions(
    system: GroundTransitionSystem,
    action_sequences: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Differentially execute `action_sequences` against direct Mealy semantics and compiled Signal8 IR."""
    if not system.complete:
        raise TransitionCompileError("Differential verification requires a complete GroundTransitionSystem.")
    compiler = TransitionCompiler()
    cache: dict[int, CompiledTransitionProgram] = {}
    errors: list[dict[str, Any]] = []
    for raw_seq in action_sequences:
        seq = tuple(raw_seq)
        if len(seq) not in cache:
            cache[len(seq)] = compiler.compile(system, len(seq))
        direct = system.run(seq)
        compiled = cache[len(seq)].execute(seq)
        if direct.state_trace != compiled.state_trace or direct.outputs != compiled.outputs or direct.final_state != compiled.final_state:
            errors.append(
                {
                    "actions": list(seq),
                    "compiled_outputs": [list(x) for x in compiled.outputs],
                    "compiled_states": list(compiled.state_trace),
                    "direct_outputs": [list(x) for x in direct.outputs],
                    "direct_states": list(direct.state_trace),
                }
            )
    return {
        "cases_checked": len(action_sequences),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "system_fingerprint": system.fingerprint,
    }
