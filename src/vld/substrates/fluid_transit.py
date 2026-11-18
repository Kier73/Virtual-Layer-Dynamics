r"""Phase 9 — Fluid In-Transit Computation Substrate: Topological Lifting of CPU, GPU & Virtual Memory Algorithms (`vld.substrates.fluid_transit`).

Mathematical foundations:
Instead of mimicking hardware registers, warps, or physical DRAM `memcpy` buses inside isolated machines,
computation is lifted into **in-transit algebraic transformations** across chains of stream interpreters
$v_0 \xrightarrow{e_1} v_1 \xrightarrow{e_2} \dots \xrightarrow{e_k} v_k$:

1. **CPU Loop & Branch Lifting (Semiring & Finite-Transducer Stream Monoids):**
   - Sequential dynamic programming / shortest-path loops $d^{(t+1)}_j = \min_i (d^{(t)}_i + W^{(t)}_{i,j})$
     lift to associative $(\mathbb{R} \cup \{+\infty\}, \min, +)$ `TropicalMinPlusMatrix` products in transit:
     $$d^{(T)} = d^{(0)} \odot \left(\bigodot_{t=0}^{T-1} W^{(t)}\right)$$
   - Branching finite-state loops $s_{t+1} = \delta(s_t, u_t)$ over $[0, S)$ lift to functional composition in
     the transformation monoid $\operatorname{End}([0, S))$, collapsing arbitrary-length sub-streams into $S$-entry
     endomorphism tables in transit.

2. **GPU SIMT Pipeline Lifting (Associative Online Flash-Stream Monoid):**
   - Distributed multi-shard softmax attention over KV partitions $\{(K^{(h)}, V^{(h)})\}_{h=1}^H$ lifts to the
     exact associative monoid on $\mathbb{R} \times \mathbb{R}_{\ge 0} \times \mathbb{R}^{d_v}$:
     $$(m_A, \ell_A, \mathbf{o}_A) \otimes (m_B, \ell_B, \mathbf{o}_B)
       = \left(m_{AB},\; e^{m_A - m_{AB}}\ell_A + e^{m_B - m_{AB}}\ell_B,\;
         e^{m_A - m_{AB}}\mathbf{o}_A + e^{m_B - m_{AB}}\mathbf{o}_B\right)$$
     with $m_{AB} = \max(m_A, m_B)$ and identity $(-\infty, 0, \mathbf{0})$, normalizing once at the terminal
     receiver ($\mathbf{o}/\ell$) with zero $[T_q, T_k]$ intermediate HBM materialization.

3. **Virtual Memory Movement Lifting ($O(1)$ View Descriptors & Pullback Commutation):**
   - Address-space operations (`slice`, `reverse`, `rotate`, `transpose_2d`, `cow_patch`) compose on compact
     index descriptors rather than copying payload bytes, and permutation-invariant reductions (`sum`, `min`, `max`, `xor`)
     pull back across permutations ($R(\pi^* B) = R(B)$) with zero scatter/gather movement.

4. **Path-Encoded Emulated Machines & Schedule Rewriting (`InTransitPathMachine`):**
   - Information packets carry their current representation, ordered transformation schedule, and terminal
     interpretable contract through stream hops (including `HeadlessOS` POSIX pipes), fusing/commuting operations
     in transit while recording hash-chained `TransitHopReceipt`s.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Any, Callable, Mapping, Sequence

from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.journal import GENESIS_PREV_DIGEST
from vld.substrates.headless_os import HeadlessOS
from vld.substrates.vm import VMInstruction


class FluidTransitError(RuntimeError):
    """Raised when an in-transit transformation, monoid dimension, view bound, or hop receipt fails."""


# ============================================================================
# 1. CPU Algorithm Lifting: Tropical (min, +) & Finite-Transducer Monoids
# ============================================================================


@dataclass(frozen=True, slots=True)
class TropicalMinPlusMatrix:
    r"""Associative $(\mathbb{R} \cup \{+\infty\}, \min, +)$ semiring element lifting sequential DP/Bellman-Ford loops."""

    dim: int
    entries: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if isinstance(self.dim, bool) or not isinstance(self.dim, int) or self.dim <= 0:
            raise FluidTransitError("TropicalMinPlusMatrix dim must be a positive integer.")
        if len(self.entries) != self.dim or any(len(r) != self.dim for r in self.entries):
            raise FluidTransitError("TropicalMinPlusMatrix entries must have shape (dim, dim).")
        norm_rows: list[tuple[float, ...]] = []
        for row in self.entries:
            norm_r: list[float] = []
            for v in row:
                fv = float(v)
                if math.isnan(fv) or fv == -math.inf:
                    raise FluidTransitError("TropicalMinPlusMatrix forbids NaN and -inf entries.")
                norm_r.append(fv)
            norm_rows.append(tuple(norm_r))
        object.__setattr__(self, "entries", tuple(norm_rows))

    @classmethod
    def identity(cls, dim: int) -> TropicalMinPlusMatrix:
        return cls(
            dim=dim,
            entries=tuple(
                tuple(0.0 if i == j else math.inf for j in range(dim))
                for i in range(dim)
            ),
        )

    def compose(self, right: TropicalMinPlusMatrix) -> TropicalMinPlusMatrix:
        r"""Compute associative tropical product $(A \odot B)_{i,j} = \min_k (A_{i,k} + B_{k,j})$."""
        if self.dim != right.dim:
            raise FluidTransitError(f"Tropical dimension mismatch: {self.dim} vs {right.dim}.")
        d = self.dim
        a = self.entries
        b = right.entries
        out = tuple(
            tuple(min(a[i][k] + b[k][j] for k in range(d)) for j in range(d))
            for i in range(d)
        )
        return TropicalMinPlusMatrix(dim=d, entries=out)

    def apply_left_state(self, state_vec: Sequence[float]) -> tuple[float, ...]:
        r"""Apply $s' = s \odot M$: $s'_j = \min_i (s_i + M_{i,j})$."""
        if len(state_vec) != self.dim:
            raise FluidTransitError("State vector length disagrees with TropicalMinPlusMatrix dim.")
        s = [float(x) for x in state_vec]
        if any(math.isnan(x) or x == -math.inf for x in s):
            raise FluidTransitError("Invalid NaN/-inf in tropical state vector.")
        return tuple(min(s[i] + self.entries[i][j] for i in range(self.dim)) for j in range(self.dim))

    def power(self, steps: int) -> TropicalMinPlusMatrix:
        r"""Fast $O(d^3 \log T)$ tropical binary exponentiation for stationary DP horizons."""
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise FluidTransitError("steps must be a non-negative integer.")
        acc = TropicalMinPlusMatrix.identity(self.dim)
        base = self
        n = steps
        while n > 0:
            if n & 1:
                acc = acc.compose(base)
            n >>= 1
            if n > 0:
                base = base.compose(base)
        return acc


@dataclass(frozen=True, slots=True)
class FiniteTransducerSummary:
    r"""Endomorphism $\tau \in \operatorname{End}([0, S))$ lifting branch-heavy sequential state loops into transit."""

    num_states: int
    mapping: tuple[int, ...]
    steps_summarized: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.num_states, bool) or not isinstance(self.num_states, int) or self.num_states <= 0:
            raise FluidTransitError("num_states must be a positive integer.")
        if len(self.mapping) != self.num_states:
            raise FluidTransitError("mapping length must equal num_states.")
        clean = tuple(int(v) for v in self.mapping)
        if any(v < 0 or v >= self.num_states for v in clean):
            raise FluidTransitError("Transducer state image out of bounds [0, num_states).")
        if self.steps_summarized < 0:
            raise FluidTransitError("steps_summarized must be non-negative.")
        object.__setattr__(self, "mapping", clean)

    @classmethod
    def identity(cls, num_states: int) -> FiniteTransducerSummary:
        return cls(num_states=num_states, mapping=tuple(range(num_states)), steps_summarized=0)

    @classmethod
    def from_input_chunk(
        cls,
        num_states: int,
        step_fn: Callable[[int, int], int],
        inputs: Sequence[int],
    ) -> FiniteTransducerSummary:
        cur = list(range(num_states))
        for sym in inputs:
            isym = int(sym)
            cur = [int(step_fn(s, isym)) for s in cur]
        return cls(num_states=num_states, mapping=tuple(cur), steps_summarized=len(inputs))

    def then(self, later: FiniteTransducerSummary) -> FiniteTransducerSummary:
        """Compose `self` followed by `later`: `(later o self)(s) = later.mapping[self.mapping[s]]`."""
        if self.num_states != later.num_states:
            raise FluidTransitError("FiniteTransducerSummary state-space mismatch.")
        return FiniteTransducerSummary(
            num_states=self.num_states,
            mapping=tuple(later.mapping[self.mapping[s]] for s in range(self.num_states)),
            steps_summarized=self.steps_summarized + later.steps_summarized,
        )

    def evaluate(self, initial_state: int) -> int:
        if not (0 <= initial_state < self.num_states):
            raise FluidTransitError("initial_state out of range.")
        return self.mapping[initial_state]


def fold_cpu_stream_in_transit(
    chunks: Sequence[Sequence[int]],
    *,
    num_states: int,
    step_fn: Callable[[int, int], int],
    initial_state: int,
) -> tuple[int, FiniteTransducerSummary]:
    """Lift a sequential branching CPU state loop across stream chunks into associative monoid summaries."""
    summary = FiniteTransducerSummary.identity(num_states)
    for chunk in chunks:
        local = FiniteTransducerSummary.from_input_chunk(num_states, step_fn, chunk)
        summary = summary.then(local)
    return summary.evaluate(initial_state), summary


# ============================================================================
# 2. GPU SIMT Lifting: Associative Online Flash-Stream Attention Monoid
# ============================================================================


@dataclass(frozen=True, slots=True)
class OnlineFlashStreamState:
    r"""Associative per-query `(max_score, sum_exp, unnormalized_out)` transit packet for multi-hop attention."""

    num_queries: int
    value_dim: int
    max_scores: tuple[float, ...]
    sum_exps: tuple[float, ...]
    weighted_values: tuple[tuple[float, ...], ...]
    keys_accumulated: int = 0

    def __post_init__(self) -> None:
        if self.num_queries <= 0 or self.value_dim <= 0:
            raise FluidTransitError("num_queries and value_dim must be positive.")
        if (
            len(self.max_scores) != self.num_queries
            or len(self.sum_exps) != self.num_queries
            or len(self.weighted_values) != self.num_queries
            or any(len(r) != self.value_dim for r in self.weighted_values)
        ):
            raise FluidTransitError("OnlineFlashStreamState shape mismatch.")

    @classmethod
    def identity(cls, num_queries: int, value_dim: int) -> OnlineFlashStreamState:
        return cls(
            num_queries=num_queries,
            value_dim=value_dim,
            max_scores=tuple(-math.inf for _ in range(num_queries)),
            sum_exps=tuple(0.0 for _ in range(num_queries)),
            weighted_values=tuple(tuple(0.0 for _ in range(value_dim)) for _ in range(num_queries)),
            keys_accumulated=0,
        )

    @classmethod
    def from_kv_shard(
        cls,
        queries: Sequence[Sequence[float]],
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
        *,
        query_start_pos: int = 0,
        key_start_pos: int = 0,
        causal: bool = True,
        scale: float | None = None,
    ) -> OnlineFlashStreamState:
        if not queries:
            raise FluidTransitError("queries cannot be empty.")
        nq = len(queries)
        dk = len(queries[0])
        if dk == 0 or any(len(q) != dk for q in queries):
            raise FluidTransitError("Malformed query vectors.")
        if len(keys) != len(values):
            raise FluidTransitError("keys and values shard lengths must match.")
        if not keys:
            raise FluidTransitError("Empty KV shard requires explicit value_dim via OnlineFlashStreamState.identity.")
        dv = len(values[0])
        if dv == 0 or any(len(k) != dk for k in keys) or any(len(v) != dv for v in values):
            raise FluidTransitError("Malformed key or value dimensions in shard.")
        sc = float(scale) if scale is not None else (1.0 / math.sqrt(dk))

        m_list: list[float] = []
        l_list: list[float] = []
        o_list: list[tuple[float, ...]] = []
        for qi, q_vec in enumerate(queries):
            q_abs = query_start_pos + qi
            valid_scores: list[tuple[float, Sequence[float]]] = []
            for ki, (k_vec, v_vec) in enumerate(zip(keys, values)):
                k_abs = key_start_pos + ki
                if causal and k_abs > q_abs:
                    continue
                dot = math.fsum(float(q_vec[d]) * float(k_vec[d]) for d in range(dk)) * sc
                if not math.isfinite(dot):
                    raise FluidTransitError("Non-finite attention score in OnlineFlashStreamState.")
                valid_scores.append((dot, v_vec))
            if not valid_scores:
                m_list.append(-math.inf)
                l_list.append(0.0)
                o_list.append(tuple(0.0 for _ in range(dv)))
            else:
                m_q = max(s for s, _ in valid_scores)
                exps = [math.exp(s - m_q) for s, _ in valid_scores]
                l_q = math.fsum(exps)
                o_q = tuple(
                    math.fsum(w * float(vv[d]) for w, (_, vv) in zip(exps, valid_scores))
                    for d in range(dv)
                )
                m_list.append(m_q)
                l_list.append(l_q)
                o_list.append(o_q)

        return cls(
            num_queries=nq,
            value_dim=dv,
            max_scores=tuple(m_list),
            sum_exps=tuple(l_list),
            weighted_values=tuple(o_list),
            keys_accumulated=len(keys),
        )

    def combine(self, other: OnlineFlashStreamState) -> OnlineFlashStreamState:
        r"""Exact associative monoid product $(m_A, \ell_A, O_A) \otimes (m_B, \ell_B, O_B)$."""
        if self.num_queries != other.num_queries or self.value_dim != other.value_dim:
            raise FluidTransitError("OnlineFlashStreamState shape mismatch in combine.")
        m_out: list[float] = []
        l_out: list[float] = []
        o_out: list[tuple[float, ...]] = []
        for i in range(self.num_queries):
            ma, la, oa = self.max_scores[i], self.sum_exps[i], self.weighted_values[i]
            mb, lb, ob = other.max_scores[i], other.sum_exps[i], other.weighted_values[i]
            if la == 0.0 and lb == 0.0:
                m_out.append(-math.inf)
                l_out.append(0.0)
                o_out.append(tuple(0.0 for _ in range(self.value_dim)))
            elif la == 0.0:
                m_out.append(mb)
                l_out.append(lb)
                o_out.append(ob)
            elif lb == 0.0:
                m_out.append(ma)
                l_out.append(la)
                o_out.append(oa)
            else:
                m_new = max(ma, mb)
                sa = math.exp(ma - m_new)
                sb = math.exp(mb - m_new)
                l_out.append(sa * la + sb * lb)
                o_out.append(tuple(sa * oa[d] + sb * ob[d] for d in range(self.value_dim)))
                m_out.append(m_new)
        return OnlineFlashStreamState(
            num_queries=self.num_queries,
            value_dim=self.value_dim,
            max_scores=tuple(m_out),
            sum_exps=tuple(l_out),
            weighted_values=tuple(o_out),
            keys_accumulated=self.keys_accumulated + other.keys_accumulated,
        )

    def finalize(self) -> list[list[float]]:
        """Normalize at the terminal receiver interpreter (`O / l`), refusing empty-support queries."""
        out: list[list[float]] = []
        for i in range(self.num_queries):
            denom = self.sum_exps[i]
            if denom <= 0.0 or not math.isfinite(denom):
                raise FluidTransitError(f"Query {i} attended to zero valid keys; cannot normalize.")
            out.append([v / denom for v in self.weighted_values[i]])
        return out

    @property
    def transit_payload_bytes(self) -> int:
        """Exact wire byte count of `(m, l, O)` in binary32 (`4 * T_q * (2 + d_v)`)."""
        return 4 * self.num_queries * (2 + self.value_dim)


def stream_flash_attention_in_transit(
    queries: Sequence[Sequence[float]],
    kv_shards: Sequence[tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]],
    *,
    query_start_pos: int = 0,
    causal: bool = True,
) -> tuple[list[list[float]], dict[str, int]]:
    """Execute multi-hop streaming attention across KV shards with zero full-matrix materialization."""
    if not queries or not kv_shards:
        raise FluidTransitError("queries and kv_shards must be non-empty.")
    dv = len(kv_shards[0][1][0])
    state = OnlineFlashStreamState.identity(len(queries), dv)
    k_offset = 0
    hop_bytes = 0
    total_keys = 0
    for k_shard, v_shard in kv_shards:
        local = OnlineFlashStreamState.from_kv_shard(
            queries,
            k_shard,
            v_shard,
            query_start_pos=query_start_pos,
            key_start_pos=k_offset,
            causal=causal,
        )
        state = state.combine(local)
        k_offset += len(k_shard)
        total_keys += len(k_shard)
        hop_bytes += state.transit_payload_bytes
    naive_hbm_bytes = 4 * (len(queries) * total_keys + total_keys * (len(queries[0]) + dv))
    return state.finalize(), {
        "hops": len(kv_shards),
        "keys_processed": total_keys,
        "naive_materialized_bytes": naive_hbm_bytes,
        "transit_stream_bytes": hop_bytes,
    }


def verify_premature_normalization_counterexample() -> dict[str, Any]:
    """Constructive negative control proving naive averaging of locally normalized softmax shards is unsound."""
    q = [[2.0]]
    s1 = ([[0.0]], [[1.0]])
    s2 = ([[4.0]], [[10.0]])
    exact_out, _ = stream_flash_attention_in_transit(q, (s1, s2), causal=False)
    local1 = OnlineFlashStreamState.from_kv_shard(q, s1[0], s1[1], causal=False).finalize()[0][0]
    local2 = OnlineFlashStreamState.from_kv_shard(q, s2[0], s2[1], causal=False).finalize()[0][0]
    naive_avg = 0.5 * (local1 + local2)
    discrepancy = abs(exact_out[0][0] - naive_avg)
    if discrepancy < 1.0:
        raise FluidTransitError("Counterexample failed to separate sufficient-state monoid from premature normalization.")
    return {
        "discrepancy": discrepancy,
        "exact_monoid_output": exact_out[0][0],
        "naive_premature_average": naive_avg,
        "sound_collapse_requires_max_and_sum_exp": True,
    }


# ============================================================================
# 3. Virtual Memory Movement Lifting: O(1) View Morphisms & Pullback Commutation
# ============================================================================


@dataclass(frozen=True, slots=True)
class VirtualMemoryMovementView:
    """Zero-copy virtual buffer view tracking index morphisms + sparse CoW patches without moving backing bytes."""

    base_id: str
    base_length: int
    logical_length: int
    op_chain: tuple[tuple[str, tuple[int, ...]], ...] = ()
    cow_patches: tuple[tuple[int, int], ...] = ()  # sorted (logical_index, byte_val)

    def __post_init__(self) -> None:
        if self.base_length <= 0 or self.logical_length < 0:
            raise FluidTransitError("Invalid base_length or logical_length in VirtualMemoryMovementView.")
        for idx, val in self.cow_patches:
            if not (0 <= idx < self.logical_length) or not (0 <= val <= 255):
                raise FluidTransitError("CoW patch index or byte value out of bounds.")

    @classmethod
    def from_buffer(cls, base_id: str, length: int) -> VirtualMemoryMovementView:
        return cls(base_id=base_id, base_length=length, logical_length=length)

    def _map_to_base_index(self, logical_idx: int) -> int:
        if not (0 <= logical_idx < self.logical_length):
            raise FluidTransitError(f"Logical index {logical_idx} out of bounds [0, {self.logical_length}).")
        idx = logical_idx
        cur_len = self.logical_length
        # Walk inverse index morphisms from outermost view back to base buffer!
        for kind, params in reversed(self.op_chain):
            if kind == "reverse":
                idx = (cur_len - 1) - idx
            elif kind == "rotate":
                (shift,) = params
                # view[i] = prev[(i - shift) % cur_len]
                idx = (idx - shift) % cur_len
            elif kind == "transpose_2d":
                rows, cols = params
                # Output has shape (cols, rows): out[c * rows + r] = prev[r * cols + c]
                c, r = divmod(idx, rows)
                idx = r * cols + c
            elif kind == "slice":
                start, step, prev_len = params
                idx = start + idx * step
                cur_len = prev_len
            else:
                raise FluidTransitError(f"Unknown VM movement op {kind!r}.")
        return idx

    @property
    def is_bijective_over_base(self) -> bool:
        """True iff the composed view is a bijection over `[0, base_length)` (no sub-slicing)."""
        return self.logical_length == self.base_length and all(k != "slice" for k, _ in self.op_chain)

    def reverse_view(self) -> VirtualMemoryMovementView:
        n = self.logical_length
        new_patches = tuple(sorted(((n - 1 - i, v) for i, v in self.cow_patches), key=lambda t: t[0]))
        return VirtualMemoryMovementView(
            base_id=self.base_id,
            base_length=self.base_length,
            logical_length=n,
            op_chain=(*self.op_chain, ("reverse", ())),
            cow_patches=new_patches,
        )

    def rotate_view(self, shift: int) -> VirtualMemoryMovementView:
        n = self.logical_length
        if n == 0:
            return self
        s = int(shift) % n
        if s == 0:
            return self
        new_patches = tuple(sorted((((i + s) % n, v) for i, v in self.cow_patches), key=lambda t: t[0]))
        return VirtualMemoryMovementView(
            base_id=self.base_id,
            base_length=self.base_length,
            logical_length=n,
            op_chain=(*self.op_chain, ("rotate", (s,))),
            cow_patches=new_patches,
        )

    def transpose_2d(self, rows: int, cols: int) -> VirtualMemoryMovementView:
        if rows <= 0 or cols <= 0 or rows * cols != self.logical_length:
            raise FluidTransitError(f"transpose_2d({rows}, {cols}) incompatible with length {self.logical_length}.")
        new_patches = tuple(
            sorted(
                (((i % cols) * rows + (i // cols), v) for i, v in self.cow_patches),
                key=lambda t: t[0],
            )
        )
        return VirtualMemoryMovementView(
            base_id=self.base_id,
            base_length=self.base_length,
            logical_length=self.logical_length,
            op_chain=(*self.op_chain, ("transpose_2d", (int(rows), int(cols)))),
            cow_patches=new_patches,
        )

    def slice_view(self, start: int, stop: int, step: int = 1) -> VirtualMemoryMovementView:
        if step <= 0 or start < 0 or stop < start or stop > self.logical_length:
            raise FluidTransitError("Invalid slice_view bounds.")
        new_len = len(range(start, stop, step))
        patch_map = dict(self.cow_patches)
        new_patches: list[tuple[int, int]] = []
        for out_i, old_i in enumerate(range(start, stop, step)):
            if old_i in patch_map:
                new_patches.append((out_i, patch_map[old_i]))
        return VirtualMemoryMovementView(
            base_id=self.base_id,
            base_length=self.base_length,
            logical_length=new_len,
            op_chain=(*self.op_chain, ("slice", (int(start), int(step), self.logical_length))),
            cow_patches=tuple(new_patches),
        )

    def cow_write(self, logical_index: int, byte_value: int) -> VirtualMemoryMovementView:
        if not (0 <= logical_index < self.logical_length) or not (0 <= byte_value <= 255):
            raise FluidTransitError("cow_write index or byte_value out of bounds.")
        pmap = dict(self.cow_patches)
        pmap[int(logical_index)] = int(byte_value)
        return VirtualMemoryMovementView(
            base_id=self.base_id,
            base_length=self.base_length,
            logical_length=self.logical_length,
            op_chain=self.op_chain,
            cow_patches=tuple(sorted(pmap.items())),
        )

    def to_transit_descriptor(self) -> dict[str, Any]:
        return {
            "base_id": self.base_id,
            "base_length": self.base_length,
            "cow_patches": [[i, v] for i, v in self.cow_patches],
            "logical_length": self.logical_length,
            "op_chain": [[k, list(p)] for k, p in self.op_chain],
            "schema": "vld.vm-movement-descriptor.v1",
        }

    @property
    def descriptor_bytes_size(self) -> int:
        return len(canonical_bytes(self.to_transit_descriptor()))

    def materialize(self, base_bytes: bytes | bytearray | memoryview) -> bytes:
        """Interpret and materialize the terminal byte sequence at the destination receiver."""
        if len(base_bytes) != self.base_length:
            raise FluidTransitError("Backing buffer length disagrees with base_length.")
        pmap = dict(self.cow_patches)
        out = bytearray(self.logical_length)
        for i in range(self.logical_length):
            if i in pmap:
                out[i] = pmap[i]
            else:
                out[i] = base_bytes[self._map_to_base_index(i)]
        return bytes(out)

    def commute_invariant_reduction(
        self,
        base_bytes: bytes | bytearray | memoryview,
        reduction: str = "sum",
    ) -> int:
        """Evaluate `sum` or `xor` by pulling back across bijective permutations without permuting the buffer!"""
        if len(base_bytes) != self.base_length:
            raise FluidTransitError("Backing buffer length disagrees with base_length.")
        if not self.is_bijective_over_base:
            # Fallback to indexed streaming read without allocating full copy
            pmap = dict(self.cow_patches)
            if reduction == "sum":
                return sum(pmap[i] if i in pmap else base_bytes[self._map_to_base_index(i)] for i in range(self.logical_length))
            if reduction == "xor":
                acc = 0
                for i in range(self.logical_length):
                    acc ^= pmap[i] if i in pmap else base_bytes[self._map_to_base_index(i)]
                return acc
            raise FluidTransitError(f"Unsupported reduction {reduction!r}.")

        overwritten_base = {self._map_to_base_index(i): v for i, v in self.cow_patches}
        if reduction == "sum":
            total = sum(base_bytes)
            for b_idx, new_v in overwritten_base.items():
                total += new_v - base_bytes[b_idx]
            return total
        if reduction == "xor":
            acc = 0
            for b in base_bytes:
                acc ^= b
            for b_idx, new_v in overwritten_base.items():
                acc ^= base_bytes[b_idx] ^ new_v
            return acc
        raise FluidTransitError(f"Unsupported reduction {reduction!r}.")


# ============================================================================
# 4. Operation-Order Rewriting & Path-Encoded In-Transit Stream Machines
# ============================================================================


@dataclass(frozen=True, slots=True)
class TransitOperation:
    """Single algebraic transformation scheduled inside an in-transit stream packet."""

    op: str
    params: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TransitOperation:
        return cls(op=str(data["op"]), params=dict(data.get("params", {})))


def optimize_transit_schedule(
    schedule: Sequence[TransitOperation],
    *,
    vector_length: int | None = None,
) -> tuple[TransitOperation, ...]:
    """Algebraically fuse and commute in-transit operations so information transforms with minimal work."""
    ops: list[TransitOperation] = []
    for raw_op in schedule:
        # Drop identity rotations or identity affine maps immediately
        if raw_op.op == "rotate" and int(raw_op.params.get("shift", 0)) == 0:
            continue
        if raw_op.op == "affine_map":
            sc = int(raw_op.params.get("scale", 1))
            bi = int(raw_op.params.get("bias", 0))
            mod = int(raw_op.params["modulus"])
            sc %= mod
            bi %= mod
            if sc == 1 and bi == 0:
                continue
            raw_op = TransitOperation("affine_map", {"bias": bi, "modulus": mod, "scale": sc})

        if ops:
            prev = ops[-1]
            # 1. Involution cancellation: reverse o reverse = id
            if prev.op == "reverse" and raw_op.op == "reverse":
                ops.pop()
                continue
            # 2. Transpose cancellation: transpose_2d(r, c) then transpose_2d(c, r) = id
            if (
                prev.op == "transpose_2d"
                and raw_op.op == "transpose_2d"
                and int(prev.params["rows"]) == int(raw_op.params["cols"])
                and int(prev.params["cols"]) == int(raw_op.params["rows"])
            ):
                ops.pop()
                continue
            # 3. Rotation group fusion: rotate(s1) then rotate(s2) = rotate(s1 + s2)
            if prev.op == "rotate" and raw_op.op == "rotate":
                comb = int(prev.params["shift"]) + int(raw_op.params["shift"])
                if vector_length is not None and vector_length > 0:
                    comb %= vector_length
                ops.pop()
                if comb != 0:
                    ops.append(TransitOperation("rotate", {"shift": comb}))
                continue
            # 4. Modular affine endomorphism fusion: a2*(a1*x + b1) + b2 = (a2*a1)*x + (a2*b1 + b2) mod M
            if (
                prev.op == "affine_map"
                and raw_op.op == "affine_map"
                and int(prev.params["modulus"]) == int(raw_op.params["modulus"])
            ):
                mod = int(prev.params["modulus"])
                a1, b1 = int(prev.params["scale"]), int(prev.params["bias"])
                a2, b2 = int(raw_op.params["scale"]), int(raw_op.params["bias"])
                a_f = (a2 * a1) % mod
                b_f = (a2 * b1 + b2) % mod
                ops.pop()
                if not (a_f == 1 and b_f == 0):
                    ops.append(TransitOperation("affine_map", {"bias": b_f, "modulus": mod, "scale": a_f}))
                continue
            # 5. Pullback commutation: reduce_sum / reduce_xor absorbs preceding pure permutations!
            if raw_op.op in ("reduce_sum", "reduce_xor"):
                while ops and ops[-1].op in ("reverse", "rotate", "transpose_2d"):
                    ops.pop()

        ops.append(raw_op)
    return tuple(ops)


def _apply_single_transit_op(payload: Any, op: TransitOperation) -> tuple[Any, str]:
    if op.op == "reverse":
        if not isinstance(payload, list):
            raise FluidTransitError("reverse requires a list vector payload.")
        return list(reversed(payload)), "int_vector"
    if op.op == "rotate":
        if not isinstance(payload, list):
            raise FluidTransitError("rotate requires a list vector payload.")
        n = len(payload)
        if n == 0:
            return [], "int_vector"
        s = int(op.params["shift"]) % n
        return (payload[-s:] + payload[:-s]) if s else list(payload), "int_vector"
    if op.op == "transpose_2d":
        if not isinstance(payload, list):
            raise FluidTransitError("transpose_2d requires a flat list payload.")
        rows = int(op.params["rows"])
        cols = int(op.params["cols"])
        if rows <= 0 or cols <= 0 or rows * cols != len(payload):
            raise FluidTransitError("transpose_2d dimensions mismatch payload length.")
        return [payload[r * cols + c] for c in range(cols) for r in range(rows)], "int_vector"
    if op.op == "affine_map":
        if not isinstance(payload, list):
            raise FluidTransitError("affine_map requires a list vector payload.")
        sc = int(op.params["scale"])
        bi = int(op.params["bias"])
        mod = int(op.params["modulus"])
        return [(sc * int(x) + bi) % mod for x in payload], "int_vector"
    if op.op == "prefix_scan_mod":
        if not isinstance(payload, list):
            raise FluidTransitError("prefix_scan_mod requires a list vector payload.")
        mod = int(op.params["modulus"])
        out: list[int] = []
        acc = 0
        for x in payload:
            acc = (acc + int(x)) % mod
            out.append(acc)
        return out, "int_vector"
    if op.op == "reduce_sum":
        if not isinstance(payload, list):
            raise FluidTransitError("reduce_sum requires a list vector payload.")
        return {"scalar": sum(int(x) for x in payload)}, "interpretable_scalar"
    if op.op == "reduce_xor":
        if not isinstance(payload, list):
            raise FluidTransitError("reduce_xor requires a list vector payload.")
        acc = 0
        for x in payload:
            acc ^= int(x)
        return {"scalar": acc}, "interpretable_scalar"
    if op.op == "vm_register_poly":
        # Lifted register-machine polynomial evaluator: y_i = (c2 * x_i^2 + c1 * x_i + c0) mod M
        if not isinstance(payload, list):
            raise FluidTransitError("vm_register_poly requires a list vector payload.")
        c2 = int(op.params.get("c2", 0))
        c1 = int(op.params.get("c1", 1))
        c0 = int(op.params.get("c0", 0))
        mod = int(op.params["modulus"])
        return [(c2 * int(x) * int(x) + c1 * int(x) + c0) % mod for x in payload], "int_vector"
    raise FluidTransitError(f"Unsupported transit operation {op.op!r}.")


@dataclass(frozen=True, slots=True)
class TransitHopReceipt:
    """Cryptographic receipt emitted by one stream interpreter hop along a communication path."""

    packet_id: str
    hop_index: int
    interpreter_id: str
    ops_executed: tuple[str, ...]
    input_digest: str
    output_digest: str
    prev_hop_digest: str
    hop_digest: str = ""

    SCHEMA = "vld.fluid-transit-hop-receipt.v1"

    def __post_init__(self) -> None:
        computed = domain_digest(
            self.SCHEMA,
            {
                "hop_index": self.hop_index,
                "input_digest": self.input_digest,
                "interpreter_id": self.interpreter_id,
                "ops_executed": list(self.ops_executed),
                "output_digest": self.output_digest,
                "packet_id": self.packet_id,
                "prev_hop_digest": self.prev_hop_digest,
            },
        )
        if self.hop_digest and self.hop_digest != computed:
            raise FluidTransitError("TransitHopReceipt digest tampering detected.")
        object.__setattr__(self, "hop_digest", computed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_digest": self.hop_digest,
            "hop_index": self.hop_index,
            "input_digest": self.input_digest,
            "interpreter_id": self.interpreter_id,
            "ops_executed": list(self.ops_executed),
            "output_digest": self.output_digest,
            "packet_id": self.packet_id,
            "prev_hop_digest": self.prev_hop_digest,
        }


@dataclass(frozen=True, slots=True)
class TransitInformationPacket:
    """Self-describing information frame transforming across stream interpreters toward a target schema."""

    packet_id: str
    current_representation: str
    target_representation: str
    payload: Any
    remaining_ops: tuple[TransitOperation, ...]
    receipts: tuple[TransitHopReceipt, ...] = ()

    def to_wire_bytes(self) -> bytes:
        return canonical_bytes(
            {
                "current_representation": self.current_representation,
                "packet_id": self.packet_id,
                "payload": self.payload,
                "receipts": [r.to_dict() for r in self.receipts],
                "remaining_ops": [o.to_dict() for o in self.remaining_ops],
                "schema": "vld.transit-packet.v1",
                "target_representation": self.target_representation,
            }
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> TransitInformationPacket:
        doc = parse_strict_json(raw)
        return cls(
            packet_id=str(doc["packet_id"]),
            current_representation=str(doc["current_representation"]),
            target_representation=str(doc["target_representation"]),
            payload=doc["payload"],
            remaining_ops=tuple(TransitOperation.from_dict(o) for o in doc["remaining_ops"]),
            receipts=tuple(
                TransitHopReceipt(
                    packet_id=str(r["packet_id"]),
                    hop_index=int(r["hop_index"]),
                    interpreter_id=str(r["interpreter_id"]),
                    ops_executed=tuple(str(x) for x in r["ops_executed"]),
                    input_digest=str(r["input_digest"]),
                    output_digest=str(r["output_digest"]),
                    prev_hop_digest=str(r["prev_hop_digest"]),
                    hop_digest=str(r["hop_digest"]),
                )
                for r in doc.get("receipts", ())
            ),
        )


class StreamInterpreterNode:
    """Machine acting purely as an in-transit stream interpreter with bounded per-hop op budget."""

    def __init__(self, interpreter_id: str, *, max_ops_per_hop: int = 2) -> None:
        if max_ops_per_hop <= 0:
            raise FluidTransitError("max_ops_per_hop must be positive.")
        self.interpreter_id = interpreter_id
        self.max_ops_per_hop = max_ops_per_hop

    def interpret_frame(self, raw_frame: bytes) -> bytes:
        pkt = TransitInformationPacket.from_wire_bytes(raw_frame)
        in_dg = hashlib.sha256(canonical_bytes(pkt.payload)).hexdigest()
        prev_dg = pkt.receipts[-1].hop_digest if pkt.receipts else GENESIS_PREV_DIGEST

        take = min(self.max_ops_per_hop, len(pkt.remaining_ops))
        batch = pkt.remaining_ops[:take]
        rest = pkt.remaining_ops[take:]

        cur_payload = parse_strict_json(canonical_json(pkt.payload))
        cur_rep = pkt.current_representation
        executed_names: list[str] = []
        for op in batch:
            cur_payload, cur_rep = _apply_single_transit_op(cur_payload, op)
            executed_names.append(op.op)

        out_dg = hashlib.sha256(canonical_bytes(cur_payload)).hexdigest()
        rcpt = TransitHopReceipt(
            packet_id=pkt.packet_id,
            hop_index=len(pkt.receipts),
            interpreter_id=self.interpreter_id,
            ops_executed=tuple(executed_names),
            input_digest=in_dg,
            output_digest=out_dg,
            prev_hop_digest=prev_dg,
        )
        next_pkt = TransitInformationPacket(
            packet_id=pkt.packet_id,
            current_representation=cur_rep,
            target_representation=pkt.target_representation,
            payload=cur_payload,
            remaining_ops=rest,
            receipts=(*pkt.receipts, rcpt),
        )
        return next_pkt.to_wire_bytes()


class InTransitPathMachine:
    """Executes lifted computations across a chain of stream interpreters (in-memory or HeadlessOS POSIX pipes)."""

    def __init__(self, interpreters: Sequence[StreamInterpreterNode]) -> None:
        if not interpreters:
            raise FluidTransitError("InTransitPathMachine requires at least one StreamInterpreterNode.")
        self.interpreters = tuple(interpreters)

    @staticmethod
    def naive_hardware_reference(initial_payload: Any, schedule: Sequence[TransitOperation]) -> tuple[Any, str]:
        cur = parse_strict_json(canonical_json(initial_payload))
        rep = "int_vector" if isinstance(cur, list) else "raw"
        for op in schedule:
            cur, rep = _apply_single_transit_op(cur, op)
        return cur, rep

    def execute_in_transit(
        self,
        packet_id: str,
        *,
        initial_payload: Any,
        schedule: Sequence[TransitOperation],
        target_representation: str,
        optimize: bool = True,
        transport: str = "stream",
    ) -> tuple[TransitInformationPacket, dict[str, Any]]:
        vec_len = len(initial_payload) if isinstance(initial_payload, list) else None
        active_ops = optimize_transit_schedule(schedule, vector_length=vec_len) if optimize else tuple(schedule)
        pkt = TransitInformationPacket(
            packet_id=packet_id,
            current_representation="int_vector" if isinstance(initial_payload, list) else "raw",
            target_representation=target_representation,
            payload=parse_strict_json(canonical_json(initial_payload)),
            remaining_ops=active_ops,
        )
        wire = pkt.to_wire_bytes()
        wire_bytes_moved = 0

        if transport == "headless_os_pipe":
            hos = HeadlessOS(page_size=4096)
            p0 = hos.spawn_process("stream.src", (VMInstruction("HALT", ()),))
            for hop_node in self.interpreters:
                hos.posix_invoke(p0, 22, address=64)
                fd_r, fd_w = struct.unpack("<ii", hos._procs[p0].vm_proc.memory.read_bytes(64, 8))
                # Write wire frame into POSIX pipe via process virtual memory and read out on interpreter side
                buf_addr = 256
                hos._procs[p0].vm_proc.memory.write_bytes(buf_addr, wire)
                hos.posix_invoke(p0, 1, fd=fd_w, address=buf_addr, size=len(wire))
                out_addr = buf_addr + len(wire) + 64
                hos.posix_invoke(p0, 0, fd=fd_r, address=out_addr, size=len(wire))
                received = hos._procs[p0].vm_proc.memory.read_bytes(out_addr, len(wire))
                hos.posix_invoke(p0, 3, fd=fd_r)
                hos.posix_invoke(p0, 3, fd=fd_w)
                wire_bytes_moved += len(received)
                wire = hop_node.interpret_frame(received)
        elif transport == "stream":
            for hop_node in self.interpreters:
                wire_bytes_moved += len(wire)
                wire = hop_node.interpret_frame(wire)
        else:
            raise FluidTransitError(f"Unsupported transport {transport!r}.")

        arrived = TransitInformationPacket.from_wire_bytes(wire)
        if arrived.remaining_ops:
            raise FluidTransitError(
                f"Packet {packet_id!r} arrived with {len(arrived.remaining_ops)} unexecuted operations; add path capacity."
            )
        if arrived.current_representation != target_representation:
            raise FluidTransitError(
                f"Arrived representation {arrived.current_representation!r} != target {target_representation!r}."
            )
        return arrived, {
            "ops_after_optimization": len(active_ops),
            "ops_before_optimization": len(schedule),
            "ops_eliminated_in_transit": len(schedule) - len(active_ops),
            "transport": transport,
            "wire_bytes_moved": wire_bytes_moved,
        }


__all__ = [
    "FiniteTransducerSummary",
    "FluidTransitError",
    "InTransitPathMachine",
    "OnlineFlashStreamState",
    "StreamInterpreterNode",
    "TransitHopReceipt",
    "TransitInformationPacket",
    "TransitOperation",
    "TropicalMinPlusMatrix",
    "VirtualMemoryMovementView",
    "fold_cpu_stream_in_transit",
    "optimize_transit_schedule",
    "stream_flash_attention_in_transit",
    "verify_premature_normalization_counterexample",
]
