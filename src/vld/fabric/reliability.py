r"""Phase 4 — Sliding-Window Replay Protection & Crash-Resumable Reliable Channels (`vld.fabric.reliability`).

Mathematical contract:
1. **Sliding-Window Replay Guard (`ReplayWindowGuard`):**
   Maintains per-sender high-water sequence $H_u \in \mathbb{Z}_{\ge -1}$ and $W$-bit bitmap $B_u \in [0, 2^W)$.
   Sequence $s \ge 0$ from sender $u$ is admitted iff:
     - $s > H_u$ (advances $H_u \leftarrow s$ and shifts $B_u$), OR
     - $0 \le H_u - s < W$ and bit $(H_u - s)$ of $B_u$ is $0$ (marks bit $1$).
   Duplicates and ancient frames ($s \le H_u - W$) raise `ReplayRejectedError`.
2. **Transport vs. Semantic Identity Separation (`ReliableChannel`):**
   Transport sequence numbers $s_{\text{wire}} \in \mathbb{N}_1$ govern link-level sliding window $W$,
   ACKs, and lease retransmissions, while `op_id` identifies semantic operations end-to-end.
   `export_checkpoint()` / `from_checkpoint()` preserve pending sender frames and receiver deduplication
   tables across process crashes and simulated link drops without re-applying committed `op_id`s.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from vld.canonical import canonical_json, domain_digest, parse_strict_json


class ReliableChannelError(ValueError):
    """Base error for reliable channel framing, window capacity, or conflict violations."""


class ReliableChannelWouldBlock(ReliableChannelError):
    """Raised when the bounded sender window $|P| = W$ has no free in-flight slot."""


class ReliableChannelConflictError(ReliableChannelError):
    """Raised when a duplicate transport sequence or `op_id` arrives with a conflicting payload digest."""


class ReplayRejectedError(PermissionError):
    """Raised when an envelope sequence number is replayed or falls behind the sliding window."""


class ReplayWindowGuard:
    r"""Per-sender anti-replay sliding window ($W \ge 1$ slots) with crash-resumable state export."""

    def __init__(self, window_size: int = 64) -> None:
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0 or window_size > 4096:
            raise ValueError("window_size must be an integer in [1, 4096].")
        self.window_size = window_size
        self._mask_limit = (1 << window_size) - 1
        self._state: dict[str, tuple[int, int]] = {}  # sender_id -> (high_water, bitmask)

    def check_and_record(self, sender_id: str, sequence: int) -> None:
        """Validate `(sender_id, sequence)` against the sliding window and mark seen atomically."""
        if not isinstance(sender_id, str) or not sender_id.strip():
            raise ReplayRejectedError("sender_id must be a non-empty string.")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ReplayRejectedError("sequence must be a non-negative integer.")

        hw, mask = self._state.get(sender_id, (-1, 0))
        if sequence > hw:
            shift = sequence - hw
            if hw < 0 or shift >= self.window_size:
                new_mask = 1
            else:
                new_mask = ((mask << shift) | 1) & self._mask_limit
            self._state[sender_id] = (sequence, new_mask)
            return

        delta = hw - sequence
        if delta >= self.window_size:
            raise ReplayRejectedError(
                f"Sequence {sequence} from {sender_id!r} is outside sliding window [{hw - self.window_size + 1}, {hw}]."
            )
        bit = 1 << delta
        if mask & bit:
            raise ReplayRejectedError(f"Duplicate replay of sequence {sequence} from {sender_id!r} rejected.")
        self._state[sender_id] = (hw, mask | bit)

    def export_state(self) -> dict[str, Any]:
        return {
            "schema": "vld.replay-window.v1",
            "senders": {k: {"bitmask_hex": hex(m)[2:], "high_water": hw} for k, (hw, m) in sorted(self._state.items())},
            "window_size": self.window_size,
        }

    @classmethod
    def from_state(cls, data: Mapping[str, Any]) -> ReplayWindowGuard:
        if not isinstance(data, Mapping) or data.get("schema") != "vld.replay-window.v1":
            raise ValueError("Invalid ReplayWindowGuard state.")
        guard = cls(window_size=int(data["window_size"]))
        for k, entry in data["senders"].items():
            guard._state[str(k)] = (int(entry["high_water"]), int(str(entry["bitmask_hex"]), 16))
        return guard


@dataclass(frozen=True, slots=True)
class ReliabilityPolicy:
    """Sliding window, lease timeout, and retransmission bounds for `ReliableChannel`."""

    window_size: int = 32
    lease_ticks: int = 3
    max_attempts: int = 5

    def __post_init__(self) -> None:
        for name in ("window_size", "lease_ticks", "max_attempts"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ReliableChannelError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class WireFrame:
    """Transport frame separating link-local `wire_seq` from end-to-end semantic `op_id`."""

    channel_id: str
    kind: str  # "DATA" or "ACK"
    sender: str
    recipient: str
    wire_seq: int
    op_id: str = ""
    payload: Any = None
    frame_digest: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("DATA", "ACK"):
            raise ReliableChannelError(f"Invalid WireFrame kind {self.kind!r}.")
        if isinstance(self.wire_seq, bool) or not isinstance(self.wire_seq, int) or self.wire_seq <= 0:
            raise ReliableChannelError("wire_seq must be a positive integer.")
        if self.kind == "DATA" and (not isinstance(self.op_id, str) or not self.op_id.strip()):
            raise ReliableChannelError("DATA WireFrame requires a non-empty semantic op_id.")
        clean_p = parse_strict_json(canonical_json(self.payload))
        object.__setattr__(self, "payload", clean_p)
        computed = domain_digest("vld.wire-frame.v1", self._body())
        if self.frame_digest and self.frame_digest != computed:
            raise ReliableChannelError("WireFrame digest mismatch.")
        object.__setattr__(self, "frame_digest", computed)

    def _body(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "kind": self.kind,
            "op_id": self.op_id,
            "payload": self.payload,
            "recipient": self.recipient,
            "sender": self.sender,
            "wire_seq": self.wire_seq,
        }

    @property
    def payload_digest(self) -> str:
        return domain_digest("vld.wire-payload.v1", {"op_id": self.op_id, "payload": self.payload})

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "frame_digest": self.frame_digest}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WireFrame:
        req = {"channel_id", "frame_digest", "kind", "op_id", "payload", "recipient", "sender", "wire_seq"}
        if not isinstance(data, Mapping) or set(data.keys()) != req:
            raise ReliableChannelError("Invalid WireFrame keyset.")
        return cls(
            channel_id=data["channel_id"],
            kind=data["kind"],
            sender=data["sender"],
            recipient=data["recipient"],
            wire_seq=data["wire_seq"],
            op_id=data["op_id"],
            payload=data["payload"],
            frame_digest=data["frame_digest"],
        )


class LossyInMemoryLink:
    """Deterministic duplex message link supporting programmable drop predicates for fault-injection tests."""

    def __init__(self, drop_predicate: Callable[[WireFrame], bool] | None = None) -> None:
        self.drop_predicate = drop_predicate
        self._queues: dict[str, deque[dict[str, Any]]] = {}
        self.dropped_count = 0
        self.transmitted_count = 0

    def send(self, frame: WireFrame) -> None:
        self.transmitted_count += 1
        if self.drop_predicate is not None and self.drop_predicate(frame):
            self.dropped_count += 1
            return
        self._queues.setdefault(frame.recipient, deque()).append(frame.to_dict())

    def pop_for(self, recipient: str) -> WireFrame | None:
        q = self._queues.get(recipient)
        if not q:
            return None
        return WireFrame.from_dict(q.popleft())


@dataclass
class _PendingEntry:
    frame: WireFrame
    attempts: int
    last_sent_tick: int


class ReliableChannel:
    """Crash-resumable reliable channel separating transport sequence numbers from semantic `op_id`s."""

    def __init__(
        self,
        *,
        channel_id: str,
        local_peer: str,
        remote_peer: str,
        send_fn: Callable[[WireFrame], None],
        policy: ReliabilityPolicy | None = None,
    ) -> None:
        if not channel_id.strip() or not local_peer.strip() or not remote_peer.strip():
            raise ReliableChannelError("channel_id, local_peer, and remote_peer must be non-empty.")
        self.channel_id = channel_id
        self.local_peer = local_peer
        self.remote_peer = remote_peer
        self._send_fn = send_fn
        self.policy = policy or ReliabilityPolicy()

        self._tick = 0
        self._next_wire_seq = 1
        self._pending: dict[int, _PendingEntry] = {}
        self._sent_ops: dict[str, tuple[int, str]] = {}  # op_id -> (wire_seq, payload_digest)
        self.exhausted_sequences: list[int] = []

        self._next_recv_seq = 1
        self._recv_seq_digests: dict[int, str] = {}
        self._committed_ops: dict[str, str] = {}  # op_id -> payload_digest
        self._out_of_order_buffer: dict[int, WireFrame] = {}

    @property
    def available_window(self) -> int:
        return max(0, self.policy.window_size - len(self._pending))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def next_receive_sequence(self) -> int:
        return self._next_recv_seq

    def send_operation(self, op_id: str, payload: Any) -> int:
        """Send semantic operation `op_id`; idempotent if `op_id` is already in-flight/sent with same payload."""
        probe_frame = WireFrame(
            channel_id=self.channel_id,
            kind="DATA",
            sender=self.local_peer,
            recipient=self.remote_peer,
            wire_seq=1,
            op_id=op_id,
            payload=payload,
        )
        p_digest = probe_frame.payload_digest
        if op_id in self._sent_ops:
            prev_seq, prev_digest = self._sent_ops[op_id]
            if prev_digest != p_digest:
                raise ReliableChannelConflictError(
                    f"Sender op_id {op_id!r} reused with conflicting payload."
                )
            if prev_seq in self._pending:
                self._send_fn(self._pending[prev_seq].frame)
            return prev_seq

        if len(self._pending) >= self.policy.window_size:
            raise ReliableChannelWouldBlock(f"Sender window ({self.policy.window_size}) is full.")

        seq = self._next_wire_seq
        self._next_wire_seq += 1
        frame = WireFrame(
            channel_id=self.channel_id,
            kind="DATA",
            sender=self.local_peer,
            recipient=self.remote_peer,
            wire_seq=seq,
            op_id=op_id,
            payload=payload,
        )
        self._pending[seq] = _PendingEntry(frame=frame, attempts=1, last_sent_tick=self._tick)
        self._sent_ops[op_id] = (seq, p_digest)
        self._send_fn(frame)
        return seq

    def tick_retransmit(self) -> tuple[int, ...]:
        """Advance logical tick and retransmit any unacknowledged frames whose lease has expired."""
        self._tick += 1
        retransmitted: list[int] = []
        for seq in sorted(list(self._pending.keys())):
            entry = self._pending[seq]
            if self._tick - entry.last_sent_tick >= self.policy.lease_ticks:
                if entry.attempts >= self.policy.max_attempts:
                    del self._pending[seq]
                    self.exhausted_sequences.append(seq)
                else:
                    entry.attempts += 1
                    entry.last_sent_tick = self._tick
                    self._send_fn(entry.frame)
                    retransmitted.append(seq)
        return tuple(retransmitted)

    def receive_frame(
        self,
        frame: WireFrame,
        commit_fn: Callable[[str, Any], None] | None = None,
    ) -> list[tuple[int, str, Any]]:
        """Process one inbound `DATA` or `ACK` frame, delivering contiguous FIFO operations at most once."""
        if frame.channel_id != self.channel_id or frame.recipient != self.local_peer or frame.sender != self.remote_peer:
            raise ReliableChannelError("Inbound WireFrame endpoint/channel mismatch.")

        if frame.kind == "ACK":
            self._pending.pop(frame.wire_seq, None)
            return []

        seq = frame.wire_seq
        p_digest = frame.payload_digest

        # Detect conflicting transport sequence reuse
        if seq in self._recv_seq_digests and self._recv_seq_digests[seq] != p_digest:
            raise ReliableChannelConflictError(f"Conflicting payload on wire_seq {seq}.")

        # Detect conflicting semantic op_id reuse even on a fresh transport sequence!
        if frame.op_id in self._committed_ops and self._committed_ops[frame.op_id] != p_digest:
            raise ReliableChannelConflictError(f"Conflicting payload on semantic op_id {frame.op_id!r}.")

        if seq < self._next_recv_seq:
            # Already delivered: re-emit ACK without re-running commit_fn!
            self._emit_ack(seq)
            return []

        if seq - self._next_recv_seq >= self.policy.window_size:
            raise ReliableChannelWouldBlock(f"Inbound wire_seq {seq} exceeds receiver window.")

        self._recv_seq_digests[seq] = p_digest
        self._out_of_order_buffer[seq] = frame

        delivered: list[tuple[int, str, Any]] = []
        while self._next_recv_seq in self._out_of_order_buffer:
            cur = self._out_of_order_buffer.pop(self._next_recv_seq)
            cur_digest = cur.payload_digest
            if cur.op_id in self._committed_ops:
                if self._committed_ops[cur.op_id] != cur_digest:
                    raise ReliableChannelConflictError(f"Conflicting payload on semantic op_id {cur.op_id!r}.")
            else:
                if commit_fn is not None:
                    commit_fn(cur.op_id, cur.payload)
                self._committed_ops[cur.op_id] = cur_digest
                delivered.append((cur.wire_seq, cur.op_id, cur.payload))
            self._emit_ack(cur.wire_seq)
            self._next_recv_seq += 1

        return delivered

    def _emit_ack(self, wire_seq: int) -> None:
        ack = WireFrame(
            channel_id=self.channel_id,
            kind="ACK",
            sender=self.local_peer,
            recipient=self.remote_peer,
            wire_seq=wire_seq,
        )
        self._send_fn(ack)

    def export_checkpoint(self) -> dict[str, Any]:
        body = {
            "channel_id": self.channel_id,
            "committed_ops": dict(sorted(self._committed_ops.items())),
            "exhausted_sequences": list(self.exhausted_sequences),
            "local_peer": self.local_peer,
            "next_recv_seq": self._next_recv_seq,
            "next_wire_seq": self._next_wire_seq,
            "out_of_order": {str(k): v.to_dict() for k, v in sorted(self._out_of_order_buffer.items())},
            "pending": {
                str(seq): {
                    "attempts": ent.attempts,
                    "frame": ent.frame.to_dict(),
                    "last_sent_tick": ent.last_sent_tick,
                }
                for seq, ent in sorted(self._pending.items())
            },
            "policy": {
                "lease_ticks": self.policy.lease_ticks,
                "max_attempts": self.policy.max_attempts,
                "window_size": self.policy.window_size,
            },
            "recv_seq_digests": {str(k): v for k, v in sorted(self._recv_seq_digests.items())},
            "remote_peer": self.remote_peer,
            "schema": "vld.reliable-channel-checkpoint.v1",
            "sent_ops": {op: [seq, dg] for op, (seq, dg) in sorted(self._sent_ops.items())},
            "tick": self._tick,
        }
        return {**body, "checkpoint_digest": domain_digest("vld.reliable-channel-checkpoint.v1", body)}

    @classmethod
    def from_checkpoint(
        cls,
        data: Mapping[str, Any],
        *,
        send_fn: Callable[[WireFrame], None],
    ) -> ReliableChannel:
        if not isinstance(data, Mapping) or data.get("schema") != "vld.reliable-channel-checkpoint.v1":
            raise ReliableChannelError("Invalid ReliableChannel checkpoint schema.")
        body = {k: v for k, v in data.items() if k != "checkpoint_digest"}
        if domain_digest("vld.reliable-channel-checkpoint.v1", body) != data.get("checkpoint_digest"):
            raise ReliableChannelError("ReliableChannel checkpoint digest mismatch.")

        pol_raw = data["policy"]
        pol = ReliabilityPolicy(
            window_size=int(pol_raw["window_size"]),
            lease_ticks=int(pol_raw["lease_ticks"]),
            max_attempts=int(pol_raw["max_attempts"]),
        )
        ch = cls(
            channel_id=str(data["channel_id"]),
            local_peer=str(data["local_peer"]),
            remote_peer=str(data["remote_peer"]),
            send_fn=send_fn,
            policy=pol,
        )
        ch._tick = int(data["tick"])
        ch._next_wire_seq = int(data["next_wire_seq"])
        ch._next_recv_seq = int(data["next_recv_seq"])
        ch.exhausted_sequences = [int(x) for x in data["exhausted_sequences"]]
        ch._committed_ops = {str(k): str(v) for k, v in data["committed_ops"].items()}
        ch._recv_seq_digests = {int(k): str(v) for k, v in data["recv_seq_digests"].items()}
        ch._sent_ops = {str(op): (int(pair[0]), str(pair[1])) for op, pair in data["sent_ops"].items()}
        ch._out_of_order_buffer = {int(k): WireFrame.from_dict(v) for k, v in data["out_of_order"].items()}
        ch._pending = {
            int(k): _PendingEntry(
                frame=WireFrame.from_dict(ent["frame"]),
                attempts=int(ent["attempts"]),
                last_sent_tick=int(ent["last_sent_tick"]),
            )
            for k, ent in data["pending"].items()
        }
        return ch


__all__ = [
    "LossyInMemoryLink",
    "ReliabilityPolicy",
    "ReliableChannel",
    "ReliableChannelConflictError",
    "ReliableChannelError",
    "ReliableChannelWouldBlock",
    "ReplayRejectedError",
    "ReplayWindowGuard",
    "WireFrame",
]
