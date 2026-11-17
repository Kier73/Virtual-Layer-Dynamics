r"""Phase 6 — Paged KV-State Ground & Crash-Resumable Multi-Host Model-Orbit Continuity (`vld.models.orbit`)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping, Sequence

from vld.accel import NumericEngine, decode_f32_le, encode_f32_le
from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.memory import PagedByteManifest, PagedObjectStore
from vld.models.artifacts import ModelWeightSnapshot, TransformerArchitectureSpec
from vld.models.realizers import MoERoutingReceipt, TransformerLayerRealizer, hidden_state_digest


class ModelOrbitError(ValueError):
    """Raised on invalid KV-cache collapse claims, corrupted orbit handoffs, or lineage mismatches."""


@dataclass(frozen=True, slots=True)
class LayerKVPage:
    """Content-addressed M1 page pair storing `(K, V)` cache states for one layer and token prefix."""

    model_fingerprint: str
    layer_index: int
    seq_len: int
    num_kv_heads: int
    head_dim: int
    k_manifest: PagedByteManifest
    v_manifest: PagedByteManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_dim": self.head_dim,
            "k_manifest": self.k_manifest.to_dict(),
            "layer_index": self.layer_index,
            "model_fingerprint": self.model_fingerprint,
            "num_kv_heads": self.num_kv_heads,
            "seq_len": self.seq_len,
            "v_manifest": self.v_manifest.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayerKVPage:
        return cls(
            model_fingerprint=str(data["model_fingerprint"]),
            layer_index=int(data["layer_index"]),
            seq_len=int(data["seq_len"]),
            num_kv_heads=int(data["num_kv_heads"]),
            head_dim=int(data["head_dim"]),
            k_manifest=PagedByteManifest.from_dict(data["k_manifest"]),
            v_manifest=PagedByteManifest.from_dict(data["v_manifest"]),
        )


class PagedKVCacheGround:
    """M1-backed KV-State Ground supporting verified causal prefix reuse and refusing invalid softmax collapse."""

    def __init__(self, store: PagedObjectStore) -> None:
        self.store = store
        self._prefix_index: dict[tuple[str, tuple[int, ...]], dict[int, LayerKVPage]] = {}

    @staticmethod
    def attempt_affine_kv_collapse() -> None:
        """Explicitly refuse linear/affine sufficient-state collapse across nonlinear softmax attention."""
        raise ModelOrbitError(
            "Refused: causal softmax attention exp(q k^T / sqrt(d)) is non-bilinear; "
            "prefix KV states cannot be collapsed into a fixed-width affine summary without approximation."
        )

    def store_layer_kv(
        self,
        arch: TransformerArchitectureSpec,
        *,
        layer_index: int,
        token_prefix: Sequence[int],
        k_3d: Sequence[Sequence[Sequence[float]]],
        v_3d: Sequence[Sequence[Sequence[float]]],
    ) -> LayerKVPage:
        seq_len = len(token_prefix)
        if len(k_3d) != seq_len or len(v_3d) != seq_len:
            raise ModelOrbitError("KV cache length disagrees with token_prefix length.")
        flat_k = [float(x) for tok in k_3d for hv in tok for x in hv]
        flat_v = [float(x) for tok in v_3d for hv in tok for x in hv]
        km = self.store.put_bytes(encode_f32_le(flat_k))
        vm = self.store.put_bytes(encode_f32_le(flat_v))
        page = LayerKVPage(
            model_fingerprint=arch.fingerprint,
            layer_index=layer_index,
            seq_len=seq_len,
            num_kv_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim,
            k_manifest=km,
            v_manifest=vm,
        )
        key = (arch.fingerprint, tuple(int(t) for t in token_prefix))
        self._prefix_index.setdefault(key, {})[layer_index] = page
        return page

    def register_prefix_pages(
        self,
        arch: TransformerArchitectureSpec,
        token_prefix: Sequence[int],
        pages_by_layer: Mapping[int, LayerKVPage],
    ) -> None:
        """Index verified layer KV pages for `token_prefix` on a resumed host."""
        tok_tup = tuple(int(t) for t in token_prefix)
        if not tok_tup or not pages_by_layer:
            return
        slot = self._prefix_index.setdefault((arch.fingerprint, tok_tup), {})
        for l_idx, page in pages_by_layer.items():
            if page.model_fingerprint != arch.fingerprint or page.seq_len != len(tok_tup):
                raise ModelOrbitError("KV page metadata disagrees with registered token_prefix.")
            slot[int(l_idx)] = page

    def load_layer_kv(
        self,
        page: LayerKVPage,
        *,
        remote_range_fetcher: Callable[[str, int, int], bytes] | None = None,
    ) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
        def _read(m: PagedByteManifest) -> list[float]:
            if remote_range_fetcher is not None:
                raw = remote_range_fetcher(m.manifest_digest, 0, m.size_bytes)
                if len(raw) != m.size_bytes or hashlib.sha256(raw).hexdigest() != m.content_digest:
                    raise ModelOrbitError("Remote KV page failed SHA-256 content_digest verification.")
            else:
                raw = self.store.read_bytes(m)
            return decode_f32_le(raw)

        fk = _read(page.k_manifest)
        fv = _read(page.v_manifest)
        s, nh, hd = page.seq_len, page.num_kv_heads, page.head_dim
        stride_s = nh * hd
        if len(fk) != s * stride_s or len(fv) != s * stride_s:
            raise ModelOrbitError("Decoded KV page float count disagrees with LayerKVPage dimensions.")
        k_3d = [
            [fk[t * stride_s + h * hd : t * stride_s + (h + 1) * hd] for h in range(nh)]
            for t in range(s)
        ]
        v_3d = [
            [fv[t * stride_s + h * hd : t * stride_s + (h + 1) * hd] for h in range(nh)]
            for t in range(s)
        ]
        return k_3d, v_3d

    def lookup_longest_prefix(
        self,
        arch: TransformerArchitectureSpec,
        token_ids: Sequence[int],
    ) -> tuple[int, dict[int, LayerKVPage]]:
        r"""Find longest strict causal prefix `token_ids[:L]` ($L < |\text{token\_ids}|$) with complete all-layer KV pages."""
        toks = tuple(int(t) for t in token_ids)
        for length in range(len(toks) - 1, 0, -1):
            sub = toks[:length]
            pages = self._prefix_index.get((arch.fingerprint, sub))
            if pages is not None and len(pages) == arch.num_hidden_layers:
                return length, dict(pages)
        return 0, {}


@dataclass(frozen=True, slots=True)
class ModelOrbitCheckpoint:
    """Portable, digest-fenced multi-host decode & mid-layer orbit checkpoint."""

    model_fingerprint: str
    weight_snapshot_identity: str
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    active_step_tokens: tuple[int, ...]
    completed_layers_in_step: int
    hidden_manifest: PagedByteManifest | None
    hidden_seq_len: int
    hidden_dim: int
    kv_pages_by_layer: Mapping[int, LayerKVPage]
    step_kv_pages_by_layer: Mapping[int, LayerKVPage]
    checkpoint_digest: str = ""

    SCHEMA = "vld.model-orbit-checkpoint.v1"

    def __post_init__(self) -> None:
        computed = domain_digest(self.SCHEMA, self.body_without_digest())
        if self.checkpoint_digest and self.checkpoint_digest != computed:
            raise ModelOrbitError("ModelOrbitCheckpoint digest mismatch (tampered or corrupted checkpoint).")
        object.__setattr__(self, "checkpoint_digest", computed)

    def body_without_digest(self) -> dict[str, Any]:
        return {
            "active_step_tokens": list(self.active_step_tokens),
            "completed_layers_in_step": self.completed_layers_in_step,
            "generated_tokens": list(self.generated_tokens),
            "hidden_dim": self.hidden_dim,
            "hidden_manifest": None if self.hidden_manifest is None else self.hidden_manifest.to_dict(),
            "hidden_seq_len": self.hidden_seq_len,
            "kv_pages_by_layer": {str(k): v.to_dict() for k, v in sorted(self.kv_pages_by_layer.items())},
            "model_fingerprint": self.model_fingerprint,
            "prompt_tokens": list(self.prompt_tokens),
            "schema": self.SCHEMA,
            "step_kv_pages_by_layer": {str(k): v.to_dict() for k, v in sorted(self.step_kv_pages_by_layer.items())},
            "weight_snapshot_identity": self.weight_snapshot_identity,
        }

    def to_json(self) -> str:
        body = self.body_without_digest()
        body["checkpoint_digest"] = self.checkpoint_digest
        return canonical_json(body)

    @classmethod
    def from_json(cls, raw: str | bytes) -> ModelOrbitCheckpoint:
        data = parse_strict_json(raw)
        if not isinstance(data, dict) or data.get("schema") != cls.SCHEMA:
            raise ModelOrbitError("Invalid ModelOrbitCheckpoint JSON schema.")
        return cls(
            model_fingerprint=str(data["model_fingerprint"]),
            weight_snapshot_identity=str(data["weight_snapshot_identity"]),
            prompt_tokens=tuple(int(x) for x in data["prompt_tokens"]),
            generated_tokens=tuple(int(x) for x in data["generated_tokens"]),
            active_step_tokens=tuple(int(x) for x in data["active_step_tokens"]),
            completed_layers_in_step=int(data["completed_layers_in_step"]),
            hidden_manifest=(
                None if data["hidden_manifest"] is None else PagedByteManifest.from_dict(data["hidden_manifest"])
            ),
            hidden_seq_len=int(data["hidden_seq_len"]),
            hidden_dim=int(data["hidden_dim"]),
            kv_pages_by_layer={int(k): LayerKVPage.from_dict(v) for k, v in data["kv_pages_by_layer"].items()},
            step_kv_pages_by_layer={int(k): LayerKVPage.from_dict(v) for k, v in data["step_kv_pages_by_layer"].items()},
            checkpoint_digest=str(data["checkpoint_digest"]),
        )


@dataclass(frozen=True, slots=True)
class GenerationResult:
    prompt_tokens: tuple[int, ...]
    generated_tokens: tuple[int, ...]
    all_tokens: tuple[int, ...]
    last_step_logits: list[float]
    final_checkpoint: ModelOrbitCheckpoint
    moe_receipts: tuple[MoERoutingReceipt, ...]
    loaded_tensor_names: tuple[str, ...]


class AutoregressiveModelRunner:
    """Crash-resumable multi-token autoregressive LLM runner over M1 KV pages and remote/local shards."""

    def __init__(
        self,
        arch: TransformerArchitectureSpec,
        snapshot: ModelWeightSnapshot,
        state_store: PagedObjectStore,
        *,
        weight_store: PagedObjectStore | None = None,
        remote_weight_fetcher: Callable[[str, int, int], bytes] | None = None,
        engine: NumericEngine | None = None,
    ) -> None:
        self.arch = arch
        self.snapshot = snapshot
        self.state_store = state_store
        self.kv_ground = PagedKVCacheGround(state_store)
        self.realizer = TransformerLayerRealizer(
            arch,
            snapshot,
            store=weight_store if weight_store is not None else (state_store if remote_weight_fetcher is None else None),
            remote_range_fetcher=remote_weight_fetcher,
            engine=engine,
        )

    def _save_hidden(self, hidden_2d: Sequence[Sequence[float]]) -> PagedByteManifest:
        flat = [float(v) for row in hidden_2d for v in row]
        return self.state_store.put_bytes(encode_f32_le(flat))

    def _load_hidden(self, manifest: PagedByteManifest, seq_len: int) -> list[list[float]]:
        vals = decode_f32_le(self.state_store.read_bytes(manifest))
        h = self.arch.hidden_size
        return [vals[r * h : (r + 1) * h] for r in range(seq_len)]

    def generate(
        self,
        prompt_tokens: Sequence[int],
        *,
        max_new_tokens: int = 4,
        resume_checkpoint: ModelOrbitCheckpoint | None = None,
        expected_checkpoint_digest: str | None = None,
        on_checkpoint: Callable[[ModelOrbitCheckpoint, int, int], None] | None = None,
    ) -> GenerationResult:
        """Run or resume deterministic greedy autoregressive generation with per-layer M1 orbit checkpoints."""
        if max_new_tokens < 1:
            raise ModelOrbitError("max_new_tokens must be >= 1.")

        all_moe_receipts: list[MoERoutingReceipt] = []
        all_loaded_tensors: list[str] = []

        if resume_checkpoint is not None:
            if expected_checkpoint_digest is not None and resume_checkpoint.checkpoint_digest != expected_checkpoint_digest:
                raise ModelOrbitError("Resume checkpoint disagrees with external high-water digest pin.")
            if (
                resume_checkpoint.model_fingerprint != self.arch.fingerprint
                or resume_checkpoint.weight_snapshot_identity != self.snapshot.identity
            ):
                raise ModelOrbitError("Resume checkpoint belongs to a different model or weight snapshot.")
            if prompt_tokens and tuple(int(t) for t in prompt_tokens) != resume_checkpoint.prompt_tokens:
                raise ModelOrbitError("Supplied prompt_tokens disagrees with resume_checkpoint.prompt_tokens.")
            prompt_tup = resume_checkpoint.prompt_tokens
            if not prompt_tup:
                raise ModelOrbitError("Checkpoint prompt_tokens cannot be empty.")
            generated = list(resume_checkpoint.generated_tokens)
            if len(generated) > max_new_tokens:
                raise ModelOrbitError("Checkpoint already exceeds requested max_new_tokens.")
            committed_kv = dict(resume_checkpoint.kv_pages_by_layer)
            step_tokens = list(resume_checkpoint.active_step_tokens)
            start_layer = resume_checkpoint.completed_layers_in_step
            step_kv = dict(resume_checkpoint.step_kv_pages_by_layer)
            if not (0 <= start_layer <= self.arch.num_hidden_layers):
                raise ModelOrbitError("Checkpoint completed_layers_in_step out of bounds.")
            if not step_tokens:
                raise ModelOrbitError("Checkpoint active_step_tokens cannot be empty.")
            if start_layer > 0:
                if (
                    resume_checkpoint.hidden_manifest is None
                    or resume_checkpoint.hidden_seq_len != len(step_tokens)
                    or resume_checkpoint.hidden_dim != self.arch.hidden_size
                    or set(step_kv.keys()) != set(range(start_layer))
                ):
                    raise ModelOrbitError(
                        "Malformed mid-layer orbit checkpoint: missing hidden_manifest or partial layer KV pages."
                    )
            hidden: list[list[float]] | None = (
                self._load_hidden(resume_checkpoint.hidden_manifest, resume_checkpoint.hidden_seq_len)
                if resume_checkpoint.hidden_manifest is not None
                else None
            )
            full_cur_tokens = (*prompt_tup, *generated)
            if committed_kv:
                committed_len = next(iter(committed_kv.values())).seq_len
                if committed_len + len(step_tokens) != len(full_cur_tokens) and len(generated) < max_new_tokens:
                    raise ModelOrbitError("Checkpoint KV prefix length + active_step_tokens disagrees with sequence length.")
                self.kv_ground.register_prefix_pages(self.arch, full_cur_tokens[:committed_len], committed_kv)
            if step_kv:
                self.kv_ground.register_prefix_pages(self.arch, full_cur_tokens, step_kv)
        else:
            if not prompt_tokens:
                raise ModelOrbitError("prompt_tokens must be non-empty.")
            prompt_tup = tuple(int(t) for t in prompt_tokens)
            generated = []
            prefix_len, cached_pages = self.kv_ground.lookup_longest_prefix(self.arch, prompt_tup)
            committed_kv = cached_pages
            step_tokens = list(prompt_tup[prefix_len:])
            start_layer = 0
            step_kv = {}
            hidden = None

        last_logits: list[float] = []
        latest_ckpt = resume_checkpoint

        while len(generated) < max_new_tokens:
            if hidden is None:
                hidden = self.realizer.embed_tokens(step_tokens)

            full_prefix_after_step = (*prompt_tup, *generated)
            for l_idx in range(start_layer, self.arch.num_hidden_layers):
                past_k: list[list[list[float]]] = []
                past_v: list[list[list[float]]] = []
                if l_idx in committed_kv:
                    past_k, past_v = self.kv_ground.load_layer_kv(committed_kv[l_idx])

                layer_out = self.realizer.forward_layer(l_idx, hidden, past_k=past_k, past_v=past_v)
                hidden = layer_out.hidden_states
                all_moe_receipts.extend(layer_out.moe_receipts)
                all_loaded_tensors.extend(layer_out.loaded_tensor_names)

                kv_page = self.kv_ground.store_layer_kv(
                    self.arch,
                    layer_index=l_idx,
                    token_prefix=full_prefix_after_step,
                    k_3d=layer_out.k_cache_full,
                    v_3d=layer_out.v_cache_full,
                )
                step_kv[l_idx] = kv_page
                h_manifest = self._save_hidden(hidden)
                latest_ckpt = ModelOrbitCheckpoint(
                    model_fingerprint=self.arch.fingerprint,
                    weight_snapshot_identity=self.snapshot.identity,
                    prompt_tokens=prompt_tup,
                    generated_tokens=tuple(generated),
                    active_step_tokens=tuple(step_tokens),
                    completed_layers_in_step=l_idx + 1,
                    hidden_manifest=h_manifest,
                    hidden_seq_len=len(hidden),
                    hidden_dim=self.arch.hidden_size,
                    kv_pages_by_layer=committed_kv,
                    step_kv_pages_by_layer=step_kv,
                )
                if on_checkpoint is not None:
                    on_checkpoint(latest_ckpt, len(generated), l_idx)

            logits_2d = self.realizer.final_logits(hidden)
            last_logits = logits_2d[-1]
            # Deterministic argmax with lowest-token-ID tie break
            next_tok = min(range(len(last_logits)), key=lambda idx: (-last_logits[idx], idx))
            generated.append(next_tok)
            committed_kv = dict(step_kv)
            step_kv = {}
            step_tokens = [next_tok]
            start_layer = 0
            hidden = None

            latest_ckpt = ModelOrbitCheckpoint(
                model_fingerprint=self.arch.fingerprint,
                weight_snapshot_identity=self.snapshot.identity,
                prompt_tokens=prompt_tup,
                generated_tokens=tuple(generated),
                active_step_tokens=tuple(step_tokens),
                completed_layers_in_step=0,
                hidden_manifest=None,
                hidden_seq_len=0,
                hidden_dim=self.arch.hidden_size,
                kv_pages_by_layer=committed_kv,
                step_kv_pages_by_layer={},
            )
            if on_checkpoint is not None:
                on_checkpoint(latest_ckpt, len(generated), -1)

        assert latest_ckpt is not None
        _ = hidden_state_digest([[last_logits[0]]]) if last_logits else ""
        return GenerationResult(
            prompt_tokens=prompt_tup,
            generated_tokens=tuple(generated),
            all_tokens=(*prompt_tup, *generated),
            last_step_logits=last_logits,
            final_checkpoint=latest_ckpt,
            moe_receipts=tuple(all_moe_receipts),
            loaded_tensor_names=tuple(all_loaded_tensors),
        )


__all__ = [
    "AutoregressiveModelRunner",
    "GenerationResult",
    "LayerKVPage",
    "ModelOrbitCheckpoint",
    "ModelOrbitError",
    "PagedKVCacheGround",
]
