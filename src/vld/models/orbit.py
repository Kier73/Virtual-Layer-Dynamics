r"""Phase 6 — Paged KV-State Ground & Crash-Resumable Multi-Host Model-Orbit Continuity (`vld.models.orbit`)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Mapping, Sequence

from vld.accel import NumericEngine, decode_f32_le, encode_f32_le
from vld.canonical import canonical_json, domain_digest, parse_strict_json
from vld.memory import PagedByteManifest, PagedObjectStore
from vld.models.artifacts import LazySafetensorsWeightGround, ModelWeightSnapshot, TransformerArchitectureSpec
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

    def referenced_manifests(self) -> tuple[PagedByteManifest, ...]:
        """Return all M1 manifests (`K`, `V`, and active `hidden_manifest`) required to resume this checkpoint (`G15`)."""
        seen: dict[str, PagedByteManifest] = {}
        if self.hidden_manifest is not None:
            seen[self.hidden_manifest.manifest_digest] = self.hidden_manifest
        for page in (*self.kv_pages_by_layer.values(), *self.step_kv_pages_by_layer.values()):
            seen[page.k_manifest.manifest_digest] = page.k_manifest
            seen[page.v_manifest.manifest_digest] = page.v_manifest
        return tuple(seen[d] for d in sorted(seen.keys()))

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
        snapshot: ModelWeightSnapshot | LazySafetensorsWeightGround,
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


# ============================================================================
# AKS-001 / G18: Exact Sufficient-State Affine-Kernel Attention Ground over F_p
# ============================================================================


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # Deterministic Miller-Rabin for all 64-bit integers + small prime bases
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
            break
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True, slots=True)
class AffineKernelQueryReceipt:
    """Verified constant-state query receipt over `AffineKernelAttentionGround` (`AKS-001`, `G18`)."""

    ground_fingerprint: str
    prefix_length: int
    head_digest: str
    query: tuple[int, ...]
    numerator: tuple[int, ...]
    mass: int
    normalized: bool
    value: tuple[int, ...]
    receipt_digest: str


class AffineKernelAttentionGround:
    r"""Exact fixed-size sufficient-state attention Ground for $K(q, k) = (1 + \langle q, k \rangle) \bmod p$ (`G18`).

    Retains $S_T = \sum_{t=1}^T \phi(k_t) \otimes v_t \in \mathbb{F}_p^{(d_k+1)\times d_v}$ and
    $z_T = \sum_{t=1}^T \phi(k_t) \in \mathbb{F}_p^{d_k+1}$ where $\phi(x) = (1, x_1, \dots, x_{d_k})$,
    answering arbitrary causal prefix queries in $O(d_k d_v)$ independent of prefix length $T$,
    while binding every admitted `(k_t, v_t)` into a SHA-256 lineage chain (`head_digest`).
    """

    def __init__(self, *, modulus_p: int, key_dim: int, value_dim: int) -> None:
        if isinstance(modulus_p, bool) or not isinstance(modulus_p, int) or not _is_prime(modulus_p):
            raise ModelOrbitError(f"modulus_p must be a prime integer >= 2, got {modulus_p!r}.")
        if isinstance(key_dim, bool) or not isinstance(key_dim, int) or key_dim < 1:
            raise ModelOrbitError("key_dim must be a positive integer.")
        if isinstance(value_dim, bool) or not isinstance(value_dim, int) or value_dim < 1:
            raise ModelOrbitError("value_dim must be a positive integer.")
        self.modulus_p = modulus_p
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.prefix_length = 0
        self.head_digest = "0" * 64
        self._state_s: list[list[int]] = [[0] * value_dim for _ in range(key_dim + 1)]
        self._state_z: list[int] = [0] * (key_dim + 1)

    @property
    def ground_fingerprint(self) -> str:
        return domain_digest(
            "vld.aks-attention-ground.v1",
            {"key_dim": self.key_dim, "modulus_p": self.modulus_p, "value_dim": self.value_dim},
        )

    @property
    def state_shape(self) -> tuple[tuple[int, int], int]:
        return ((self.key_dim + 1, self.value_dim), self.key_dim + 1)

    def admit_token(self, key: Sequence[int], value: Sequence[int]) -> str:
        """Update sufficient state `(S_T, z_T)` in `O(d_k * d_v)` and advance token lineage digest."""
        if len(key) != self.key_dim or len(value) != self.value_dim:
            raise ModelOrbitError("Token key/value dimension mismatch in AffineKernelAttentionGround.")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in (*key, *value)):
            raise ModelOrbitError("Token key/value entries must be exact integers.")
        p = self.modulus_p
        k_mod = [x % p for x in key]
        v_mod = [y % p for y in value]
        feat = [1, *k_mod]
        for a, f_a in enumerate(feat):
            self._state_z[a] = (self._state_z[a] + f_a) % p
            row = self._state_s[a]
            for b, v_b in enumerate(v_mod):
                row[b] = (row[b] + f_a * v_b) % p
        self.prefix_length += 1
        self.head_digest = domain_digest(
            "vld.aks-token-evidence.v1",
            {
                "ground_fingerprint": self.ground_fingerprint,
                "key": k_mod,
                "prefix_index": self.prefix_length,
                "prev_digest": self.head_digest,
                "state_s": self._state_s,
                "state_z": self._state_z,
                "value": v_mod,
            },
        )
        return self.head_digest

    def query(
        self,
        q: Sequence[int],
        *,
        normalized: bool = True,
        expected_head: str | None = None,
    ) -> AffineKernelQueryReceipt:
        """Evaluate exact affine-kernel attention at `q` in `O(d_k * d_v)` without replaying token history."""
        if expected_head is not None and expected_head != self.head_digest:
            raise ModelOrbitError("Trusted receiver expected_head pin disagrees with current lineage head_digest.")
        if len(q) != self.key_dim or any(isinstance(x, bool) or not isinstance(x, int) for x in q):
            raise ModelOrbitError("Query vector must contain key_dim exact integers.")
        p = self.modulus_p
        q_mod = tuple(x % p for x in q)
        feat = (1, *q_mod)
        mass = sum(feat[a] * self._state_z[a] for a in range(self.key_dim + 1)) % p
        num = tuple(
            sum(feat[a] * self._state_s[a][b] for a in range(self.key_dim + 1)) % p
            for b in range(self.value_dim)
        )
        if normalized:
            if mass == 0:
                raise ModelOrbitError(f"Zero kernel mass denominator in F_{p}; normalized observation undefined.")
            inv_mass = pow(mass, -1, p)
            out = tuple((n_b * inv_mass) % p for n_b in num)
        else:
            out = num
        body = {
            "ground_fingerprint": self.ground_fingerprint,
            "head_digest": self.head_digest,
            "mass": mass,
            "normalized": normalized,
            "numerator": list(num),
            "prefix_length": self.prefix_length,
            "query": list(q_mod),
            "value": list(out),
        }
        return AffineKernelQueryReceipt(
            ground_fingerprint=self.ground_fingerprint,
            prefix_length=self.prefix_length,
            head_digest=self.head_digest,
            query=q_mod,
            numerator=num,
            mass=mass,
            normalized=normalized,
            value=out,
            receipt_digest=domain_digest("vld.aks-query-receipt.v1", body),
        )

    def to_snapshot(self) -> dict[str, Any]:
        body = {
            "ground_fingerprint": self.ground_fingerprint,
            "head_digest": self.head_digest,
            "key_dim": self.key_dim,
            "modulus_p": self.modulus_p,
            "prefix_length": self.prefix_length,
            "schema": "vld.aks-snapshot.v1",
            "state_s": [list(r) for r in self._state_s],
            "state_z": list(self._state_z),
            "value_dim": self.value_dim,
        }
        return {**body, "snapshot_digest": domain_digest("vld.aks-snapshot.v1", body)}

    @classmethod
    def from_snapshot(cls, data: Mapping[str, Any], *, expected_head: str | None = None) -> AffineKernelAttentionGround:
        if not isinstance(data, Mapping) or data.get("schema") != "vld.aks-snapshot.v1":
            raise ModelOrbitError("Invalid AffineKernelAttentionGround snapshot schema.")
        body = {k: v for k, v in data.items() if k != "snapshot_digest"}
        if data.get("snapshot_digest") != domain_digest("vld.aks-snapshot.v1", body):
            raise ModelOrbitError("AffineKernelAttentionGround snapshot_digest mismatch.")
        inst = cls(modulus_p=int(data["modulus_p"]), key_dim=int(data["key_dim"]), value_dim=int(data["value_dim"]))
        if data.get("ground_fingerprint") != inst.ground_fingerprint:
            raise ModelOrbitError("AffineKernelAttentionGround fingerprint mismatch.")
        if expected_head is not None and data.get("head_digest") != expected_head:
            raise ModelOrbitError("Restored AKS snapshot disagrees with trusted expected_head.")
        s_mat = [[int(v) % inst.modulus_p for v in row] for row in data["state_s"]]
        z_vec = [int(v) % inst.modulus_p for v in data["state_z"]]
        if len(s_mat) != inst.key_dim + 1 or any(len(r) != inst.value_dim for r in s_mat) or len(z_vec) != inst.key_dim + 1:
            raise ModelOrbitError("Corrupted AKS state tensor dimensions.")
        inst.prefix_length = int(data["prefix_length"])
        inst.head_digest = str(data["head_digest"])
        inst._state_s = s_mat
        inst._state_z = z_vec
        return inst


def direct_affine_kernel_oracle(
    modulus_p: int,
    keys: Sequence[Sequence[int]],
    values: Sequence[Sequence[int]],
    query: Sequence[int],
    *,
    normalized: bool = True,
) -> tuple[int, ...]:
    """Independent `O(T * d_k * d_v)` per-token replay oracle for `AffineKernelAttentionGround`."""
    if len(keys) != len(values):
        raise ModelOrbitError("keys and values length mismatch.")
    d_v = len(values[0]) if values else 0
    num = [0] * d_v
    mass = 0
    for k_t, v_t in zip(keys, values):
        w_t = (1 + sum((q_j % modulus_p) * (k_j % modulus_p) for q_j, k_j in zip(query, k_t))) % modulus_p
        mass = (mass + w_t) % modulus_p
        for j in range(d_v):
            num[j] = (num[j] + w_t * (v_t[j] % modulus_p)) % modulus_p
    if not normalized:
        return tuple(num)
    if mass == 0:
        raise ModelOrbitError("Zero kernel mass denominator in direct_affine_kernel_oracle.")
    inv_m = pow(mass, -1, modulus_p)
    return tuple((x * inv_m) % modulus_p for x in num)


def verify_softmax_insufficiency_witness() -> dict[str, Any]:
    r"""Constructive proof (`AKS-001`) that identical `(S_T, z_T)` cannot authorize exact softmax attention."""
    g_a = AffineKernelAttentionGround(modulus_p=97, key_dim=1, value_dim=1)
    g_b = AffineKernelAttentionGround(modulus_p=97, key_dim=1, value_dim=1)
    keys = ([0], [1], [2])
    vals_a = ([2], [2], [2])
    vals_b = ([3], [0], [3])
    for k, va, vb in zip(keys, vals_a, vals_b):
        g_a.admit_token(k, va)
        g_b.admit_token(k, vb)

    assert g_a._state_s == g_b._state_s and g_a._state_z == g_b._state_z
    # Yet under exponential softmax at q = 1, scores = [0, 1, 2]:
    weights = [math.exp(0.0), math.exp(1.0), math.exp(2.0)]
    denom = sum(weights)
    softmax_a = sum(w * va[0] for w, va in zip(weights, vals_a)) / denom
    softmax_b = sum(w * vb[0] for w, vb in zip(weights, vals_b)) / denom
    return {
        "distinct_lineage_heads": g_a.head_digest != g_b.head_digest,
        "identical_affine_sufficient_state": (g_a._state_s == g_b._state_s and g_a._state_z == g_b._state_z),
        "softmax_discrepancy": abs(softmax_a - softmax_b),
        "softmax_output_a": softmax_a,
        "softmax_output_b": softmax_b,
    }


__all__ = [
    "AffineKernelAttentionGround",
    "AffineKernelQueryReceipt",
    "AutoregressiveModelRunner",
    "GenerationResult",
    "LayerKVPage",
    "ModelOrbitCheckpoint",
    "ModelOrbitError",
    "PagedKVCacheGround",
    "direct_affine_kernel_oracle",
    "verify_softmax_insufficiency_witness",
]
