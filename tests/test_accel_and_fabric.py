"""Wave 3 tests: Multi-tier numeric/native/structural acceleration (`vld.accel`) and HTTP/stream vantage hosting (`vld.fabric`)."""

from __future__ import annotations

import io
import math

import pytest

from vld.accel import (
    LinearRecurrenceGround,
    NumericEngine,
    StructuralLinearOperator,
    benchmark_virtual_vs_traditional,
    decode_bf16_le,
    decode_f32_le,
    encode_bf16_le,
    encode_f32_le,
    get_native_accelerator,
    has_numpy,
)
from vld.fabric import (
    InformationFlowCodec,
    StreamPipeVantage,
    VLDHttpVantageClient,
    VLDHttpVantageServer,
    VLDVantageNode,
    VantageAuthError,
    VantageEnvelope,
    VantageError,
    WireEncoding,
)
from vld.ground import AffineOrbitLaw, VirtualGroundSpec
from vld.journal import IdempotencyConflictError
from vld.memory import InMemoryPageBacking, PagedObjectStore


def test_numeric_engine_stdlib_numpy_and_native_rust_parity() -> None:
    eng_std = NumericEngine(mode="stdlib")
    vals = [-3.5, -0.125, 0.0, 0.75, 12.25]
    raw_f32 = encode_f32_le(vals)
    assert decode_f32_le(raw_f32, use_numpy=False) == pytest.approx(vals, abs=1e-6)
    raw_bf16 = encode_bf16_le(vals)
    assert decode_bf16_le(raw_bf16, use_numpy=False) == pytest.approx(vals, rel=1e-2)

    with pytest.raises(ValueError):
        encode_f32_le([1.0, float("nan")])

    x = [[0.5, -1.25, 2.0, 0.75], [-0.25, 1.5, -0.5, 3.0]]
    w_norm = [1.1, 0.9, 1.2, 0.8]
    w_proj = [[0.2, -0.3, 0.5, 0.1], [-0.4, 0.6, -0.2, 0.8], [0.7, 0.1, -0.5, 0.3]]

    norm_std = eng_std.rms_norm_2d(x, w_norm, 1e-6)
    lin_std = eng_std.linear_x_wt(norm_std, w_proj)
    swi_std = eng_std.swiglu_2d(lin_std, lin_std)

    if has_numpy():
        eng_np = NumericEngine(mode="numpy")
        assert decode_bf16_le(raw_bf16, use_numpy=True) == decode_bf16_le(raw_bf16, use_numpy=False)
        norm_np = eng_np.rms_norm_2d(x, w_norm, 1e-6)
        lin_np = eng_np.linear_x_wt(norm_np, w_proj)
        swi_np = eng_np.swiglu_2d(lin_np, lin_np)
        assert NumericEngine.max_abs_diff(norm_std, norm_np) < 1e-6
        assert NumericEngine.max_abs_diff(lin_std, lin_np) < 1e-6
        assert NumericEngine.max_abs_diff(swi_std, swi_np) < 1e-5

    native = get_native_accelerator()
    if native.available:
        # Verify native Rust u128 LCG jump against dual-derivation Python AffineOrbitLaw
        law = AffineOrbitLaw(1664525, 1013904223, 2**32)
        assert native.lcg_jump_u64(1664525, 1013904223, 2**32, 73, 10**12) == law.navigate(73, 10**12).value
        # Verify native Rust FP32 RMSNorm against pure-stdlib
        norm_rust = native.rms_norm_f32(x, w_norm, 1e-6)
        assert NumericEngine.max_abs_diff(norm_std, norm_rust) < 1e-6


def test_structural_operator_contraction_and_pareto_benchmark() -> None:
    op = StructuralLinearOperator(
        rows=96,
        cols=128,
        alpha=0.3,
        beta=-0.01,
        gamma=0.005,
        delta=0.0002,
        periodic_orbit=(0.5, -0.25, 0.125, -0.75, 0.2),
        orbit_stride=2,
        low_rank_u=tuple((math.sin(0.1 * i), math.cos(0.1 * i)) for i in range(96)),
        low_rank_v=tuple((math.cos(0.07 * j), -math.sin(0.07 * j)) for j in range(128)),
        sparse_entries=((3, 7, 4.5), (50, 100, -2.25)),
    )
    x = [math.sin(0.13 * j) + 0.2 for j in range(128)]
    dense = op.materialize_dense()
    y_dense = [math.fsum(r * xv for r, xv in zip(row, x)) for row in dense]
    y_py = op.contract(x, allow_native=False)
    y_nat = op.contract(x, allow_native=True)

    assert NumericEngine.max_abs_diff(y_dense, y_py) < 1e-11
    assert NumericEngine.max_abs_diff(y_dense, y_nat) < 1e-11
    assert op.structural_flops < op.dense_flops
    assert op.structural_bytes_f64 < op.dense_bytes_f64

    # Unstable matrix exponentiation must be refused!
    with pytest.raises(ValueError, match="unstable"):
        LinearRecurrenceGround(((2.0, 0.0), (0.0, 2.0)), (0.0, 0.0))

    bench = benchmark_virtual_vs_traditional(matrix_dim=512, recurrence_steps=5_000)
    assert bench["structural_operator_contraction"]["pareto_dominates_dense_flops_and_bytes"] is True
    assert bench["structural_operator_contraction"]["vld_pareto_dominates_cost_vector"] is True
    assert bench["linear_recurrence_leap"]["pareto_dominates_traditional"] is True


def test_information_flow_codec_and_stream_pipe_vantage() -> None:
    secret = b"vld-super-secret-capability-key!"
    env = VantageEnvelope(
        vantage_id="edge-sensor-01",
        flow_kind="op_request",
        sequence=7,
        payload={"args": {"coordinate": 500, "increment": 3, "initial_state": 1, "modulus": 97, "multiplier": 5}, "operation": "affine_navigate", "request_id": "pipe-req-1"},
    ).with_hmac(secret)

    for enc in WireEncoding:
        wire = InformationFlowCodec.encode(env, enc)
        decoded = InformationFlowCodec.decode(wire, enc)
        assert decoded == env
        assert decoded.verify_hmac(secret)

    node = VLDVantageNode("pipe-host", capability_secret=secret)
    pipe = StreamPipeVantage(node)
    in_buf = io.BytesIO(InformationFlowCodec.encode(env, WireEncoding.LENGTH_PREFIXED_BINARY))
    out_buf = io.BytesIO()
    resp = pipe.process_one_frame(in_buf, out_buf)
    assert resp.flow_kind == "op_receipt"
    out_buf.seek(0)
    wire_resp = InformationFlowCodec.read_length_prefixed(out_buf)
    assert wire_resp == resp
    assert wire_resp.payload["result"]["value"] == AffineOrbitLaw(5, 3, 97).navigate(1, 500).value


def test_live_http_vantage_server_auth_idempotency_and_m1_range_streaming() -> None:
    secret = b"0123456789abcdef0123456789abcdef"
    backing = InMemoryPageBacking(page_size=64)
    store = PagedObjectStore(backing)
    node = VLDVantageNode("http-vantage-01", store=store, capability_secret=secret)
    ground_spec = VirtualGroundSpec(
        ground_id="vld.http.test",
        version="0.1.0",
        language="vld.affine.v1",
        state_domain="Z/2^32Z",
        transitions=("navigate",),
        invariants=("0 <= x < 2^32",),
        observers=("residue",),
    )
    node.register_ground(ground_spec)

    # Publish a 640-byte (10-page) M1 object directly in node store
    blob = bytes((i * 19 + 5) % 256 for i in range(640))
    manifest = store.put_bytes(blob, media_type="application/octet-stream")

    with VLDHttpVantageServer(node) as server:
        unauth_client = VLDHttpVantageClient(server.base_url, client_id="intruder")
        auth_client = VLDHttpVantageClient(server.base_url, client_id="trusted-peer", capability_secret=secret)

        health = auth_client.health()
        assert health["status"] == "READY"
        assert "vld.http.test" in health["hosted_grounds"]

        # 1. Unauthenticated mutation/op must be rejected with 403 VantageAuthError
        with pytest.raises(VantageAuthError):
            unauth_client.invoke_op(
                "affine_navigate",
                {"coordinate": 10, "increment": 1, "initial_state": 0, "modulus": 16, "multiplier": 1},
                request_id="unauth-1",
            )

        # 2. Admit a Structural Memory law & query distant coordinate over HTTP
        admit_resp = auth_client.invoke_op(
            "admit_structural_law",
            {"declared_modulus": 97, "domain": "Z/97Z", "memory_id": "remote.prog", "samples": [90, 95, 3, 8, 13]},
            request_id="admit-prog-1",
        )
        assert admit_resp.payload["replayed"] is False
        assert admit_resp.payload["result"]["family"] == "int"

        # Exact retry returns idempotent cached receipt without re-running!
        replay_resp = auth_client.invoke_op(
            "admit_structural_law",
            {"declared_modulus": 97, "domain": "Z/97Z", "memory_id": "remote.prog", "samples": [90, 95, 3, 8, 13]},
            request_id="admit-prog-1",
        )
        assert replay_resp.payload["replayed"] is True
        assert replay_resp.payload["journal_event_digest"] == admit_resp.payload["journal_event_digest"]

        # Conflicting retry with same request_id raises IdempotencyConflictError (HTTP 409)
        with pytest.raises(IdempotencyConflictError):
            auth_client.invoke_op(
                "admit_structural_law",
                {"declared_modulus": 97, "domain": "Z/97Z", "memory_id": "DIFFERENT", "samples": [90, 95, 3, 8, 13]},
                request_id="admit-prog-1",
            )

        read_resp = auth_client.invoke_op(
            "structural_read",
            {"coordinate_start": 10**12, "length": 3, "memory_id": "remote.prog"},
            request_id="read-trillion",
        )
        assert read_resp.payload["result"]["expanded_intermediate_states"] == 0
        assert read_resp.payload["result"]["values"] == [(90 + 5 * (10**12 + k)) % 97 for k in range(3)]

        # 3. HTTP 206 Range slice read touches ONLY the 2 intersecting M1 pages out of 10!
        fetched_manifest = auth_client.fetch_manifest(manifest.manifest_digest)
        assert fetched_manifest == manifest
        backing.pages_read = 0
        remote_slice = auth_client.fetch_m1_range(manifest.manifest_digest, 60, 40)
        assert remote_slice == blob[60:100]
        assert backing.pages_read == 2  # bytes 60..99 span page 0 (0..63) and page 1 (64..127) ONLY!

        # Out-of-range HTTP Range request raises VantageError (HTTP 416)
        with pytest.raises(VantageError):
            auth_client.fetch_m1_range(manifest.manifest_digest, 600, 100)
