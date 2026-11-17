"""Phase 4 tests: Hierarchical lazy `NavigableComputationDAG` (>256 nodes), interval & bisimulation pre-gates,
crash-resumable `ReliableChannel`, mTLS PKI/replay protection, and `G11` terminal-to-terminal M1 forwarding.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path

import pytest

from vld.canonical import canonical_bytes, domain_digest
from vld.fabric import (
    ForwardingCoordinator,
    ForwardingWorkerNode,
    InformationFlowCodec,
    LossyInMemoryLink,
    MTLSPeerClient,
    MTLSPeerServer,
    MutualTLSError,
    ReliabilityPolicy,
    ReliableChannel,
    ReliableChannelConflictError,
    ReplayWindowGuard,
    UnauthorizedCommitError,
    VLDHttpVantageClient,
    VLDHttpVantageServer,
    VLDVantageNode,
    VantageAuthError,
    VantageEnvelope,
    WireEncoding,
    WorkerCompletionReceipt,
    create_mtls_client_context,
    create_mtls_server_context,
    generate_ephemeral_mtls_pki,
)
from vld.memory import InMemoryPageBacking, PagedObjectStore
from vld.navigation import (
    ComputationDAGError,
    ComputationNode,
    IntervalBound,
    NavigableComputationDAG,
    SubgraphThunk,
    analyze_dag_intervals,
    check_strong_bisimulation,
)
from vld.triad import IndeterminateGuardTrap, TruthDisposition


def test_hierarchical_lazy_dag_beyond_256_ceiling_guards_and_checkpoint() -> None:
    ctx = domain_digest("test.ctx", "phase4-dag")
    dag = NavigableComputationDAG("dag.moe-scale")

    root = ComputationNode("router", 1, "moe.route", "/layer/0", ctx, (), {"top_k": 2})
    dag.add_node(root)

    # Register a lazy SubgraphThunk that expands 320 nodes (> legacy 256 limit!) ONLY after router commits!
    def _expand_moe(committed: dict) -> list[ComputationNode]:
        chosen = committed["router"].value["experts"]
        nodes: list[ComputationNode] = []
        prev_ids = ["router"]
        for step in range(160):
            n_a = ComputationNode(f"exp_{chosen[0]}_{step}", 1, "linear", "/ffn", ctx, tuple(prev_ids), {"s": step})
            n_b = ComputationNode(f"exp_{chosen[1]}_{step}", 1, "linear", "/ffn", ctx, tuple(prev_ids), {"s": step})
            nodes.extend([n_a, n_b])
            prev_ids = [n_a.node_id, n_b.node_id]
        # Add conditional debug branch guarded by Kleene K_3
        nodes.append(
            ComputationNode("debug_branch", 1, "trace", "/dbg", ctx, (prev_ids[0],), {}, guard_key="enable_trace")
        )
        nodes.append(
            ComputationNode("debug_child", 1, "trace_sink", "/dbg", ctx, ("debug_branch",), {})
        )
        return nodes

    thunk = SubgraphThunk("thunk.experts", ("router",), domain_digest("test.thunk", "moe320"), _expand_moe)
    dag.add_subgraph_thunk(thunk)

    # Initially only 1 node is materialized!
    assert len(dag.nodes) == 1
    assert not dag.is_complete
    assert [n.node_id for n in dag.ready_frontier()] == ["router"]

    # Cycle self-dependency or cycle injection must be rejected
    with pytest.raises(ComputationDAGError):
        ComputationNode("bad", 1, "op", "/r", ctx, ("bad",), {})

    dag.commit_result("router", {"experts": [3, 17]})
    # Expanding ready frontier materializes all 322 child nodes (total 323 > 256)!
    frontier = dag.ready_frontier()
    assert len(dag.nodes) == 323
    assert len(frontier) == 2

    # Commit first 160 stages
    while True:
        # Before reaching debug_branch, guard_evaluator isn't triggered; at debug_branch test INDETERMINATE trap!
        try:
            ready = dag.ready_frontier(guard_evaluator=lambda _n, _c: TruthDisposition.INDETERMINATE)
        except IndeterminateGuardTrap:
            # Atomic trap: neither committed nor pruned!
            assert "debug_branch" not in dag.committed and "debug_branch" not in dag.pruned
            break
        for n in ready:
            dag.commit_result(n.node_id, {"ok": True})

    # Checkpoint mid-execution, tamper test, and restore!
    ckpt = dag.export_checkpoint()
    bad_ckpt = {**ckpt, "pruned": ["router"]}
    with pytest.raises(ComputationDAGError):
        NavigableComputationDAG.from_checkpoint(bad_ckpt)

    restored = NavigableComputationDAG.from_checkpoint(ckpt, thunks=[thunk])
    # Now evaluate guard as FALSE -> prunes both `debug_branch` and `debug_child` and completes DAG!
    final_ready = restored.ready_frontier(guard_evaluator=lambda _n, _c: TruthDisposition.FALSE)
    assert final_ready == ()
    assert "debug_branch" in restored.pruned and "debug_child" in restored.pruned
    assert restored.is_complete


def test_interval_abstract_interpretation_and_strong_bisimulation_pre_gates() -> None:
    ctx = domain_digest("test.ctx", "gates")
    dag = NavigableComputationDAG("dag.intervals")
    dag.add_node(ComputationNode("x", 1, "const", "/in", ctx, (), {"value": "0"}))
    dag.add_node(ComputationNode("aff", 1, "affine", "/op", ctx, ("x",), {"scale": "-2", "bias": "3"}))
    dag.add_node(ComputationNode("pos", 1, "relu", "/op", ctx, ("aff",), {}))
    dag.add_node(ComputationNode(" safe_div", 1, "const", "/in", ctx, (), {"value": "2"}) if False else ComputationNode("denom", 1, "const", "/in", ctx, (), {"value": "2"}))
    dag.add_node(ComputationNode("quot", 1, "div", "/op", ctx, ("pos", "denom"), {}))

    # For x in [-1, 4]: aff = -2x + 3 in [-5, 5], pos = ReLU(aff) in [0, 5], quot = pos / 2 in [0, 5/2]
    rep_safe = analyze_dag_intervals(
        dag,
        input_bounds={"x": IntervalBound(Fraction(-1), Fraction(4))},
        safety_contracts={"quot": IntervalBound(Fraction(0), Fraction(3))},
    )
    assert rep_safe.bounds["quot"] == IntervalBound(Fraction(0), Fraction(5, 2))
    assert rep_safe.overall_disposition is TruthDisposition.TRUE

    # Tighter contract [0, 2] straddles [0, 5/2] -> INDETERMINATE!
    rep_indet = analyze_dag_intervals(
        dag,
        input_bounds={"x": IntervalBound(Fraction(-1), Fraction(4))},
        safety_contracts={"quot": IntervalBound(Fraction(0), Fraction(2))},
    )
    assert rep_indet.overall_disposition is TruthDisposition.INDETERMINATE

    # Division straddling zero -> INDETERMINATE
    rep_zero = analyze_dag_intervals(
        dag,
        input_bounds={
            "x": IntervalBound(Fraction(0), Fraction(1)),
            "denom": IntervalBound(Fraction(-1), Fraction(1)),
        },
    )
    assert rep_zero.dispositions["quot"] is TruthDisposition.INDETERMINATE

    # Strong bisimulation: unminimized 4-state graph vs minimized 2-state quotient
    bisim_ok = check_strong_bisimulation(
        actions=("step", "reset"),
        left_states=("a0", "a1", "b0", "b1"),
        left_transitions={
            ("a0", "step"): ("b0",),
            ("a1", "step"): ("b1",),
            ("b0", "step"): ("a1",),
            ("b1", "step"): ("a0",),
            ("a0", "reset"): ("a0",),
            ("a1", "reset"): ("a0",),
            ("b0", "reset"): ("a1",),
            ("b1", "reset"): ("a1",),
        },
        left_observations={"a0": "LOW", "a1": "LOW", "b0": "HIGH", "b1": "HIGH"},
        left_initial="a0",
        right_states=("q_low", "q_high"),
        right_transitions={
            ("q_low", "step"): ("q_high",),
            ("q_high", "step"): ("q_low",),
            ("q_low", "reset"): ("q_low",),
            ("q_high", "reset"): ("q_low",),
        },
        right_observations={"q_low": "LOW", "q_high": "HIGH"},
        right_initial="q_low",
    )
    assert bisim_ok.equivalent is True
    assert bisim_ok.distinguishing_trace is None

    # Mutate one deep transition on right -> not bisimilar + concrete distinguishing trace!
    bisim_bad = check_strong_bisimulation(
        actions=("step", "reset"),
        left_states=("a0", "b0"),
        left_transitions={
            ("a0", "step"): ("b0",),
            ("b0", "step"): ("a0",),
            ("a0", "reset"): ("a0",),
            ("b0", "reset"): ("a0",),
        },
        left_observations={"a0": "LOW", "b0": "HIGH"},
        left_initial="a0",
        right_states=("q_low", "q_high"),
        right_transitions={
            ("q_low", "step"): ("q_high",),
            ("q_high", "step"): ("q_high",),  # diverges after ("step", "step")!
            ("q_low", "reset"): ("q_low",),
            ("q_high", "reset"): ("q_low",),
        },
        right_observations={"q_low": "LOW", "q_high": "HIGH"},
        right_initial="q_low",
    )
    assert bisim_bad.equivalent is False
    assert bisim_bad.distinguishing_trace == ("step", "step")


def test_crash_resumable_reliable_channel_and_vantage_replay_guard(tmp_path: Path) -> None:
    # 1. ReliableChannel across simulated packet loss + mid-stream sender/receiver crash-restart
    drop_first_seq2 = {"dropped": False}

    def _drop_pred(frame) -> bool:
        if frame.kind == "DATA" and frame.wire_seq == 2 and not drop_first_seq2["dropped"]:
            drop_first_seq2["dropped"] = True
            return True
        return False

    link = LossyInMemoryLink(drop_predicate=_drop_pred)
    pol = ReliabilityPolicy(window_size=8, lease_ticks=2, max_attempts=4)
    sender = ReliableChannel(channel_id="ch.1", local_peer="n1", remote_peer="n2", send_fn=link.send, policy=pol)
    receiver = ReliableChannel(channel_id="ch.1", local_peer="n2", remote_peer="n1", send_fn=link.send, policy=pol)

    committed_log: list[tuple[str, dict]] = []
    s1 = sender.send_operation("sem-op-alpha", {"val": 10})
    s2 = sender.send_operation("sem-op-beta", {"val": 20})
    s3 = sender.send_operation("sem-op-gamma", {"val": 30})
    assert (s1, s2, s3) == (1, 2, 3)

    # Drain initial frames at receiver (seq 2 was dropped on wire, so seq 1 commits and seq 3 buffers out-of-order!)
    while (f := link.pop_for("n2")) is not None:
        receiver.receive_frame(f, commit_fn=lambda op, p: committed_log.append((op, p)))
    while (f := link.pop_for("n1")) is not None:
        sender.receive_frame(f)

    assert [op for op, _ in committed_log] == ["sem-op-alpha"]
    assert sender.pending_count == 2

    # Crash and restore BOTH sender and receiver from exported checkpoints!
    sender_ckpt = sender.export_checkpoint()
    receiver_ckpt = receiver.export_checkpoint()
    sender_restored = ReliableChannel.from_checkpoint(sender_ckpt, send_fn=link.send)
    receiver_restored = ReliableChannel.from_checkpoint(receiver_ckpt, send_fn=link.send)

    # Advance ticks to trigger lease retransmission of missing wire_seq=2!
    sender_restored.tick_retransmit()
    retrans = sender_restored.tick_retransmit()
    assert 2 in retrans

    while (f := link.pop_for("n2")) is not None:
        receiver_restored.receive_frame(f, commit_fn=lambda op, p: committed_log.append((op, p)))
    while (f := link.pop_for("n1")) is not None:
        sender_restored.receive_frame(f)

    # Exact-once FIFO semantic commit across crash + packet drop!
    assert committed_log == [
        ("sem-op-alpha", {"val": 10}),
        ("sem-op-beta", {"val": 20}),
        ("sem-op-gamma", {"val": 30}),
    ]
    assert sender_restored.pending_count == 0

    # Conflicting payload on already-sent semantic op_id must raise!
    with pytest.raises(ReliableChannelConflictError):
        sender_restored.send_operation("sem-op-alpha", {"val": 999})

    # 2. HTTPS Vantage + ReplayWindowGuard rejecting Wire envelope replay!
    pki = generate_ephemeral_mtls_pki(tmp_path / "https_pki", ["127.0.0.1", "client.vld.local"])
    srv_ssl = create_mtls_server_context(pki["127.0.0.1"])
    cli_ssl = create_mtls_client_context(pki["client.vld.local"])
    secret = b"vld-https-capability-secret-0123"
    node = VLDVantageNode("https-node", capability_secret=secret, replay_guard=ReplayWindowGuard(32))

    with VLDHttpVantageServer(node, host="127.0.0.1", ssl_context=srv_ssl) as srv:
        assert srv.base_url.startswith("https://")
        cli = VLDHttpVantageClient(srv.base_url, client_id="client.vld.local", capability_secret=secret, ssl_context=cli_ssl)
        resp = cli.invoke_op(
            "affine_navigate",
            {"coordinate": 100, "increment": 3, "initial_state": 2, "modulus": 97, "multiplier": 5},
            request_id="https-req-1",
        )
        assert resp.flow_kind == "op_receipt"

        # Direct replay of sequence=0 on the node must be rejected by ReplayWindowGuard!
        replayed_env = VantageEnvelope(
            vantage_id="client.vld.local",
            flow_kind="op_request",
            sequence=0,
            payload={"args": {"coordinate": 100, "increment": 3, "initial_state": 2, "modulus": 97, "multiplier": 5}, "operation": "affine_navigate", "request_id": "https-req-1"},
        ).with_hmac(secret)
        with pytest.raises(VantageAuthError, match="Duplicate replay"):
            node.handle_envelope(replayed_env)


def test_mtls_cluster_multi_megabyte_terminal_to_terminal_forwarding_g11(tmp_path: Path) -> None:
    """Verify multi-MiB M1 forwarding $H_1 -> H_2$ across link drops with zero coordinator bulk bloat (`G11`)."""
    peers = ("coord.vld.local", "worker-a.vld.local", "worker-b.vld.local", "rogue.vld.local")
    pki = generate_ephemeral_mtls_pki(tmp_path / "cluster_pki", peers)
    other_pki = generate_ephemeral_mtls_pki(tmp_path / "untrusted_pki", ["intruder.vld.local"])

    # 2.2 MiB deterministic bulk payload (exceeds 64 KiB control ceiling by >35x!)
    bulk_size = 2_300_000
    pattern = bytes((i * 37 + 11) % 256 for i in range(4096))
    expected_bulk = (pattern * ((bulk_size // 4096) + 1))[:bulk_size]
    expected_sha = hashlib.sha256(expected_bulk).hexdigest()

    ctx_digest = domain_digest("test.ctx", "g11-forwarding")
    dag = NavigableComputationDAG("dag.g11")
    dag.add_node(ComputationNode("produce_tensor", 1, "tensor.synthesize", "/gpu/0", ctx_digest, (), {"bytes": bulk_size}))
    dag.add_node(ComputationNode("consume_reduce", 1, "tensor.checksum_xor", "/gpu/1", ctx_digest, ("produce_tensor",), {}))

    coord_sec = b"coord-master-secret-0123456789ab"
    w_secrets = {
        "worker-a.vld.local": b"worker-a-secret-key-0123456789ab",
        "worker-b.vld.local": b"worker-b-secret-key-0123456789ab",
    }
    coordinator = ForwardingCoordinator(dag, coordinator_secret=coord_sec, worker_secrets=w_secrets)

    allowed_mesh = frozenset({"coord.vld.local", "worker-a.vld.local", "worker-b.vld.local"})

    def _exec_a(node: ComputationNode, _up: dict[str, bytes]) -> tuple[dict, bytes | None]:
        return {"produced_bytes": len(expected_bulk)}, expected_bulk

    def _exec_b(node: ComputationNode, up: dict[str, bytes]) -> tuple[dict, bytes | None]:
        raw = up["produce_tensor"]
        return {"received_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}, None

    worker_a = ForwardingWorkerNode(
        worker_dns="worker-a.vld.local",
        worker_secret=w_secrets["worker-a.vld.local"],
        store=PagedObjectStore(InMemoryPageBacking(page_size=16384)),
        server_ssl_context=create_mtls_server_context(pki["worker-a.vld.local"]),
        client_ssl_context=create_mtls_client_context(pki["worker-a.vld.local"]),
        allowed_peers=allowed_mesh,
        executor_fn=_exec_a,
        drop_stream_after_bytes=600_000,  # Force mid-stream TLS drop during 2.3 MB transfer!
    )
    worker_b = ForwardingWorkerNode(
        worker_dns="worker-b.vld.local",
        worker_secret=w_secrets["worker-b.vld.local"],
        store=PagedObjectStore(InMemoryPageBacking(page_size=16384)),
        server_ssl_context=create_mtls_server_context(pki["worker-b.vld.local"]),
        client_ssl_context=create_mtls_client_context(pki["worker-b.vld.local"]),
        allowed_peers=allowed_mesh,
        executor_fn=_exec_b,
    )

    host_a, port_a = worker_a.start()
    host_b, port_b = worker_b.start()
    try:
        # 1. Verify untrusted CA and unauthorized SAN ("rogue.vld.local") are rejected!
        bad_ca_cli = MTLSPeerClient(
            host=host_a,
            port=port_a,
            expected_server_dns="worker-a.vld.local",
            ssl_context=create_mtls_client_context(other_pki["intruder.vld.local"]),
        )
        with pytest.raises(MutualTLSError):
            bad_ca_cli.ping()

        rogue_cli = MTLSPeerClient(
            host=host_a,
            port=port_a,
            expected_server_dns="worker-a.vld.local",
            ssl_context=create_mtls_client_context(pki["rogue.vld.local"]),
        )
        with pytest.raises(MutualTLSError, match="peer_san_unauthorized"):
            rogue_cli.ping()

        # 2. Coordinator dispatches `produce_tensor` to Worker A and `consume_reduce` to Worker B
        coord_ssl = create_mtls_client_context(pki["coord.vld.local"])
        cli_to_a = MTLSPeerClient(host=host_a, port=port_a, expected_server_dns="worker-a.vld.local", ssl_context=coord_ssl)
        cli_to_b = MTLSPeerClient(host=host_b, port=port_b, expected_server_dns="worker-b.vld.local", ssl_context=coord_ssl)

        assign_a = coordinator.issue_assignment("produce_tensor", "worker-a.vld.local")
        res_a, rcpt_a = coordinator.dispatch_and_commit(assign_a, cli_to_a)
        assert rcpt_a.output_ticket is not None and rcpt_a.output_ticket.size_bytes == bulk_size

        assign_b = coordinator.issue_assignment("consume_reduce", "worker-b.vld.local")
        # Forged worker commit receipt before execution must be refused by Origin Coordinator!
        forged_receipt = WorkerCompletionReceipt(
            dag_id=dag.dag_id,
            node_id="consume_reduce",
            node_digest=assign_b.node.digest,
            worker_dns="worker-b.vld.local",
            assignment_mac="0" * 64,
            summary_value={"hacked": True},
            output_ticket=None,
            direct_reconnects=0,
            worker_signature_mac="0" * 64,
        )
        with pytest.raises(UnauthorizedCommitError):
            coordinator.verify_and_commit_receipt(forged_receipt)

        res_b, rcpt_b = coordinator.dispatch_and_commit(assign_b, cli_to_b)
        assert dag.is_complete
        # Link dropped at 600 KB -> Worker B automatically reconnected and resumed exact byte range!
        assert rcpt_b.direct_reconnects == 1
        assert res_b.value == {"received_bytes": bulk_size, "sha256": expected_sha}
        # Coordinator control plane carried < 8 KiB total while 2.3 MB flowed directly Worker A -> Worker B!
        assert coordinator.coordinator_control_bytes < 8_192
    finally:
        worker_a.close()
        worker_b.close()


def test_galois_sound_extended_intervals_milner_bisimulation_and_crash_resumable_mtls_pages(tmp_path: Path) -> None:
    """Verify extended-interval soundness, Milner nondeterministic bisimulation, guard bypass refusal,
    >64 KiB 1,024-page M1 manifest streaming, subprocess os._exit page-flush resume (`G14`), and worker auth.
    """
    import subprocess
    import sys
    from vld.fabric import M1ForwardingTicket
    from vld.memory import DiskPageBacking

    ctx = domain_digest("test.ctx", "hardening-phase4")
    # 1. commit_result MUST refuse guarded node before guard_key evaluates TRUE!
    g_dag = NavigableComputationDAG("dag.guarded")
    g_dag.add_node(ComputationNode("gated", 1, "op", "/r", ctx, (), {}, guard_key="must_pass"))
    with pytest.raises(IndeterminateGuardTrap):
        g_dag.commit_result("gated", {"bad": True})

    # 2. Galois soundness on zero-straddling division (10^20 / [-1, 1]) & rational mod_residue (79/10 mod 4 = 3.9)!
    i_dag = NavigableComputationDAG("dag.soundness")
    i_dag.add_node(ComputationNode("big", 1, "const", "/r", ctx, (), {"value": "100000000000000000000"}))
    i_dag.add_node(ComputationNode("den", 1, "const", "/r", ctx, (), {"value": "1"}))
    i_dag.add_node(ComputationNode("quot", 1, "div", "/r", ctx, ("big", "den"), {}))
    i_dag.add_node(ComputationNode("rem", 1, "mod_residue", "/r", ctx, ("den",), {"modulus": 4}))

    rep_div = analyze_dag_intervals(
        i_dag,
        input_bounds={"den": IntervalBound(Fraction(-1), Fraction(1))},
        safety_contracts={"quot": IntervalBound(Fraction(-10**19), Fraction(10**19))},
    )
    assert rep_div.bounds["quot"] == IntervalBound.unbounded()
    assert rep_div.dispositions["quot"] is TruthDisposition.INDETERMINATE

    rep_mod = analyze_dag_intervals(
        i_dag,
        input_bounds={"den": IntervalBound(Fraction(79, 10), Fraction(81, 10))},
    )
    assert rep_mod.bounds["rem"].lo <= Fraction(39, 10) <= rep_mod.bounds["rem"].hi

    # 3. Classic Milner nondeterministic branching counterexample: a.(b.1 + b.2) vs a.b.1 + a.b.2
    milner = check_strong_bisimulation(
        actions=("a", "b"),
        left_states=("l0", "l1", "l2", "l3", "l4"),
        left_transitions={
            ("l0", "a"): ("l1", "l2"),
            ("l1", "b"): ("l3",),
            ("l2", "b"): ("l4",),
        },
        left_observations={"l0": 0, "l1": 0, "l2": 0, "l3": 1, "l4": 2},
        left_initial="l0",
        right_states=("r0", "r1", "r2", "r3"),
        right_transitions={
            ("r0", "a"): ("r1",),
            ("r1", "b"): ("r2", "r3"),
        },
        right_observations={"r0": 0, "r1": 0, "r2": 1, "r3": 2},
        right_initial="r0",
    )
    assert milner.equivalent is False
    assert milner.distinguishing_trace == ("a", "b")

    # 4. 4 MiB M1 object with 1,024 DISTINCT 4 KiB pages (>117 KB manifest!) + subprocess os._exit mid-stream resume!
    pki = generate_ephemeral_mtls_pki(tmp_path / "pki_g14", ["srv.vld.local", "cli.vld.local", "peer.vld.local"])
    src_store = PagedObjectStore(InMemoryPageBacking(page_size=4096))
    payload_4mb = b"".join(
        p_idx.to_bytes(4, "little") + bytes(((p_idx + i) * 73 + 19) % 256 for i in range(4092))
        for p_idx in range(1024)
    )
    m_4mb = src_store.put_bytes(payload_4mb)

    disk_dir = tmp_path / "client_m1_disk"
    with MTLSPeerServer(
        peer_id="srv.vld.local",
        ssl_context=create_mtls_server_context(pki["srv.vld.local"]),
        allowed_peer_dns=frozenset({"cli.vld.local", "peer.vld.local"}),
        store=src_store,
    ) as srv:
        host, port = srv.endpoint
        # Subprocess 1 fetches 300 pages onto DiskPageBacking then hard-kills via os._exit(94)!
        kill_cli_script = f"""
import os
from pathlib import Path
from vld.fabric import MTLSPeerClient, PeerIdentityBundle, create_mtls_client_context
from vld.memory import DiskPageBacking, PagedObjectStore

b = PeerIdentityBundle("cli.vld.local", Path({str(pki['cli.vld.local'].ca_cert_path)!r}), Path({str(pki['cli.vld.local'].cert_path)!r}), Path({str(pki['cli.vld.local'].key_path)!r}))
cli = MTLSPeerClient(host={host!r}, port={port}, expected_server_dns="srv.vld.local", ssl_context=create_mtls_client_context(b))
tgt = PagedObjectStore(DiskPageBacking(Path({str(disk_dir)!r}), page_size=4096))
cli.fetch_m1_object_resumable(
    {m_4mb.manifest_digest!r},
    tgt,
    _fault_hook_after_pages=lambda n: os._exit(94) if n >= 300 else None,
)
"""
        p_kill = subprocess.run([sys.executable, "-c", kill_cli_script], capture_output=True, text=True)
        assert p_kill.returncode == 94, p_kill.stderr
        bytes_before_resume = srv.bulk_bytes_streamed

        # Fresh client resumes from DiskPageBacking: streams ONLY the remaining 724 pages (not all 1,024)!
        cli_resume = MTLSPeerClient(
            host=host,
            port=port,
            expected_server_dns="srv.vld.local",
            ssl_context=create_mtls_client_context(pki["cli.vld.local"]),
        )
        tgt_resume = PagedObjectStore(DiskPageBacking(disk_dir, page_size=4096))
        got_m, recon = cli_resume.fetch_m1_object_resumable(m_4mb.manifest_digest, tgt_resume)
        resumed_delta_bytes = srv.bulk_bytes_streamed - bytes_before_resume
        assert recon == 0
        assert resumed_delta_bytes == (1024 - 300) * 4096
        assert tgt_resume.read_bytes(got_m) == payload_4mb

    # 5. ForwardingWorkerNode rejects non-coordinator peer execute_assignment & coordinator rejects spoofed output_ticket.node_id!
    sec_c = b"coord-secret-key-0123456789abcde"
    sec_w = b"worker-secret-key-0123456789abcd"
    f_dag = NavigableComputationDAG("dag.auth")
    f_dag.add_node(ComputationNode("n1", 1, "op", "/r", ctx, (), {}))
    f_coord = ForwardingCoordinator(f_dag, coordinator_secret=sec_c, worker_secrets={"srv.vld.local": sec_w})
    assign = f_coord.issue_assignment("n1", "srv.vld.local")

    w_node = ForwardingWorkerNode(
        worker_dns="srv.vld.local",
        worker_secret=sec_w,
        store=PagedObjectStore(InMemoryPageBacking(page_size=4096)),
        server_ssl_context=create_mtls_server_context(pki["srv.vld.local"]),
        client_ssl_context=create_mtls_client_context(pki["srv.vld.local"]),
        allowed_peers=frozenset({"cli.vld.local", "peer.vld.local"}),
        allowed_coordinators=frozenset({"cli.vld.local"}),
        coordinator_secret=sec_c,
        executor_fn=lambda _n, _u: ({"ok": 1}, None),
    )
    wh, wp = w_node.start()
    try:
        peer_cli = MTLSPeerClient(
            host=wh,
            port=wp,
            expected_server_dns="srv.vld.local",
            ssl_context=create_mtls_client_context(pki["peer.vld.local"]),
        )
        with pytest.raises(MutualTLSError, match="not an authorized coordinator"):
            peer_cli.invoke_control({"assignment": assign.to_dict(), "op": "execute_assignment"})
    finally:
        w_node.close()

    bad_ticket = M1ForwardingTicket("WRONG_NODE", m_4mb.manifest_digest, m_4mb.content_digest, m_4mb.size_bytes, "srv.vld.local", "127.0.0.1", 9999)
    bad_sig = WorkerCompletionReceipt.compute_worker_mac(
        sec_w,
        dag_id="dag.auth",
        node_id="n1",
        node_digest=assign.node.digest,
        worker_dns="srv.vld.local",
        assignment_mac=assign.assignment_mac,
        summary_value={"ok": 1},
        output_manifest_digest=bad_ticket.manifest_digest,
    )
    bad_rcpt = WorkerCompletionReceipt("dag.auth", "n1", assign.node.digest, "srv.vld.local", assign.assignment_mac, {"ok": 1}, bad_ticket, 0, bad_sig)
    with pytest.raises(UnauthorizedCommitError, match="node_id"):
        f_coord.verify_and_commit_receipt(bad_rcpt)

