"""Comprehensive tests for `HeadlessOS`, IPv4/Internet Path Grounds, Observer-Orbit/Commutant Blocks, mTLS Tensor-Cut Fabric, and Multi-Process Continuous Virtual System Recovery."""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from vld.accel import NumericEngine, encode_f32_le
from vld.authority import (
    CrossSchedulerComparisonError,
    QuorumAuthorityMember,
    QuorumLineageCoordinator,
    QuorumMembership,
    QuorumProposal,
    StaleTermError,
    TemporalCoordinate,
    VectorClock,
)
from vld.canonical import domain_digest
from vld.cli import main as cli_main
from vld.continuous import (
    ContinuousStageSpec,
    ContinuousSystemError,
    ContinuousVirtualSystem,
    run_self_improving_continuous_campaign,
)
from vld.fabric import (
    ComputationalFabricRouter,
    FabricBlockerCode,
    FabricLink,
    FabricResolutionStatus,
    FabricRoutingError,
    HostAdvertisement,
    PartitionedTensorCutCoordinator,
    RouteMetric,
    StaleClaimEpochError,
    TensorCutWorkerNode,
    create_mtls_client_context,
    create_mtls_server_context,
    generate_ephemeral_mtls_pki,
)
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.journal import HashChainedJournal
from vld.memory import DiskPageBacking, InMemoryPageBacking, PagedObjectStore, StructuralMemoryRegistry
from vld.navigation import (
    AttributedObserverReading,
    ComposedComponentSpec,
    ComposedGroundNavigator,
    ContradictoryObservationError,
    CrossObserverCongruenceGate,
    FiniteFieldOperatorChain,
    ObserverOrbitError,
    discover_similarity_blocks,
)
from vld.navigation.observer_orbit import _invert_matrix_fp, _mat_mul_fp, _mat_pow_fp, _mat_vec_fp
from vld.signal import Signal8Backend
from vld.substrates import (
    Errno,
    FD_CLOEXEC,
    F_GETFD,
    F_SETFD,
    HeadlessOS,
    InternetNavigationPolicy,
    InternetObservationCase,
    InternetObservationRouter,
    InternetPathRefused,
    MAP_ANONYMOUS,
    MAP_PRIVATE,
    O_CREAT,
    O_EXCL,
    O_RDONLY,
    O_RDWR,
    O_WRONLY,
    PF_R,
    PF_X,
    PROT_EXEC,
    PROT_READ,
    PROT_WRITE,
    SEEK_END,
    SEEK_SET,
    TTL_PROBES,
    VMInstruction,
    WNOHANG,
    build_ethernet_ipv4_udp_frame,
    build_minimal_elf64_image,
    canonical_rfc_url,
    navigate_bounded_hypertext_ground,
    verify_internet_navigation_capsule,
    verify_ipv4_router_hop_ground,
)


def test_headless_os_posix_syscalls_cow_execve_rollback_and_m1_snapshot() -> None:
    sm = StructuralMemoryRegistry()
    sm.certify_and_admit("petabyte.law", [(37 + 19 * k) % 251 for k in range(8)], domain="Z/251Z", declared_modulus=251)
    hos = HeadlessOS(page_size=256, structural_memory=sm)
    hos.vfs.register_structural_file("/proc/petabyte.bin", memory_id="petabyte.law", virtual_size=10**15)

    p1 = hos.spawn_process("init", (VMInstruction("HALT", ()),))
    assert hos.posix_invoke(p1, 39).result == p1  # getpid
    assert hos.posix_invoke(p1, 999).errno == int(Errno.ENOSYS)

    # Read from offset 10^12 in petabyte StructuralMemory virtual file without allocating backing bytes!
    fd_s = hos.posix_invoke(p1, 2, path="/proc/petabyte.bin", flags=O_RDONLY).result
    assert hos.posix_invoke(p1, 2, path="/proc/petabyte.bin", flags=O_WRONLY).errno == int(Errno.EACCES)
    assert hos.posix_invoke(p1, 8, fd=fd_s, offset=10**12, whence=SEEK_SET).result == 10**12
    assert hos.posix_invoke(p1, 0, fd=fd_s, address=32, size=6).result == 6
    got_s = list(hos._procs[p1].vm_proc.memory.read_bytes(32, 6))
    assert got_s == [(37 + 19 * (10**12 + i)) % 251 for i in range(6)]
    hos.posix_invoke(p1, 5, fd=fd_s, stat_address=64)  # fstat
    _ino, _mode, sz_stat = struct.unpack("<QQQ", hos._procs[p1].vm_proc.memory.read_bytes(64, 24))
    assert sz_stat == 10**15

    # Regular file O_CREAT | O_EXCL, write, lseek, dup, dup2, fcntl FD_CLOEXEC
    fd_w = hos.posix_invoke(p1, 2, path="/tmp/hello.txt", flags=O_RDWR | O_CREAT | O_EXCL).result
    assert hos.posix_invoke(p1, 2, path="/tmp/hello.txt", flags=O_RDWR | O_CREAT | O_EXCL).errno == int(Errno.EEXIST)
    hos._procs[p1].vm_proc.memory.write_bytes(128, b"VLD-OS")
    assert hos.posix_invoke(p1, 1, fd=fd_w, address=128, size=6).result == 6
    assert hos.posix_invoke(p1, 8, fd=fd_w, offset=0, whence=SEEK_END).result == 6
    fd_dup = hos.posix_invoke(p1, 32, fd=fd_w).result
    fd_dup2 = hos.posix_invoke(p1, 33, fd=fd_w, new_fd=12).result
    assert fd_dup2 == 12
    assert hos.posix_invoke(p1, 72, fd=fd_dup, command=F_SETFD, arg=FD_CLOEXEC).ok
    assert hos.posix_invoke(p1, 72, fd=fd_dup, command=F_GETFD).result == FD_CLOEXEC

    # mmap + W^X refusal in mmap and mprotect + munmap
    assert hos.posix_invoke(p1, 9, length=256, prot=PROT_WRITE | PROT_EXEC, flags=MAP_PRIVATE | MAP_ANONYMOUS).errno == int(
        Errno.EACCES
    )
    m_addr = hos.posix_invoke(p1, 9, length=256, prot=PROT_READ | PROT_WRITE, flags=MAP_PRIVATE | MAP_ANONYMOUS).result
    assert m_addr >= HeadlessOS.MMAP_BASE_START
    assert hos.posix_invoke(p1, 10, address=m_addr, length=256, prot=PROT_WRITE | PROT_EXEC).errno == int(Errno.EACCES)
    assert hos.posix_invoke(p1, 11, address=m_addr, length=256).ok

    # Pipe IPC + CoW fork + wait4 + EOF on writer close
    assert hos.posix_invoke(p1, 22, address=160).ok
    pr, pw = struct.unpack("<ii", hos._procs[p1].vm_proc.memory.read_bytes(160, 8))
    assert hos.posix_invoke(p1, 0, fd=pr, address=192, size=4).errno == int(Errno.EAGAIN)

    c_pid = hos.posix_invoke(p1, 57).result  # fork CoW
    # Child mutates memory at 128; parent memory at 128 must stay intact!
    hos._procs[c_pid].vm_proc.memory.write_bytes(128, b"CHILD!")
    assert hos._procs[p1].vm_proc.memory.read_bytes(128, 6) == b"VLD-OS"

    # Snapshot entire OS to M1 and verify CoW page deduplication + exact restore (G28)
    os_store = PagedObjectStore(InMemoryPageBacking(256))
    snap = hos.snapshot_os_to_m1(os_store)
    restored = HeadlessOS.restore_os_from_m1(snap, os_store, structural_memory=sm)
    assert restored._procs[p1].vm_proc.memory.read_bytes(128, 6) == b"VLD-OS"
    assert restored._procs[c_pid].vm_proc.memory.read_bytes(128, 6) == b"CHILD!"

    # Transactional execve on child: bad ELF64 must roll back cleanly without corrupting child memory or closing FDs!
    hos.install_elf_binary("/bin/corrupt.elf", b"\x7fELF\x00\x00\x00corrupt")
    bad_exec = hos.posix_invoke(c_pid, 59, path="/bin/corrupt.elf")
    assert bad_exec.errno == int(Errno.EINVAL)
    assert hos._procs[c_pid].vm_proc.memory.read_bytes(128, 6) == b"CHILD!"
    assert fd_dup in hos._procs[c_pid].fds  # FD_CLOEXEC NOT closed on failed execve!

    # Valid ELF64 execve closes FD_CLOEXEC descriptors and runs to completion
    good_elf = build_minimal_elf64_image([(0x401000, PF_R | PF_X, b"\x90\xc3", 16)])
    hos.install_elf_binary("/bin/good.elf", good_elf)
    assert hos.posix_invoke(c_pid, 59, path="/bin/good.elf").ok
    assert fd_dup not in hos._procs[c_pid].fds
    assert hos.posix_invoke(p1, 61, child_pid=c_pid, options=WNOHANG).result == 0  # still running
    hos.posix_invoke(c_pid, 60, code=42)  # exit(42)
    assert hos.posix_invoke(p1, 61, child_pid=c_pid, status_address=200).result == c_pid
    wstatus = struct.unpack("<I", hos._procs[p1].vm_proc.memory.read_bytes(200, 4))[0]
    assert (wstatus >> 8) == 42

    # Close parent's write end of pipe -> read returns 0 (EOF)
    hos.posix_invoke(p1, 3, fd=pw)
    assert hos.posix_invoke(p1, 0, fd=pr, address=192, size=4).result == 0


def test_ipv4_router_hop_hypertext_capsule_and_observation_router() -> None:
    sf = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t) for i, t in enumerate(TTL_PROBES)]
    r1 = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t - 1) for i, t in enumerate(TTL_PROBES)]
    r2 = [build_ethernet_ipv4_udp_frame(sequence=i, ttl=t - 2) for i, t in enumerate(TTL_PROBES)]

    rep2 = verify_ipv4_router_hop_ground(
        "two",
        sender_frames=sf,
        hop_captures=[("two.r1", sf, r1), ("two.r2", r1, r2)],
        receiver_ttls=[t - 2 for t in TTL_PROBES],
        require_in_path_transform=True,
    )
    assert rep2.l3_hops == 2 and rep2.performs_in_path_ttl_transform is True
    assert rep2.certified_ttl_law is not None and rep2.certified_ttl_law.navigate(64, 10).value == 44

    # L2 bridge negative control: passes when transform not required, deterministically refused when required!
    rep_br = verify_ipv4_router_hop_ground(
        "bridge",
        sender_frames=sf,
        hop_captures=[("bridge.r1", sf, sf)],
        receiver_ttls=list(TTL_PROBES),
        require_in_path_transform=False,
    )
    assert rep_br.performs_in_path_ttl_transform is False
    with pytest.raises(InternetPathRefused):
        verify_ipv4_router_hop_ground(
            "bridge",
            sender_frames=sf,
            hop_captures=[("bridge.r1", sf, sf)],
            receiver_ttls=list(TTL_PROBES),
            require_in_path_transform=True,
        )

    # Corrupted RFC 1071 IPv4 header checksum must be rejected
    bad_r1 = list(r1)
    corrupted = bytearray(bad_r1[0])
    corrupted[24] ^= 0xFF
    bad_r1[0] = bytes(corrupted)
    with pytest.raises(InternetPathRefused, match="checksum"):
        verify_ipv4_router_hop_ground(
            "one",
            sender_frames=sf,
            hop_captures=[("one.r1", sf, bad_r1)],
            receiver_ttls=[t - 1 for t in TTL_PROBES],
        )

    # Bounded RFC hypertext Ground (INTERNET-004)
    pages = {
        canonical_rfc_url(9110): b'<html><head><title>HTTP Semantics</title></head><body><a href="/rfc/rfc9112.html">HTTP/1.1</a><a href="https://evil.example/x">ignore</a></body></html>',
        canonical_rfc_url(9112): b'<html><head><title>HTTP/1.1</title></head><body><a href="/rfc/rfc9114.html">HTTP/3</a></body></html>',
    }
    pol = InternetNavigationPolicy(seeds=(9110,), target=9114, max_pages=4, max_depth=2)
    cap = navigate_bounded_hypertext_ground(pol, lambda u: (200, "text/html; charset=utf-8", pages[u]))
    assert cap["status"] == "reached" and len(cap["path"]) == 3
    ver = verify_internet_navigation_capsule(cap, pol)
    assert ver["status"] == "reached" and ver["verified_pages"] == 2

    # Redirect refusal & tamper refusal
    with pytest.raises(InternetPathRefused, match="redirect"):
        navigate_bounded_hypertext_ground(pol, lambda _u: (302, "text/html", b""))
    tampered = dict(cap)
    tampered["path"] = [canonical_rfc_url(9110)]
    with pytest.raises(InternetPathRefused):
        verify_internet_navigation_capsule(tampered, pol)

    # Evidence-gated InternetObservationRouter (INTERNET-005)
    g_fp = "b" * 64
    obs_router = InternetObservationRouter({"http": lambda c: c.expected_projection["status"] == 200})
    obs_router.admit_case(
        InternetObservationCase(
            provider_id="edge.fast",
            kind="http",
            operation="observe.content",
            source_rep="url",
            target_rep="verified_head",
            ground_fingerprint=g_fp,
            observed_at=0,
            expires_at=50,
            expected_projection={"status": 200},
            observed_projection={"status": 200},
            costs=(RouteMetric("latency_ms", "ms", 5),),
        )
    )
    obs_router.admit_case(
        InternetObservationCase(
            provider_id="edge.slow",
            kind="http",
            operation="observe.content",
            source_rep="url",
            target_rep="verified_head",
            ground_fingerprint=g_fp,
            observed_at=0,
            expires_at=200,
            expected_projection={"status": 200},
            observed_projection={"status": 200},
            costs=(RouteMetric("latency_ms", "ms", 40),),
        )
    )
    sel_now10 = obs_router.route_observation(
        source_rep="url", target_rep="verified_head", ground_fingerprint=g_fp, now=10, metric_order=(("latency_ms", "ms"),)
    )
    assert sel_now10[0].provider_id == "edge.fast"
    # After edge.fast lease expires at t=50, router automatically shifts to non-expired edge.slow!
    sel_now100 = obs_router.route_observation(
        source_rep="url", target_rep="verified_head", ground_fingerprint=g_fp, now=100, metric_order=(("latency_ms", "ms"),)
    )
    assert sel_now100[0].provider_id == "edge.slow"


def test_fp_commutant_similarity_blocks_and_observer_congruence() -> None:
    p = 97
    p_mat = (
        (1, 2, 0, 1, 0, 0),
        (0, 1, 1, 0, 0, 1),
        (0, 0, 1, 3, 1, 0),
        (0, 0, 0, 1, 2, 0),
        (0, 0, 0, 0, 1, 4),
        (0, 0, 0, 0, 0, 1),
    )
    p_inv = _invert_matrix_fp(p_mat, p)
    b1 = ((3, 5, 0, 0, 0, 0), (2, 7, 0, 0, 0, 0), (0, 0, 11, 4, 0, 0), (0, 0, 6, 9, 0, 0), (0, 0, 0, 0, 13, 8), (0, 0, 0, 0, 5, 15))
    b2 = ((2, 1, 0, 0, 0, 0), (4, 3, 0, 0, 0, 0), (0, 0, 5, 2, 0, 0), (0, 0, 3, 7, 0, 0), (0, 0, 0, 0, 9, 1), (0, 0, 0, 0, 4, 6))
    a1 = _mat_mul_fp(_mat_mul_fp(p_mat, b1, p), p_inv, p)
    a2 = _mat_mul_fp(_mat_mul_fp(p_mat, b2, p), p_inv, p)
    chain = FiniteFieldOperatorChain(dimension=6, modulus=p, factors=(a1, a2))

    decomp = discover_similarity_blocks(chain)
    assert decomp.block_sizes == (2, 2, 2)
    horizon = 10**15
    v0 = [5, 19, 31, 43, 61, 89]
    jumped, stats = decomp.navigate_horizon(v0, horizon)
    direct = _mat_vec_fp(_mat_pow_fp(chain.composite_matrix(), horizon, p), v0, p)
    assert jumped == direct and stats["block_cubic_ops"] < stats["dense_cubic_ops"]

    # Irreducible / non-splittable commutant must be refused cleanly
    irred_chain = FiniteFieldOperatorChain(
        dimension=2,
        modulus=97,
        factors=(((0, 1), (1, 0)), ((1, 0), (0, 96))),  # generates full M_2(F_97), commutant = F_97 * I_2!
    )
    with pytest.raises(ObserverOrbitError):
        discover_similarity_blocks(irred_chain)

    # ComposedGroundNavigator counterexample invalidation & scheduler check
    g_spec = VirtualGroundSpec("comp.g", "1.0", "orbit", "dict", ("proj",), ("inv",), ("obs",))
    sm = StructuralMemoryRegistry()
    sm.certify_and_admit("c.a", [2, 5, 8, 11, 14], domain="Z/97Z", declared_modulus=97)
    nav = ComposedGroundNavigator(
        g_spec,
        scheduler_id="sched.comp",
        template={"a": 0},
        components=(ComposedComponentSpec("comp_a", ("a",), "c.a", "integer"),),
    )
    proj = nav.project(sm, coordinate=500, temporal=TemporalCoordinate("sched.comp", 1, VectorClock.from_mapping({"n": 1})))
    assert proj.projected_state["a"] == (2 + 3 * 500) % 97
    assert nav.verify_projection(proj, sm)[0] is True
    sm.inject_counterexample("c.a", 10, 0)
    assert nav.verify_projection(proj, sm)[0] is False
    with pytest.raises(CrossSchedulerComparisonError):
        nav.project(sm, coordinate=1, temporal=TemporalCoordinate("wrong.sched", 1, VectorClock.from_mapping({"n": 1})))

    # CrossObserverCongruenceGate: Signal8 (u=3, u=5) + ModularRing + canonical_int + contradiction refusal
    s8 = Signal8Backend()
    sv3 = s8.encode_symbol(6, observer=3)
    sv5 = s8.encode_symbol(6, observer=5)
    secs = {"o3": b"sec3", "o5": b"sec5", "om": b"secm"}
    gate = CrossObserverCongruenceGate(secs, quotient_modulus=8)
    r3 = AttributedObserverReading.sign(secs["o3"], observer_id="o3", ground_fingerprint=g_spec.fingerprint, scheduler_id="s", coordinate=7, representation_kind="signal8_phasor", raw_reading={"im": sv3.phasor.imag, "observer_unit": 3, "phase_eps": 0.0, "re": sv3.phasor.real})
    r5 = AttributedObserverReading.sign(secs["o5"], observer_id="o5", ground_fingerprint=g_spec.fingerprint, scheduler_id="s", coordinate=7, representation_kind="signal8_phasor", raw_reading={"im": sv5.phasor.imag, "observer_unit": 5, "phase_eps": 0.0, "re": sv5.phasor.real})
    rm = AttributedObserverReading.sign(secs["om"], observer_id="om", ground_fingerprint=g_spec.fingerprint, scheduler_id="s", coordinate=7, representation_kind="modular_residue", raw_reading={"modulus": 16, "residue": 14})
    assert gate.verify_congruence((r3, r5, rm))["agreed_quotient_value"] == 6

    rm_bad = AttributedObserverReading.sign(secs["om"], observer_id="om", ground_fingerprint=g_spec.fingerprint, scheduler_id="s", coordinate=7, representation_kind="canonical_int", raw_reading={"value": 5})
    with pytest.raises(ContradictoryObservationError):
        gate.verify_congruence((r3, rm_bad))


def test_mtls_partitioned_tensor_cut_failover_fencing_and_corrupted_worker_refusal(tmp_path: Path) -> None:
    pki = generate_ephemeral_mtls_pki(tmp_path / "pki", ("coord.vld.local", "w0.vld.local", "w1.vld.local", "w2.vld.local", "w3.vld.local"))
    coord_sec = b"cut-coord-secret-0123456789abcdef"
    w_secs = {f"w{i}.vld.local": f"w-sec-{i}-0123456789abcdef".encode("utf-8") for i in range(4)}
    peer_secs = {f"p.{i}": w_secs[f"w{i}.vld.local"] for i in range(4)}
    all_dns = frozenset({"coord.vld.local", *w_secs.keys()})
    g_fp = "c" * 64

    router = ComputationalFabricRouter("p.coord", peer_secs)
    # Empty router returns NO_HOST
    assert router.resolve_routes(ground_fingerprint=g_fp, operator_contract="tensor_cut.swiglu.v1", now=5, metric_order=(("latency_ms", "ms"),)).blockers == (FabricBlockerCode.NO_HOST,)

    workers: list[TensorCutWorkerNode] = []
    try:
        for i in range(4):
            dns = f"w{i}.vld.local"
            wn = TensorCutWorkerNode(
                peer_id=f"p.{i}",
                host_id=f"h.{i}",
                worker_dns=dns,
                worker_secret=w_secs[dns],
                coordinator_secret=coord_sec,
                allowed_peers=all_dns,
                server_ssl_context=create_mtls_server_context(pki[dns]),
                client_ssl_context=create_mtls_client_context(pki[dns]),
                store=PagedObjectStore(InMemoryPageBacking(4096)),
            )
            ip, port = wn.start()
            workers.append(wn)
            router.add_link(FabricLink("p.coord", f"p.{i}", (RouteMetric("latency_ms", "ms", 1 + i), RouteMetric("bytes", "B", 32))))
            router.register_advertisement(
                HostAdvertisement.signed(
                    peer_secs[f"p.{i}"],
                    peer_id=f"p.{i}",
                    host_id=f"h.{i}",
                    tls_dns=dns,
                    endpoint_host=ip,
                    port=port,
                    ground_fingerprint=g_fp,
                    operator_contract="tensor_cut.swiglu.v1",
                    substrate_kind="mtls-worker",
                    local_metrics=(RouteMetric("latency_ms", "ms", 1), RouteMetric("bytes", "B", 0)),
                    observed_at=0,
                    expires_at=500,
                )
            )

        # Resource ceiling & hop limit blocker checks
        ceil_res = router.resolve_routes(ground_fingerprint=g_fp, operator_contract="tensor_cut.swiglu.v1", now=10, metric_order=(("latency_ms", "ms"),), ceilings={"latency_ms": 1})
        assert FabricBlockerCode.RESOURCE_LIMIT in ceil_res.blockers
        hop_res = router.resolve_routes(ground_fingerprint=g_fp, operator_contract="tensor_cut.swiglu.v1", now=10, metric_order=(("latency_ms", "ms"),), max_hops=0)
        assert FabricBlockerCode.HOP_LIMIT in hop_res.blockers

        coord = PartitionedTensorCutCoordinator(
            cut_id="test.cut",
            ground_fingerprint=g_fp,
            coordinator_dns="coord.vld.local",
            coordinator_secret=coord_sec,
            worker_secrets=w_secs,
            origin_store=PagedObjectStore(InMemoryPageBacking(4096)),
            origin_server_ssl=create_mtls_server_context(pki["coord.vld.local"]),
            client_ssl=create_mtls_client_context(pki["coord.vld.local"]),
            router=router,
        )
        coord.start()
        try:
            wg = [[0.1 * ((r + c) % 5 - 2) for c in range(8)] for r in range(16)]
            wu = [[0.07 * ((2 * r + c) % 5 - 2) for c in range(8)] for r in range(16)]
            wd = [[0.09 * ((r + 2 * c) % 5 - 2) for c in range(16)] for r in range(8)]
            b = [0.01 * i for i in range(8)]
            claims = coord.prepare_and_provision_swiglu(wg, wu, wd, b, partition_count=4)
            assert len(claims) == 4 and coord.cold_provision_bytes > 0

            x_in = [[0.2 * (c + 1) for c in range(8)], [-0.15 * (c + 1) for c in range(8)]]
            rep1 = coord.execute_warm_cut(x_in)
            assert rep1.warm_weight_bytes_on_wire == 0 and rep1.reassignments_performed == 0 and rep1.scalar_oracle_max_diff < 1e-5

            # Kill worker 1 mid-session -> coordinator reassigns partition 1 to a healthy peer, increments epoch to 2,
            # provisions shard onto backup worker, and fences stale epoch 1!
            workers[1].close()
            rep2 = coord.execute_warm_cut(x_in)
            assert rep2.reassignments_performed == 1 and NumericEngine.max_abs_diff(rep2.output_2d, rep1.output_2d) < 1e-5
            with pytest.raises(StaleClaimEpochError):
                coord.verify_receipt_epoch({"assignment_epoch": 1, "partition_index": 1})

            # Adversarial worker test: worker 0 alters arithmetic output, rehashes SHA-256, and signs a valid HMAC
            # receipt -> must be deterministically caught and refused by the independent math.fsum scalar oracle!
            workers[0].corrupt_output = True
            with pytest.raises(FabricRoutingError, match="oracle"):
                coord.execute_warm_cut(x_in)
        finally:
            coord.close()
    finally:
        for w in workers:
            w.close()


def test_continuous_campaign_cli_and_real_subprocess_os_exit_crash_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1. End-to-end CLI `vld continuous` self-improving campaign
    assert cli_main(["continuous", "--horizon", "1000000000"]) == 0
    camp_json = json.loads(capsys.readouterr().out)
    assert camp_json["pass2_pareto_dominates_pass1"] is True
    assert camp_json["origin_hits_delta_pass1"] == 1 and camp_json["origin_hits_delta_pass2"] == 0
    assert camp_json["pass2_stages_recovered_from_quorum"] == 3
    assert camp_json["pass2_stages_executed_after_host_switch"] == 3
    assert "adversarial_unwrapped" in camp_json["refused_upgrades"]

    # 2. Real OS child subprocess `os._exit(73)` crash mid-pipeline + member-only Paxos Phase 1b/2a replacement host!
    m1_dir = tmp_path / "m1_pages"
    env_journal_path = tmp_path / "continuous_env.jsonl"
    q_paths = {m: tmp_path / f"{m}.jsonl" for m in ("q0", "q1", "q2")}
    exec_trace_path = tmp_path / "stage_executions.log"

    child_code = f"""
import os
from pathlib import Path
from vld.authority import QuorumAuthorityMember, QuorumLineageCoordinator, QuorumMembership
from vld.continuous import ContinuousStageSpec, ContinuousVirtualSystem
from vld.ground import NavigationCostVector, VirtualGroundSpec
from vld.journal import HashChainedJournal
from vld.memory import DiskPageBacking, PagedObjectStore

root = Path({str(tmp_path)!r})
trace = root / "stage_executions.log"
spec = VirtualGroundSpec("sub.g", "1.0", "math", "int", ("step",), ("inv",), ("v",))

def mk_stage(idx, fn_expr, crash=False):
    def _run(st):
        with trace.open("a", encoding="utf-8") as fh:
            fh.write(f"host1:s{{idx}}\\n")
            fh.flush()
        if crash:
            os._exit(73)
        return ({{"v": fn_expr(int(st["v"]))}}, NavigationCostVector(0.2, 2.0, 16.0, 1.0), f"PASS:s{{idx}}")
    return ContinuousStageSpec(f"s{{idx}}", spec, f"rep{{idx}}", f"rep{{idx+1}}", f"sched.{{idx}}", _run)

stages = (
    mk_stage(0, lambda v: v + 10),
    mk_stage(1, lambda v: v * 4),
    mk_stage(2, lambda v: v - 3, crash=True),
    mk_stage(3, lambda v: v * 2),
)
q_mem = QuorumMembership("proc.crash.lineage", ("q0", "q1", "q2"))
secs = {{m: f"proc-sec-{{m}}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}}
nodes = [QuorumAuthorityMember(m, secs[m], journal_path=root / f"{{m}}.jsonl") for m in q_mem.member_ids]
# Note: coordinator ledger is ephemeral in child memory (lost on os._exit(73)!)
coord = QuorumLineageCoordinator(q_mem, nodes, secs)
sys1 = ContinuousVirtualSystem(
    "proc.crash.pipe",
    store=PagedObjectStore(DiskPageBacking(root / "m1_pages", page_size=512)),
    quorum=coord,
    active_quorum_members=("q0", "q1"),
    journal=HashChainedJournal(root / "continuous_env.jsonl"),
)
sys1.execute_stages(stages, {{"v": 5}}, host_id="host.crash.1")
"""
    proc = subprocess.run([sys.executable, "-c", child_code], capture_output=True, text=True, check=False)
    assert proc.returncode == 73

    # Replacement Host 2 starts in parent process, talking ONLY to ("q1", "q2") (q0 offline, NO shared coordinator ledger!)
    spec = VirtualGroundSpec("sub.g", "1.0", "math", "int", ("step",), ("inv",), ("v",))

    def _mk_parent_stage(idx: int, fn: Any) -> ContinuousStageSpec:
        def _run(st: dict[str, Any]) -> tuple[dict[str, Any], NavigationCostVector, str]:
            with exec_trace_path.open("a", encoding="utf-8") as fh:
                fh.write(f"host2:s{idx}\n")
            return ({"v": fn(int(st["v"]))}, NavigationCostVector(0.2, 2.0, 16.0, 1.0), f"PASS:s{idx}")

        return ContinuousStageSpec(f"s{idx}", spec, f"rep{idx}", f"rep{idx+1}", f"sched.{idx}", _run)

    parent_stages = (
        _mk_parent_stage(0, lambda v: v + 10),
        _mk_parent_stage(1, lambda v: v * 4),
        _mk_parent_stage(2, lambda v: v - 3),
        _mk_parent_stage(3, lambda v: v * 2),
    )
    q_mem = QuorumMembership("proc.crash.lineage", ("q0", "q1", "q2"))
    secs = {m: f"proc-sec-{m}-0123456789abcdef".encode("utf-8") for m in q_mem.member_ids}
    nodes = [QuorumAuthorityMember(m, secs[m], journal_path=q_paths[m]) for m in q_mem.member_ids]
    fresh_coord = QuorumLineageCoordinator(q_mem, nodes, secs)

    sys2 = ContinuousVirtualSystem(
        "proc.crash.pipe",
        store=PagedObjectStore(DiskPageBacking(m1_dir, page_size=512)),
        quorum=fresh_coord,
        active_quorum_members=("q1", "q2"),
        journal=HashChainedJournal(env_journal_path),
    )
    res2 = sys2.execute_stages(parent_stages, {"v": 5}, host_id="host.resume.2")
    assert res2.stages_recovered_from_quorum == 2
    assert res2.stages_executed_this_invocation == 2
    # (((5 + 10) * 4) - 3) * 2 = (60 - 3) * 2 = 114
    assert res2.final_state == {"v": 114}
    assert len(res2.receipts) == 4

    # Verify host2 NEVER re-executed s0 or s1!
    lines = exec_trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["host1:s0", "host1:s1", "host1:s2", "host2:s2", "host2:s3"]

    # Zombie term=1 proposal on intersecting member q1 must be deterministically fenced!
    q1_member = next(n for n in nodes if n.member_id == "q1")
    with pytest.raises(StaleTermError):
        q1_member.vote(
            QuorumProposal(
                lineage_id=q_mem.lineage_id,
                membership_digest=q_mem.membership_digest,
                slot=3,
                term=1,
                predecessor_manifest_digest="0" * 64,
                state_manifest_digest=domain_digest("zombie", "z"),
                proposer_id="host.crash.1",
                operation_id="zombie.op",
            )
        )

    # Stage oracle failure must halt and refuse commit cleanly
    bad_stage = ContinuousStageSpec(
        "bad", spec, "in", "out", "sched.bad", lambda _st: ({"v": 0}, NavigationCostVector(0.1, 1.0, 1.0, 1.0), "FAIL:corrupt")
    )
    q_bad = QuorumLineageCoordinator(
        QuorumMembership("bad.lin", ("q0", "q1", "q2")),
        [QuorumAuthorityMember(m, secs[m]) for m in ("q0", "q1", "q2")],
        secs,
    )
    sys_bad = ContinuousVirtualSystem(
        "bad.pipe", store=PagedObjectStore(InMemoryPageBacking(512)), quorum=q_bad, active_quorum_members=("q0", "q1")
    )
    with pytest.raises(ContinuousSystemError, match="failed independent oracle"):
        sys_bad.execute_stages((bad_stage,), {"v": 1}, host_id="h.bad")
