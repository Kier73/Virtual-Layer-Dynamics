r"""Phase 10 — Native Internet Git Merkle-DAG & Smart-HTTP Pack Graph Operator (`vld.substrates.git_grounds`).

Mathematical & systems formulation:
A remote Git repository speaking Smart HTTP Wire Protocol v2 (`POST .../git-upload-pack`,
`Git-Protocol: version=2`) is not merely passive file storage—it exposes an **authenticated Merkle-DAG
reachability and set-difference operator**:
$$\Delta_{\text{Git}}(h \to w) = \operatorname{ReachableObjects}(w) \setminus \operatorname{ReachableObjects}(h).$$

1. **Wire Protocol v2 & Sideband Framing (`pkt-line`):**
   Implements exact `pkt-line` encoding/decoding (`0000` flush, `0001` delim, `0002` response-end),
   `ls-refs` head/branch observation, and `fetch` `want <w>` / `have <h>` negotiation.
2. **Independent `PACK` v2 & Delta Verification (`parse_and_verify_git_pack`):**
   Verifies the trailing 20-byte pack SHA-1 digest, inflates `commit`/`tree`/`blob`/`tag` objects, resolves
   both `OBJ_OFS_DELTA` (in-pack negative offset deltas) and `OBJ_REF_DELTA` (thin-pack OID deltas against
   receiver-held base objects), and recomputes canonical Git SHA-1 Merkle digests
   $\operatorname{SHA1}(\text{type} \,\|\, \text{" "} \,\|\, \text{len} \,\|\, \text{"\x00"} \,\|\, \text{payload})$
   plus commit-to-root-tree lineage!
3. **Mandatory Bypass Negative Control (`qualify_git_graph_delta_ground`):**
   Compares negotiated incremental graph transfer $|\text{Pack}(h \to w)|$ against un-negotiated full-closure
   transfer $|\text{Pack}(\emptyset \to w)|$ (`bypass_have=True`). Any endpoint that ignores `have` predicates
   (`mode="ignore_have"` negative control, returning $\Delta N_{\text{objects}} \ge 0$) is deterministically
   refused (`GitQualificationRefused`).
"""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import re
import struct
import threading
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen
import zlib

from vld.canonical import domain_digest
from vld.ground import NavigationCostVector, VirtualGroundSpec

_OID40_RE = re.compile(r"[0-9a-f]{40}\Z")

PKT_FLUSH = b"0000"
PKT_DELIM = b"0001"
PKT_RESPONSE_END = b"0002"

_PACK_TYPE_NAMES = {
    1: "commit",
    2: "tree",
    3: "blob",
    4: "tag",
}
_NAME_TO_PACK_TYPE = {v: k for k, v in _PACK_TYPE_NAMES.items()}


class GitGroundError(ValueError):
    """Raised when Git pkt-line framing, packfile checksums, delta instructions, or Merkle OIDs fail."""


class GitQualificationRefused(GitGroundError):
    """Raised when a candidate Git service fails graph-difference qualification or bypass negative controls."""


# ============================================================================
# 1. Git Object Hashing, Tree Parsing & `pkt-line` Wire Framing
# ============================================================================


def git_object_oid(obj_type: str, content: bytes) -> str:
    """Compute canonical Git SHA-1 OID `sha1(f'{type} {len}\\0' + content)`."""
    if obj_type not in _NAME_TO_PACK_TYPE:
        raise GitGroundError(f"Unsupported Git object type {obj_type!r}.")
    hdr = f"{obj_type} {len(content)}\x00".encode("ascii")
    return hashlib.sha1(hdr + content).hexdigest()


def parse_git_commit_tree_and_parents(commit_bytes: bytes) -> dict[str, Any]:
    """Extract `tree` OID and `parent` OIDs from a canonical Git commit payload."""
    header_part, _, msg = commit_bytes.partition(b"\n\n")
    tree_oid: str | None = None
    parents: list[str] = []
    for raw_line in header_part.decode("utf-8", errors="replace").splitlines():
        if raw_line.startswith("tree "):
            cand = raw_line[5:].strip()
            if _OID40_RE.fullmatch(cand) is None:
                raise GitGroundError("Malformed tree OID in Git commit.")
            tree_oid = cand
        elif raw_line.startswith("parent "):
            cand = raw_line[7:].strip()
            if _OID40_RE.fullmatch(cand) is None:
                raise GitGroundError("Malformed parent OID in Git commit.")
            parents.append(cand)
    if tree_oid is None:
        raise GitGroundError("Git commit object missing mandatory 'tree' header.")
    return {
        "message_subject": msg.decode("utf-8", errors="replace").splitlines()[0] if msg else "",
        "parents": tuple(parents),
        "tree_oid": tree_oid,
    }


def parse_git_tree_entries(tree_bytes: bytes) -> tuple[tuple[str, str, str], ...]:
    """Parse canonical Git binary `tree` object into `(mode, name, entry_oid_hex)` tuples."""
    entries: list[tuple[str, str, str]] = []
    pos = 0
    n = len(tree_bytes)
    while pos < n:
        nul_idx = tree_bytes.find(b"\x00", pos)
        if nul_idx == -1 or nul_idx + 21 > n:
            raise GitGroundError("Malformed binary Git tree entry.")
        mode_name = tree_bytes[pos:nul_idx].decode("utf-8", errors="strict")
        mode, sep, name = mode_name.partition(" ")
        if not sep or not name:
            raise GitGroundError("Invalid mode/name separator in Git tree.")
        oid_hex = tree_bytes[nul_idx + 1 : nul_idx + 21].hex()
        entries.append((mode, name, oid_hex))
        pos = nul_idx + 21
    return tuple(entries)


def encode_pkt_line(payload: bytes | str) -> bytes:
    raw = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    total = len(raw) + 4
    if total > 65520:
        raise GitGroundError("pkt-line payload exceeds 65516-byte wire limit.")
    return f"{total:04x}".encode("ascii") + raw


def decode_pkt_lines(stream: bytes) -> list[bytes | str]:
    """Decode Git `pkt-line` frames; returns `'FLUSH'`, `'DELIM'`, `'RESPONSE_END'`, or `bytes` payloads."""
    out: list[bytes | str] = []
    pos = 0
    n = len(stream)
    while pos < n:
        if pos + 4 > n:
            raise GitGroundError("Truncated 4-byte pkt-line length prefix.")
        hex_len = stream[pos : pos + 4]
        pos += 4
        if hex_len == PKT_FLUSH:
            out.append("FLUSH")
            continue
        if hex_len == PKT_DELIM:
            out.append("DELIM")
            continue
        if hex_len == PKT_RESPONSE_END:
            out.append("RESPONSE_END")
            continue
        try:
            length = int(hex_len.decode("ascii"), 16)
        except ValueError as exc:
            raise GitGroundError(f"Invalid pkt-line hex length {hex_len!r}.") from exc
        if length < 4 or pos + (length - 4) > n:
            raise GitGroundError(f"Invalid or truncated pkt-line payload of length {length}.")
        out.append(stream[pos : pos + length - 4])
        pos += length - 4
    return out


# ============================================================================
# 2. `PACK` v2 Binary Parser, Delta Decoder & Builder
# ============================================================================


def _read_delta_varint(buf: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise GitGroundError("Truncated varint inside Git delta stream.")
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7


def apply_git_delta(base: bytes, delta: bytes) -> bytes:
    """Apply canonical Git `PACK` copy/insert delta instructions to `base`."""
    src_size, pos = _read_delta_varint(delta, 0)
    if src_size != len(base):
        raise GitGroundError(f"Delta source size mismatch: expected {src_size}, base has {len(base)}.")
    tgt_size, pos = _read_delta_varint(delta, pos)
    out = bytearray()
    n = len(delta)
    while pos < n:
        cmd = delta[pos]
        pos += 1
        if cmd & 0x80:
            cp_off = 0
            cp_size = 0
            if cmd & 0x01:
                cp_off |= delta[pos]
                pos += 1
            if cmd & 0x02:
                cp_off |= delta[pos] << 8
                pos += 1
            if cmd & 0x04:
                cp_off |= delta[pos] << 16
                pos += 1
            if cmd & 0x08:
                cp_off |= delta[pos] << 24
                pos += 1
            if cmd & 0x10:
                cp_size |= delta[pos]
                pos += 1
            if cmd & 0x20:
                cp_size |= delta[pos] << 8
                pos += 1
            if cmd & 0x40:
                cp_size |= delta[pos] << 16
                pos += 1
            if cp_size == 0:
                cp_size = 0x10000
            if cp_off + cp_size > len(base):
                raise GitGroundError("Git delta COPY instruction exceeds base object bounds.")
            out.extend(base[cp_off : cp_off + cp_size])
        elif cmd > 0:
            if pos + cmd > n:
                raise GitGroundError("Truncated Git delta INSERT literal.")
            out.extend(delta[pos : pos + cmd])
            pos += cmd
        else:
            raise GitGroundError("Reserved zero opcode in Git delta stream.")
    if len(out) != tgt_size:
        raise GitGroundError(f"Delta reconstructed size {len(out)} != declared {tgt_size}.")
    return bytes(out)


def parse_and_verify_git_pack(
    pack_bytes: bytes,
    *,
    known_base_objects: Mapping[str, tuple[str, bytes]] | None = None,
) -> dict[str, tuple[str, bytes]]:
    """Verify a Git `PACK` v2 stream, resolve `OFS_DELTA` & `REF_DELTA`s, and return `{oid: (type, raw)}`."""
    if len(pack_bytes) < 32 or pack_bytes[:4] != b"PACK":
        raise GitGroundError("Invalid PACK magic or truncated packfile.")
    expected_trailer = pack_bytes[-20:]
    computed_trailer = hashlib.sha1(pack_bytes[:-20]).digest()
    if computed_trailer != expected_trailer:
        raise GitGroundError("Git PACK trailer SHA-1 checksum mismatch.")

    version, num_objects = struct.unpack("!II", pack_bytes[4:12])
    if version != 2:
        raise GitGroundError(f"Unsupported Git PACK version {version} (expected v2).")

    resolved_by_oid: dict[str, tuple[str, bytes]] = dict(known_base_objects or {})
    resolved_by_offset: dict[int, tuple[str, bytes]] = {}
    pending_ref_deltas: list[tuple[int, str, bytes]] = []

    pos = 12
    body_end = len(pack_bytes) - 20
    for _ in range(num_objects):
        if pos >= body_end:
            raise GitGroundError("Unexpected EOF before reading all declared PACK objects.")
        entry_offset = pos
        b0 = pack_bytes[pos]
        pos += 1
        obj_type = (b0 >> 4) & 0x07
        declared_size = b0 & 0x0F
        shift = 4
        b_cur = b0
        while b_cur & 0x80:
            if pos >= body_end:
                raise GitGroundError("Truncated PACK object header varint.")
            b_cur = pack_bytes[pos]
            pos += 1
            declared_size |= (b_cur & 0x7F) << shift
            shift += 7

        base_ofs: int | None = None
        base_oid: str | None = None
        if obj_type == 6:  # OBJ_OFS_DELTA
            b_ofs = pack_bytes[pos]
            pos += 1
            neg_off = b_ofs & 0x7F
            while b_ofs & 0x80:
                if pos >= body_end:
                    raise GitGroundError("Truncated OFS_DELTA negative offset.")
                b_ofs = pack_bytes[pos]
                pos += 1
                neg_off = ((neg_off + 1) << 7) | (b_ofs & 0x7F)
            base_ofs = entry_offset - neg_off
        elif obj_type == 7:  # OBJ_REF_DELTA
            if pos + 20 > body_end:
                raise GitGroundError("Truncated REF_DELTA 20-byte base OID.")
            base_oid = pack_bytes[pos : pos + 20].hex()
            pos += 20
        elif obj_type not in _PACK_TYPE_NAMES:
            raise GitGroundError(f"Unknown PACK object type {obj_type}.")

        decomp = zlib.decompressobj()
        inflated = decomp.decompress(memoryview(pack_bytes)[pos:body_end])
        inflated += decomp.flush()
        consumed = (body_end - pos) - len(decomp.unused_data)
        if consumed <= 0 or len(inflated) != declared_size:
            raise GitGroundError("Inflated PACK entry size mismatch.")
        pos += consumed

        if obj_type in _PACK_TYPE_NAMES:
            t_name = _PACK_TYPE_NAMES[obj_type]
            oid = git_object_oid(t_name, inflated)
            resolved_by_oid[oid] = (t_name, inflated)
            resolved_by_offset[entry_offset] = (t_name, inflated)
        elif obj_type == 6:
            assert base_ofs is not None
            if base_ofs not in resolved_by_offset:
                raise GitGroundError(f"OFS_DELTA references unknown prior pack offset {base_ofs}.")
            b_type, b_raw = resolved_by_offset[base_ofs]
            reconstructed = apply_git_delta(b_raw, inflated)
            oid = git_object_oid(b_type, reconstructed)
            resolved_by_oid[oid] = (b_type, reconstructed)
            resolved_by_offset[entry_offset] = (b_type, reconstructed)
        else:
            assert base_oid is not None
            pending_ref_deltas.append((entry_offset, base_oid, inflated))

    # Multi-pass REF_DELTA resolution (handles chained deltas cleanly)
    while pending_ref_deltas:
        progressed = False
        remaining: list[tuple[int, str, bytes]] = []
        for e_off, b_oid, d_raw in pending_ref_deltas:
            if b_oid in resolved_by_oid:
                b_type, b_raw = resolved_by_oid[b_oid]
                reconstructed = apply_git_delta(b_raw, d_raw)
                oid = git_object_oid(b_type, reconstructed)
                resolved_by_oid[oid] = (b_type, reconstructed)
                resolved_by_offset[e_off] = (b_type, reconstructed)
                progressed = True
            else:
                remaining.append((e_off, b_oid, d_raw))
        if not progressed:
            missing = [b_oid for _, b_oid, _ in remaining]
            raise GitGroundError(f"Unresolvable REF_DELTA base OIDs: {missing}")
        pending_ref_deltas = remaining

    if pos != body_end:
        raise GitGroundError(f"Trailing unparsed bytes inside PACK ({body_end - pos} bytes).")
    return resolved_by_oid


def build_git_pack_v2(
    entries: Sequence[tuple[str, bytes] | tuple[str, str, bytes]],
) -> bytes:
    """Build a canonical Git `PACK` v2 binary supporting raw `(type, data)` and `('ref_delta', base_oid, delta)`."""
    chunks: list[bytes] = [struct.pack("!4sII", b"PACK", 2, len(entries))]
    for item in entries:
        if len(item) == 2:
            t_str, raw = item
            obj_type = _NAME_TO_PACK_TYPE[t_str]
            extra_hdr = b""
        else:
            tag, base_oid_hex, raw = item
            if tag != "ref_delta" or _OID40_RE.fullmatch(base_oid_hex) is None:
                raise GitGroundError("Invalid ref_delta entry specification.")
            obj_type = 7
            extra_hdr = bytes.fromhex(base_oid_hex)

        sz = len(raw)
        first = ((obj_type & 0x07) << 4) | (sz & 0x0F)
        sz >>= 4
        hdr_bytes = bytearray()
        if sz > 0:
            first |= 0x80
        hdr_bytes.append(first)
        while sz > 0:
            nxt = sz & 0x7F
            sz >>= 7
            if sz > 0:
                nxt |= 0x80
            hdr_bytes.append(nxt)
        chunks.append(bytes(hdr_bytes))
        if extra_hdr:
            chunks.append(extra_hdr)
        chunks.append(zlib.compress(raw))

    body = b"".join(chunks)
    return body + hashlib.sha1(body).digest()


# ============================================================================
# 3. Git Smart HTTP `protocol-v2` Client, Fixture Server & Qualification Ground
# ============================================================================


@dataclass(frozen=True, slots=True)
class GitGraphDeltaObservation:
    """Verified telemetry & Merkle root receipt from executing a remote Git `want`/`have` graph difference."""

    repo_url: str
    want_oid: str
    have_oids: tuple[str, ...]
    pack_bytes_transferred: int
    pack_object_count: int
    commit_tree_oid: str
    commit_parents: tuple[str, ...]
    pack_sha1: str
    observation_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.git-graph-delta-obs.v1",
            {
                "commit_parents": list(self.commit_parents),
                "commit_tree_oid": self.commit_tree_oid,
                "have_oids": list(self.have_oids),
                "pack_bytes_transferred": self.pack_bytes_transferred,
                "pack_object_count": self.pack_object_count,
                "pack_sha1": self.pack_sha1,
                "repo_url": self.repo_url,
                "want_oid": self.want_oid,
            },
        )
        if self.observation_digest and self.observation_digest != comp:
            raise GitGroundError("GitGraphDeltaObservation digest mismatch.")
        object.__setattr__(self, "observation_digest", comp)


@dataclass(frozen=True, slots=True)
class QualifiedGitPathCertificate:
    """Proof that a Git Smart HTTP service computes genuine remote graph set-difference vs full-closure bypass."""

    repo_url: str
    base_oid: str
    target_oid: str
    root_tree_oid: str
    full_closure_pack_bytes: int
    full_closure_objects: int
    delta_pack_bytes: int
    delta_objects: int
    byte_reduction_ratio: float
    ground_spec: VirtualGroundSpec
    certificate_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.qualified-git-path-cert.v1",
            {
                "base_oid": self.base_oid,
                "delta_objects": self.delta_objects,
                "delta_pack_bytes": self.delta_pack_bytes,
                "full_closure_objects": self.full_closure_objects,
                "full_closure_pack_bytes": self.full_closure_pack_bytes,
                "ground_fp": self.ground_spec.fingerprint,
                "repo_url": self.repo_url,
                "root_tree_oid": self.root_tree_oid,
                "target_oid": self.target_oid,
            },
        )
        if self.certificate_digest and self.certificate_digest != comp:
            raise GitGroundError("QualifiedGitPathCertificate digest mismatch.")
        object.__setattr__(self, "certificate_digest", comp)


class GitPackGraphGround:
    """Native Internet Git Wire Protocol v2 Merkle-DAG observer and remote delta-difference operator."""

    def __init__(self, repo_url: str, *, timeout_sec: float = 15.0) -> None:
        clean = repo_url.rstrip("/")
        if not (clean.startswith("http://") or clean.startswith("https://")):
            raise GitGroundError("repo_url must use http:// or https://.")
        self.repo_url = clean
        self.timeout_sec = float(timeout_sec)

    @property
    def upload_pack_url(self) -> str:
        base = self.repo_url if self.repo_url.endswith(".git") else f"{self.repo_url}.git"
        # Allow local fixture URLs that already end in .git or bare path
        if self.repo_url.startswith("http://127.0.0.1:"):
            base = self.repo_url
        return f"{base}/git-upload-pack"

    def _post_v2(self, body: bytes) -> bytes:
        req = Request(
            self.upload_pack_url,
            data=body,
            headers={
                "Content-Type": "application/x-git-upload-pack-request",
                "Accept": "application/x-git-upload-pack-result",
                "Git-Protocol": "version=2",
                "User-Agent": "vld-git-pack-ground/0.7",
            },
            method="POST",
        )
        with urlopen(req, timeout=self.timeout_sec) as resp:
            return resp.read()

    def observe_refs(self, prefixes: Sequence[str] = ("HEAD", "refs/heads/")) -> dict[str, str]:
        """Execute Protocol v2 `ls-refs` to observe remote commit coordinates without cloning."""
        lines: list[bytes] = [
            encode_pkt_line("command=ls-refs\n"),
            PKT_DELIM,
            encode_pkt_line("peel\n"),
            encode_pkt_line("symrefs\n"),
        ]
        for pref in prefixes:
            lines.append(encode_pkt_line(f"ref-prefix {pref}\n"))
        lines.append(PKT_FLUSH)

        raw_resp = self._post_v2(b"".join(lines))
        frames = decode_pkt_lines(raw_resp)
        refs: dict[str, str] = {}
        for fr in frames:
            if not isinstance(fr, bytes):
                continue
            txt = fr.decode("utf-8", errors="replace").strip()
            if not txt:
                continue
            parts = txt.split(" ")
            if len(parts) >= 2 and _OID40_RE.fullmatch(parts[0]):
                refs[parts[1]] = parts[0]
        if not refs:
            raise GitGroundError(f"Zero Git refs returned from {self.repo_url!r}.")
        return refs

    def fetch_graph_delta(
        self,
        *,
        want_oid: str,
        have_oids: Sequence[str] = (),
        known_base_objects: Mapping[str, tuple[str, bytes]] | None = None,
    ) -> tuple[GitGraphDeltaObservation, dict[str, tuple[str, bytes]]]:
        """Execute remote graph set-difference `want \\ Ancestors(have)` and independently verify the PACK."""
        if _OID40_RE.fullmatch(want_oid) is None:
            raise GitGroundError(f"Invalid 40-hex want_oid {want_oid!r}.")
        clean_haves = tuple(sorted(set(have_oids)))
        for h in clean_haves:
            if _OID40_RE.fullmatch(h) is None:
                raise GitGroundError(f"Invalid 40-hex have_oid {h!r}.")

        req_frames: list[bytes] = [
            encode_pkt_line("command=fetch\n"),
            PKT_DELIM,
            encode_pkt_line("thin-pack\n"),
            encode_pkt_line("ofs-delta\n"),
            encode_pkt_line("no-progress\n"),
            encode_pkt_line(f"want {want_oid}\n"),
        ]
        for h in clean_haves:
            req_frames.append(encode_pkt_line(f"have {h}\n"))
        req_frames.append(encode_pkt_line("done\n"))
        req_frames.append(PKT_FLUSH)

        raw_resp = self._post_v2(b"".join(req_frames))
        frames = decode_pkt_lines(raw_resp)
        in_pack_section = False
        pack_chunks: list[bytes] = []
        for fr in frames:
            if fr in ("FLUSH", "RESPONSE_END"):
                continue
            if fr == "DELIM":
                in_pack_section = False
                continue
            assert isinstance(fr, bytes)
            if fr.strip() == b"packfile":
                in_pack_section = True
                continue
            if in_pack_section and len(fr) >= 1:
                band = fr[0]
                if band == 1:
                    pack_chunks.append(fr[1:])
                elif band == 3:
                    err_msg = fr[1:].decode("utf-8", errors="replace")
                    raise GitGroundError(f"Remote Git fatal sideband error: {err_msg}")

        pack_raw = b"".join(pack_chunks)
        if not pack_raw:
            raise GitGroundError("Remote Git response contained no packfile payload.")
        _, decl_count = struct.unpack("!II", pack_raw[4:12])
        all_objs = parse_and_verify_git_pack(pack_raw, known_base_objects=known_base_objects)
        if want_oid not in all_objs or all_objs[want_oid][0] != "commit":
            raise GitGroundError(f"Verified PACK did not materialize target commit {want_oid}.")

        c_info = parse_git_commit_tree_and_parents(all_objs[want_oid][1])
        root_tree = str(c_info["tree_oid"])
        if root_tree not in all_objs or all_objs[root_tree][0] != "tree":
            raise GitGroundError(f"Target commit root tree {root_tree} missing after Merkle reconstruction.")

        obs = GitGraphDeltaObservation(
            repo_url=self.repo_url,
            want_oid=want_oid,
            have_oids=clean_haves,
            pack_bytes_transferred=len(pack_raw),
            pack_object_count=decl_count,
            commit_tree_oid=root_tree,
            commit_parents=tuple(c_info["parents"]),
            pack_sha1=pack_raw[-20:].hex(),
        )
        return obs, all_objs

    def qualify_graph_delta_operator(
        self,
        *,
        base_oid: str,
        target_oid: str,
    ) -> QualifiedGitPathCertificate:
        """Qualify remote Git graph-difference computation against full-closure bypass negative control."""
        # 1. Materialize base closure at `base_oid`
        _, base_objs = self.fetch_graph_delta(want_oid=base_oid)
        # 2. Bypass negative control: full un-negotiated closure at `target_oid` (`have=()`)
        full_obs, _ = self.fetch_graph_delta(want_oid=target_oid, have_oids=())
        # 3. Negotiated incremental graph delta `target_oid \ Ancestors(base_oid)`
        delta_obs, merged_objs = self.fetch_graph_delta(
            want_oid=target_oid,
            have_oids=(base_oid,),
            known_base_objects=base_objs,
        )
        if delta_obs.commit_tree_oid != full_obs.commit_tree_oid:
            raise GitQualificationRefused("Negotiated delta reconstructed a different root tree than full fetch.")
        if delta_obs.pack_object_count >= full_obs.pack_object_count or delta_obs.pack_bytes_transferred >= full_obs.pack_bytes_transferred:
            raise GitQualificationRefused(
                f"Git endpoint failed graph-difference qualification: delta ({delta_obs.pack_object_count} objs, "
                f"{delta_obs.pack_bytes_transferred} B) did not strictly reduce full closure "
                f"({full_obs.pack_object_count} objs, {full_obs.pack_bytes_transferred} B)."
            )
        # Verify root tree entries parse cleanly
        _ = parse_git_tree_entries(merged_objs[delta_obs.commit_tree_oid][1])

        gspec = VirtualGroundSpec(
            ground_id=f"git.pack_delta.{hashlib.sha256(self.repo_url.encode('utf-8')).hexdigest()[:12]}",
            version="1.0",
            language="git-protocol-v2-pack",
            state_domain="merkle_commit_dag",
            transitions=("ls_refs", "fetch_graph_delta"),
            invariants=("pack_sha1_verified", "merkle_oid_verified", "have_set_difference_qualified"),
            observers=("root_tree_oid", "pack_bytes_transferred"),
        )
        ratio = round(delta_obs.pack_bytes_transferred / max(1, full_obs.pack_bytes_transferred), 6)
        return QualifiedGitPathCertificate(
            repo_url=self.repo_url,
            base_oid=base_oid,
            target_oid=target_oid,
            root_tree_oid=delta_obs.commit_tree_oid,
            full_closure_pack_bytes=full_obs.pack_bytes_transferred,
            full_closure_objects=full_obs.pack_object_count,
            delta_pack_bytes=delta_obs.pack_bytes_transferred,
            delta_objects=delta_obs.pack_object_count,
            byte_reduction_ratio=ratio,
            ground_spec=gspec,
        )


# ============================================================================
# 4. Hermetic Local Git Smart HTTP v2 Fixture Server (Positive + Negative Control)
# ============================================================================


def _make_tree_payload(entries: Sequence[tuple[str, str, str]]) -> bytes:
    parts: list[bytes] = []
    for mode, name, oid_hex in entries:
        parts.append(f"{mode} {name}\x00".encode("utf-8") + bytes.fromhex(oid_hex))
    return b"".join(parts)


class GitSmartHTTPFixtureServer:
    """Deterministic Git Wire Protocol v2 server (`mode='delta'` genuine vs `'ignore_have'` negative control)."""

    def __init__(self, *, mode: str = "delta", host: str = "127.0.0.1", port: int = 0) -> None:
        if mode not in ("delta", "ignore_have"):
            raise ValueError("mode must be 'delta' or 'ignore_have'.")
        self.mode = mode
        self._objects, self.refs, self._reachability = self._build_sample_dag()
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread: threading.Thread | None = None

    @staticmethod
    def _build_sample_dag() -> tuple[dict[str, tuple[str, bytes]], dict[str, str], dict[str, set[str]]]:
        objs: dict[str, tuple[str, bytes]] = {}

        def _put(t: str, raw: bytes) -> str:
            oid = git_object_oid(t, raw)
            objs[oid] = (t, raw)
            return oid

        b_readme = _put("blob", b"# Virtual Layer Dynamics\nShared immutable base payload.\n" * 16)
        b_core = _put("blob", b"def orbit(x): return (5 * x + 13) % 256\n" * 20)
        t1 = _put("tree", _make_tree_payload([("100644", "README.md", b_readme), ("100644", "core.py", b_core)]))
        c1_raw = (
            f"tree {t1}\n"
            "author Kier73 <Kier73research@gmail.com> 1763424000 +0000\n"
            "committer Kier73 <Kier73research@gmail.com> 1763424000 +0000\n\n"
            "Initial base commit\n"
        ).encode("utf-8")
        c1 = _put("commit", c1_raw)

        b_new = _put("blob", b"def verify(): return True\n")
        t2 = _put(
            "tree",
            _make_tree_payload(
                [("100644", "README.md", b_readme), ("100644", "core.py", b_core), ("100644", "verify.py", b_new)]
            ),
        )
        c2_raw = (
            f"tree {t2}\n"
            f"parent {c1}\n"
            "author Kier73 <Kier73research@gmail.com> 1763427600 +0000\n"
            "committer Kier73 <Kier73research@gmail.com> 1763427600 +0000\n\n"
            "Add concise verifier module\n"
        ).encode("utf-8")
        c2 = _put("commit", c2_raw)

        reach_c1 = {b_readme, b_core, t1, c1}
        reach_c2 = reach_c1 | {b_new, t2, c2}
        refs = {"HEAD": c2, "refs/heads/main": c2, "refs/tags/v0.1": c1}
        return objs, refs, {c1: reach_c1, c2: reach_c2}

    @property
    def base_url(self) -> str:
        addr, p = self._httpd.server_address[:2]
        return f"http://{addr}:{p}"

    def start(self) -> str:
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
            self._thread.start()
        return self.base_url

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def __enter__(self) -> GitSmartHTTPFixtureServer:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        parent = self

        class _GitHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                n_len = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(n_len)
                frames = decode_pkt_lines(body)
                cmd = ""
                wants: list[str] = []
                haves: list[str] = []
                for fr in frames:
                    if isinstance(fr, bytes):
                        txt = fr.decode("utf-8", errors="replace").strip()
                        if txt.startswith("command="):
                            cmd = txt.split("=", 1)[1]
                        elif txt.startswith("want "):
                            wants.append(txt[5:].strip())
                        elif txt.startswith("have "):
                            haves.append(txt[5:].strip())

                if cmd == "ls-refs":
                    out_lines = [encode_pkt_line(f"{oid} {ref}\n") for ref, oid in sorted(parent.refs.items())]
                    out_lines.append(PKT_FLUSH)
                    resp = b"".join(out_lines)
                elif cmd == "fetch" and wants:
                    w_oid = wants[0]
                    target_set = set(parent._reachability.get(w_oid, set()))
                    if parent.mode == "delta":
                        for h_oid in haves:
                            target_set -= parent._reachability.get(h_oid, set())
                    pack_entries = [parent._objects[oid] for oid in sorted(target_set)]
                    pack = build_git_pack_v2(pack_entries)
                    out_lines = [
                        encode_pkt_line("packfile\n"),
                        encode_pkt_line(b"\x01" + pack),
                        PKT_FLUSH,
                    ]
                    resp = b"".join(out_lines)
                else:
                    self.send_response(400)
                    self.end_headers()
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/x-git-upload-pack-result")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        return _GitHandler


__all__ = [
    "GitGraphDeltaObservation",
    "GitGroundError",
    "GitPackGraphGround",
    "GitQualificationRefused",
    "GitSmartHTTPFixtureServer",
    "QualifiedGitPathCertificate",
    "apply_git_delta",
    "build_git_pack_v2",
    "decode_pkt_lines",
    "encode_pkt_line",
    "git_object_oid",
    "parse_and_verify_git_pack",
    "parse_git_commit_tree_and_parents",
    "parse_git_tree_entries",
]
