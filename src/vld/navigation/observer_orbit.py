r"""Observer-Orbit Composition, Exact $\mathbb{F}_p$ Commutant Similarity Block Discovery & Cross-Observer Congruence (`vld.navigation.observer_orbit`).

Consolidates three cross-repository mathematical pillars:
1. **Compositional Virtual Ground Projections (`E-CPU observer_orbit` + `composed_ground`):**
   Combines $K$ independently certified `StructuralMemoryRegistry` generators into a structured
   template $\mathcal{T}(t)$ at coordinate $t \in \mathbb{N}_0$ in $O(\sum_k \log t)$ with zero
   tick enumeration, bound to scheduler-scoped `TemporalCoordinate` (`G13`).
2. **Exact Prime-Field Commutant & Simultaneous Block-Diagonalization (`VGI GEOMETRY-FIRST-007`):**
   For endomorphisms $A_1, \dots, A_m \in \mathbb{F}_p^{n \times n}$ ($p$ prime), computes the exact
   centralizer algebra $\mathcal{C} = \{C \mid \forall r,\; C A_r \equiv A_r C \pmod p\}$ via RREF,
   extracts split invariant subspaces $P \in \operatorname{GL}_n(\mathbb{F}_p)$, independently certifies
   $P^{-1} A_r P = \bigoplus_{j=1}^J B_r^{(j)}$, and leaps $T$ horizons in $O(\sum_j d_j^3 \log T)$.
3. **Cross-Observer Congruence & Contradiction Refusal (`IFS GCI-001..005`):**
   Verifies naturality across non-identical observer representations (`Signal8` phasors $u \in (\mathbb{Z}/8\mathbb{Z})^\times$,
   `ModularRing`, `StructuralMemory`) and refuses contradictory readings (`ContradictoryObservationError`).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import re
from typing import Any, Mapping, Sequence

from vld.authority import CrossSchedulerComparisonError, TemporalCoordinate
from vld.canonical import canonical_bytes, canonical_json, domain_digest, parse_strict_json
from vld.ground import VirtualGroundSpec
from vld.memory import StructuralMemoryRegistry, StructuralReadReceipt
from vld.signal import Signal8Backend, SignalPhasorValue

_IDENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_HEX64_RE = re.compile(r"[0-9a-f]{64}\Z")


class ObserverOrbitError(ValueError):
    """Raised when an observer-orbit address, composed Ground, or similarity block fails validation."""


class ContradictoryObservationError(ObserverOrbitError):
    """Raised when attributed observers report conflicting readings for the same Ground coordinate (`GCI-005`)."""


# ============================================================================
# 1. Contextual Orbit & Composed Virtual Ground Projections
# ============================================================================


@dataclass(frozen=True, slots=True)
class ComposedComponentSpec:
    """Binds one nested scalar leaf of a structured state template to a certified `StructuralMemoryRegistry` ID."""

    component_id: str
    field_path: tuple[str, ...]
    memory_id: str
    value_kind: str = "integer"  # "integer", "byte", or "bit"
    coordinate_offset: int = 0

    def __post_init__(self) -> None:
        if not _IDENT_RE.fullmatch(self.component_id) or not _IDENT_RE.fullmatch(self.memory_id):
            raise ObserverOrbitError("component_id and memory_id must be valid bounded identifiers.")
        if not self.field_path or any(not isinstance(p, str) or not p.strip() for p in self.field_path):
            raise ObserverOrbitError("field_path must be a non-empty tuple of field keys.")
        if self.value_kind not in ("integer", "byte", "bit"):
            raise ObserverOrbitError(f"Unsupported value_kind {self.value_kind!r}.")
        if isinstance(self.coordinate_offset, bool) or not isinstance(self.coordinate_offset, int):
            raise ObserverOrbitError("coordinate_offset must be an integer.")


@dataclass(frozen=True, slots=True)
class ComposedGroundProjection:
    """Verified structured projection evaluated at distant coordinate `coordinate` without replay."""

    ground_fingerprint: str
    scheduler_id: str
    temporal_coordinate: TemporalCoordinate
    coordinate: int
    projected_state: Mapping[str, Any]
    component_receipts: Mapping[str, StructuralReadReceipt]
    total_expanded_intermediate_states: int
    projection_digest: str


def _write_nested_leaf(target: dict[str, Any], path: tuple[str, ...], value: int) -> None:
    cur: Any = target
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur or not isinstance(cur[key], dict):
            raise ObserverOrbitError(f"Template missing parent object along path {'.'.join(path)!r}.")
        cur = cur[key]
    leaf = path[-1]
    if not isinstance(cur, dict) or leaf not in cur or isinstance(cur[leaf], (dict, list)):
        raise ObserverOrbitError(f"Template missing scalar leaf at {'.'.join(path)!r}.")
    cur[leaf] = value


class ComposedGroundNavigator:
    """Composes independently certified Structural Memory laws into one structured Virtual Ground state."""

    def __init__(
        self,
        ground_spec: VirtualGroundSpec,
        *,
        scheduler_id: str,
        template: Mapping[str, Any],
        components: Sequence[ComposedComponentSpec],
    ) -> None:
        if not _IDENT_RE.fullmatch(scheduler_id):
            raise ObserverOrbitError("scheduler_id must be a valid bounded identifier.")
        if not components:
            raise ObserverOrbitError("ComposedGroundNavigator requires at least one component.")
        seen_ids: set[str] = set()
        seen_paths: set[tuple[str, ...]] = set()
        clean_template = parse_strict_json(canonical_json(dict(template)))
        if not isinstance(clean_template, dict):
            raise ObserverOrbitError("template must be a JSON object.")
        for comp in components:
            if comp.component_id in seen_ids or comp.field_path in seen_paths:
                raise ObserverOrbitError("Duplicate component_id or field_path in ComposedGroundNavigator.")
            seen_ids.add(comp.component_id)
            seen_paths.add(comp.field_path)
            probe_copy = parse_strict_json(canonical_json(clean_template))
            _write_nested_leaf(probe_copy, comp.field_path, 0)

        self.ground_spec = ground_spec
        self.scheduler_id = scheduler_id
        self._template_json = canonical_json(clean_template)
        self.components = tuple(components)

    def project(
        self,
        registry: StructuralMemoryRegistry,
        *,
        coordinate: int,
        temporal: TemporalCoordinate,
    ) -> ComposedGroundProjection:
        """Evaluate all composed components at `coordinate` in $O(\\sum_k \\log(\\text{coordinate}))$."""
        if isinstance(coordinate, bool) or not isinstance(coordinate, int) or coordinate < 0:
            raise ObserverOrbitError("coordinate must be a non-negative integer.")
        if temporal.scheduler_id != self.scheduler_id:
            raise CrossSchedulerComparisonError(
                f"TemporalCoordinate scheduler {temporal.scheduler_id!r} != Ground scheduler {self.scheduler_id!r}."
            )

        state_obj = parse_strict_json(self._template_json)
        receipts: dict[str, StructuralReadReceipt] = {}
        total_expanded = 0

        for comp in self.components:
            eff_coord = coordinate + comp.coordinate_offset
            if eff_coord < 0:
                raise ObserverOrbitError(f"Effective coordinate {eff_coord} < 0 for {comp.component_id!r}.")
            val, rcpt = registry.read(comp.memory_id, eff_coord)
            ival = int(val)
            if comp.value_kind == "byte" and not (0 <= ival <= 255):
                raise ObserverOrbitError(f"Component {comp.component_id!r} produced {ival} outside byte [0, 255].")
            if comp.value_kind == "bit" and ival not in (0, 1):
                raise ObserverOrbitError(f"Component {comp.component_id!r} produced {ival} outside bit {{0, 1}}.")
            _write_nested_leaf(state_obj, comp.field_path, ival)
            receipts[comp.component_id] = rcpt
            total_expanded += rcpt.expanded_intermediate_states

        proj_digest = domain_digest(
            "vld.composed-ground-projection.v1",
            {
                "component_receipts": {k: v.receipt_digest for k, v in sorted(receipts.items())},
                "coordinate": coordinate,
                "ground_fingerprint": self.ground_spec.fingerprint,
                "projected_state": state_obj,
                "scheduler_id": self.scheduler_id,
                "temporal": temporal.to_dict(),
            },
        )
        return ComposedGroundProjection(
            ground_fingerprint=self.ground_spec.fingerprint,
            scheduler_id=self.scheduler_id,
            temporal_coordinate=temporal,
            coordinate=coordinate,
            projected_state=state_obj,
            component_receipts=receipts,
            total_expanded_intermediate_states=total_expanded,
            projection_digest=proj_digest,
        )

    @staticmethod
    def verify_projection(
        projection: ComposedGroundProjection,
        registry: StructuralMemoryRegistry,
    ) -> tuple[bool, str]:
        """Revalidate every component read receipt against live `StructuralMemoryRegistry` revisions."""
        for comp_id, rcpt in projection.component_receipts.items():
            ok, reason = registry.verify_read_receipt(rcpt)
            if not ok:
                return False, f"component {comp_id}: {reason}"
        return True, "verified"


# ============================================================================
# 2. Exact Prime-Field Commutant & Simultaneous Block-Diagonalization (`GEOMETRY-FIRST-007`)
# ============================================================================

MatrixFp = tuple[tuple[int, ...], ...]


def _is_prime(p: int) -> bool:
    if p < 2:
        return False
    if p in (2, 3):
        return True
    if p % 2 == 0:
        return False
    d = 3
    while d * d <= p:
        if p % d == 0:
            return False
        d += 2
    return True


def _mat_mul_fp(a: MatrixFp, b: MatrixFp, p: int) -> MatrixFp:
    rows = len(a)
    mid = len(b)
    cols = len(b[0])
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(mid)) % p for j in range(cols))
        for i in range(rows)
    )


def _mat_vec_fp(a: MatrixFp, v: Sequence[int], p: int) -> tuple[int, ...]:
    return tuple(sum(x * y for x, y in zip(row, v)) % p for row in a)


def _mat_pow_fp(a: MatrixFp, steps: int, p: int) -> MatrixFp:
    n = len(a)
    acc: MatrixFp = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
    cur = a
    k = steps
    while k > 0:
        if k & 1:
            acc = _mat_mul_fp(cur, acc, p)
        cur = _mat_mul_fp(cur, cur, p)
        k >>= 1
    return acc


def _rref_fp(rows: list[list[int]], width: int, p: int) -> tuple[list[list[int]], list[int]]:
    m = [[v % p for v in r] for r in rows]
    pivots: list[int] = []
    lead = 0
    for col in range(width):
        piv = next((r for r in range(lead, len(m)) if m[r][col] != 0), None)
        if piv is None:
            continue
        m[lead], m[piv] = m[piv], m[lead]
        inv_v = pow(m[lead][col], -1, p)
        m[lead] = [(u * inv_v) % p for u in m[lead]]
        for r in range(len(m)):
            if r != lead and m[r][col] != 0:
                fac = m[r][col]
                m[r] = [(x - fac * y) % p for x, y in zip(m[r], m[lead])]
        pivots.append(col)
        lead += 1
        if lead == len(m):
            break
    return m, pivots


def _nullspace_fp(rows: list[list[int]], width: int, p: int) -> tuple[tuple[int, ...], ...]:
    reduced, pivots = _rref_fp(rows, width, p)
    piv_set = set(pivots)
    free_cols = [j for j in range(width) if j not in piv_set]
    basis: list[tuple[int, ...]] = []
    for f_col in free_cols:
        vec = [0] * width
        vec[f_col] = 1
        for r_idx, p_col in enumerate(pivots):
            vec[p_col] = (-reduced[r_idx][f_col]) % p
        basis.append(tuple(vec))
    return tuple(basis)


def _invert_matrix_fp(mat: MatrixFp, p: int) -> MatrixFp:
    n = len(mat)
    aug = [
        [mat[i][j] % p for j in range(n)] + [1 if i == j else 0 for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        piv = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if piv is None:
            raise ObserverOrbitError("Candidate similarity basis matrix is singular over F_p.")
        aug[col], aug[piv] = aug[piv], aug[col]
        inv_v = pow(aug[col][col], -1, p)
        aug[col] = [(u * inv_v) % p for u in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                fac = aug[r][col]
                aug[r] = [(x - fac * y) % p for x, y in zip(aug[r], aug[col])]
    return tuple(tuple(row[n:]) for row in aug)


@dataclass(frozen=True, slots=True)
class FiniteFieldOperatorChain:
    r"""Endomorphism chain $A_m \circ \dots \circ A_1$ over $\mathbb{F}_p^n$ ($p$ prime)."""

    dimension: int
    modulus: int
    factors: tuple[MatrixFp, ...]

    def __post_init__(self) -> None:
        if not (2 <= self.dimension <= 32):
            raise ObserverOrbitError("dimension must lie in [2, 32].")
        if not (2 <= self.modulus <= 65537) or not _is_prime(self.modulus):
            raise ObserverOrbitError(f"modulus {self.modulus} must be a prime in [2, 65537].")
        if not self.factors:
            raise ObserverOrbitError("factors must be non-empty.")
        clean_factors: list[MatrixFp] = []
        for f in self.factors:
            if len(f) != self.dimension or any(len(r) != self.dimension for r in f):
                raise ObserverOrbitError("Each factor must be an n x n matrix.")
            clean_factors.append(tuple(tuple(int(v) % self.modulus for v in r) for r in f))
        object.__setattr__(self, "factors", tuple(clean_factors))

    def composite_matrix(self) -> MatrixFp:
        n = self.dimension
        acc: MatrixFp = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
        for f in self.factors:
            acc = _mat_mul_fp(f, acc, self.modulus)
        return acc


@dataclass(frozen=True, slots=True)
class CertifiedSimilarityDecomposition:
    r"""Verified simultaneous block-diagonalization $P^{-1} A_r P = \bigoplus_{j=1}^J B_r^{(j)}$ over $\mathbb{F}_p$."""

    chain: FiniteFieldOperatorChain
    commutant_dimension: int
    basis: MatrixFp
    basis_inverse: MatrixFp
    block_sizes: tuple[int, ...]
    block_factors: tuple[tuple[MatrixFp, ...], ...]  # [num_blocks][num_factors]
    certificate_digest: str

    def verify_independently(self) -> bool:
        """Independently verify $P^{-1} P = I$ and exact zero off-block coupling for every factor $A_r$."""
        n = self.chain.dimension
        p = self.chain.modulus
        eye = tuple(tuple(1 if i == j else 0 for j in range(n)) for i in range(n))
        if _mat_mul_fp(self.basis_inverse, self.basis, p) != eye:
            raise ObserverOrbitError("basis_inverse * basis != I over F_p.")
        if _mat_mul_fp(self.basis, self.basis_inverse, p) != eye:
            raise ObserverOrbitError("basis * basis_inverse != I over F_p.")
        if sum(self.block_sizes) != n or len(self.block_sizes) < 2:
            raise ObserverOrbitError("Invalid split block_sizes.")

        ranges: list[tuple[int, int]] = []
        cur = 0
        for sz in self.block_sizes:
            ranges.append((cur, cur + sz))
            cur += sz

        for f_idx, a_mat in enumerate(self.chain.factors):
            sim = _mat_mul_fp(_mat_mul_fp(self.basis_inverse, a_mat, p), self.basis, p)
            for b_a, (s_a, e_a) in enumerate(ranges):
                expected_blk = self.block_factors[b_a][f_idx]
                actual_blk = tuple(tuple(sim[r][c] for c in range(s_a, e_a)) for r in range(s_a, e_a))
                if actual_blk != expected_blk:
                    raise ObserverOrbitError(f"Diagonal block {b_a} mismatch on factor {f_idx}.")
                for b_b, (s_b, e_b) in enumerate(ranges):
                    if b_a != b_b:
                        if any(sim[r][c] != 0 for r in range(s_a, e_a) for c in range(s_b, e_b)):
                            raise ObserverOrbitError(f"Non-zero off-diagonal coupling between blocks {b_a} and {b_b}.")
        return True

    def navigate_horizon(self, initial_vector: Sequence[int], steps: int) -> tuple[tuple[int, ...], dict[str, int]]:
        r"""Advance $(A_m \cdots A_1)^{\text{steps}} v_0 \bmod p$ via decoupled diagonal blocks in $O(\sum_j d_j^3 \log T)$."""
        if len(initial_vector) != self.chain.dimension or steps < 0:
            raise ObserverOrbitError("Invalid initial_vector length or negative steps.")
        p = self.chain.modulus
        v0 = tuple(int(x) % p for x in initial_vector)
        local_v = _mat_vec_fp(self.basis_inverse, v0, p)

        out_local: list[int] = []
        offset = 0
        block_mul_ops = 0
        for b_idx, sz in enumerate(self.block_sizes):
            sub_v = local_v[offset : offset + sz]
            offset += sz
            comp_blk: MatrixFp = tuple(tuple(1 if i == j else 0 for j in range(sz)) for i in range(sz))
            for f_blk in self.block_factors[b_idx]:
                comp_blk = _mat_mul_fp(f_blk, comp_blk, p)
            powered_blk = _mat_pow_fp(comp_blk, steps, p)
            block_mul_ops += (sz**3) * max(1, steps.bit_length())
            out_local.extend(_mat_vec_fp(powered_blk, sub_v, p))

        result = _mat_vec_fp(self.basis, out_local, p)
        dense_ops = (self.chain.dimension**3) * max(1, steps.bit_length())
        return result, {"block_cubic_ops": block_mul_ops, "dense_cubic_ops": dense_ops}


def discover_similarity_blocks(
    chain: FiniteFieldOperatorChain,
    *,
    max_candidates: int = 32,
) -> CertifiedSimilarityDecomposition:
    r"""Discover and independently certify a common invariant block-diagonal basis over $\mathbb{F}_p$."""
    n = chain.dimension
    p = chain.modulus

    # Solve C * A_r - A_r * C == 0 (mod p) simultaneously across all factors A_r
    equations: list[list[int]] = []
    for a in chain.factors:
        for i in range(n):
            for j in range(n):
                row = [0] * (n * n)
                for k in range(n):
                    row[i * n + k] = (row[i * n + k] + a[k][j]) % p
                    row[k * n + j] = (row[k * n + j] - a[i][k]) % p
                equations.append(row)

    comm_vecs = _nullspace_fp(equations, n * n, p)
    comm_mats: list[MatrixFp] = [
        tuple(tuple(vec[i * n : (i + 1) * n]) for i in range(n))
        for vec in comm_vecs
    ]
    if len(comm_mats) <= 1:
        raise ObserverOrbitError("Commutant is 1-dimensional (scalars only); no split similarity blocks exist over F_p.")

    # Deterministic candidate weights: basis vectors first, then small deterministic combinations
    candidates: list[MatrixFp] = list(comm_mats)
    k_dim = len(comm_mats)
    for t in range(1, max_candidates):
        weights = [((t * (idx + 3) + 1) % p) for idx in range(k_dim)]
        cand = tuple(
            tuple(sum(weights[b] * comm_mats[b][i][j] for b in range(k_dim)) % p for j in range(n))
            for i in range(n)
        )
        candidates.append(cand)

    # Limit eigenvalue scan for large primes to witnessed diagonal entries + small residues + deterministic probes;
    # evaluate candidates to select the certified split minimizing cubic complexity sum(d_j^3)!
    best_decomp: CertifiedSimilarityDecomposition | None = None
    best_score: tuple[int, int] | None = None  # (sum(d_j^3), -num_blocks)

    for cand in candidates[:max_candidates]:
        eig_candidates: set[int] = set(range(min(p, 257)))
        for i in range(n):
            eig_candidates.add(cand[i][i] % p)
        eig_spaces: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
        total_dim = 0
        for lam in sorted(eig_candidates):
            shifted = [[(cand[r][c] - (lam if r == c else 0)) % p for c in range(n)] for r in range(n)]
            eb = _nullspace_fp(shifted, n, p)
            if eb:
                eig_spaces.append((lam, eb))
                total_dim += len(eb)
                if total_dim == n:
                    break

        if total_dim == n and len(eig_spaces) >= 2:
            cols: list[tuple[int, ...]] = []
            block_sizes: list[int] = []
            for _lam, eb in eig_spaces:
                block_sizes.append(len(eb))
                cols.extend(eb)
            cand_score = (sum(sz**3 for sz in block_sizes), -len(block_sizes))
            if best_score is not None and cand_score >= best_score:
                continue

            # Basis P has eigenvectors as columns!
            basis: MatrixFp = tuple(tuple(cols[col_i][row_i] for col_i in range(n)) for row_i in range(n))
            inv_basis = _invert_matrix_fp(basis, p)

            ranges: list[tuple[int, int]] = []
            cur = 0
            for sz in block_sizes:
                ranges.append((cur, cur + sz))
                cur += sz

            # Extract local factors and verify zero off-diagonal entries!
            sim_factors = [_mat_mul_fp(_mat_mul_fp(inv_basis, f, p), basis, p) for f in chain.factors]
            valid_split = True
            for sim in sim_factors:
                for b_a, (s_a, e_a) in enumerate(ranges):
                    for b_b, (s_b, e_b) in enumerate(ranges):
                        if b_a != b_b and any(sim[r][c] != 0 for r in range(s_a, e_a) for c in range(s_b, e_b)):
                            valid_split = False
                            break
                    if not valid_split:
                        break
                if not valid_split:
                    break

            if valid_split:
                blk_factors = tuple(
                    tuple(
                        tuple(tuple(sim[r][c] for c in range(s_a, e_a)) for r in range(s_a, e_a))
                        for sim in sim_factors
                    )
                    for (s_a, e_a) in ranges
                )
                cert_digest = domain_digest(
                    "vld.similarity-blocks.v1",
                    {
                        "basis": [list(r) for r in basis],
                        "block_sizes": block_sizes,
                        "commutant_dimension": k_dim,
                        "dimension": n,
                        "modulus": p,
                    },
                )
                decomp = CertifiedSimilarityDecomposition(
                    chain=chain,
                    commutant_dimension=k_dim,
                    basis=basis,
                    basis_inverse=inv_basis,
                    block_sizes=tuple(block_sizes),
                    block_factors=blk_factors,
                    certificate_digest=cert_digest,
                )
                decomp.verify_independently()
                best_decomp = decomp
                best_score = cand_score
                if len(block_sizes) >= k_dim or len(block_sizes) == n:
                    break

    if best_decomp is not None:
        return best_decomp

    raise ObserverOrbitError("No split semisimple invariant-subspace witness found within candidate budget.")


# ============================================================================
# 3. Cross-Observer Congruence & Contradiction Refusal (`GCI-001..005`)
# ============================================================================


@dataclass(frozen=True, slots=True)
class AttributedObserverReading:
    """Signed/MAC-authenticated reading of `(ground_fingerprint, coordinate)` in a declared representation."""

    observer_id: str
    ground_fingerprint: str
    scheduler_id: str
    coordinate: int
    representation_kind: str  # "canonical_int", "signal8_phasor", "modular_residue"
    raw_reading: Mapping[str, Any]
    signature_mac: str

    @staticmethod
    def sign(
        secret: bytes,
        *,
        observer_id: str,
        ground_fingerprint: str,
        scheduler_id: str,
        coordinate: int,
        representation_kind: str,
        raw_reading: Mapping[str, Any],
    ) -> AttributedObserverReading:
        if not _HEX64_RE.fullmatch(ground_fingerprint):
            raise ObserverOrbitError("ground_fingerprint must be 64-char hex.")
        payload = {
            "coordinate": coordinate,
            "domain": "vld.observer-reading.v1",
            "ground_fingerprint": ground_fingerprint,
            "observer_id": observer_id,
            "raw_reading": dict(raw_reading),
            "representation_kind": representation_kind,
            "scheduler_id": scheduler_id,
        }
        mac = hmac.new(secret, canonical_bytes(payload), hashlib.sha256).hexdigest()
        return AttributedObserverReading(
            observer_id=observer_id,
            ground_fingerprint=ground_fingerprint,
            scheduler_id=scheduler_id,
            coordinate=coordinate,
            representation_kind=representation_kind,
            raw_reading=parse_strict_json(canonical_json(dict(raw_reading))),
            signature_mac=mac,
        )


class CrossObserverCongruenceGate:
    """Verifies observer congruence across distinct representations and refuses contradictions (`GCI-005`)."""

    def __init__(self, observer_secrets: Mapping[str, bytes], *, quotient_modulus: int = 8) -> None:
        if quotient_modulus < 2:
            raise ObserverOrbitError("quotient_modulus must be >= 2.")
        self._secrets = dict(observer_secrets)
        self.quotient_modulus = quotient_modulus
        self._sig8 = Signal8Backend()

    def _normalize_reading(self, r: AttributedObserverReading) -> int:
        secret = self._secrets.get(r.observer_id)
        if secret is None:
            raise ObserverOrbitError(f"Unregistered observer {r.observer_id!r}.")
        expected = AttributedObserverReading.sign(
            secret,
            observer_id=r.observer_id,
            ground_fingerprint=r.ground_fingerprint,
            scheduler_id=r.scheduler_id,
            coordinate=r.coordinate,
            representation_kind=r.representation_kind,
            raw_reading=r.raw_reading,
        )
        if not hmac.compare_digest(r.signature_mac, expected.signature_mac):
            raise ObserverOrbitError(f"Invalid observer MAC on reading from {r.observer_id!r}.")

        if r.representation_kind == "canonical_int":
            return int(r.raw_reading["value"]) % self.quotient_modulus
        if r.representation_kind == "modular_residue":
            mod = int(r.raw_reading["modulus"])
            if mod % self.quotient_modulus != 0:
                raise ObserverOrbitError(f"Residue modulus {mod} does not project cleanly modulo {self.quotient_modulus}.")
            return int(r.raw_reading["residue"]) % self.quotient_modulus
        if r.representation_kind == "signal8_phasor":
            if self.quotient_modulus != 8:
                raise ObserverOrbitError("signal8_phasor requires quotient_modulus == 8.")
            sv = SignalPhasorValue(
                phasor=complex(float(r.raw_reading["re"]), float(r.raw_reading["im"])),
                observer=int(r.raw_reading["observer_unit"]),
                phase_error_bound=float(r.raw_reading.get("phase_eps", 0.0)),
            )
            reframed = self._sig8.reframe(sv, 1)
            return int(self._sig8.observe_symbol(reframed))
        raise ObserverOrbitError(f"Unsupported representation_kind {r.representation_kind!r}.")

    def verify_congruence(self, readings: Sequence[AttributedObserverReading]) -> dict[str, Any]:
        """Verify at least two attributed readings agree on Ground, scheduler, coordinate, and quotient value."""
        if len(readings) < 2:
            raise ObserverOrbitError("verify_congruence requires at least 2 readings.")
        anchor = readings[0]
        values: dict[str, int] = {}
        for r in readings:
            if r.ground_fingerprint != anchor.ground_fingerprint:
                raise ObserverOrbitError("Cross-Ground reading mismatch in congruence check.")
            if r.scheduler_id != anchor.scheduler_id:
                raise CrossSchedulerComparisonError("Cross-scheduler reading mismatch in congruence check.")
            if r.coordinate != anchor.coordinate:
                raise ObserverOrbitError("Coordinate mismatch in congruence check.")
            norm_v = self._normalize_reading(r)
            values[r.observer_id] = norm_v

        distinct = set(values.values())
        if len(distinct) != 1:
            raise ContradictoryObservationError(
                f"Contradictory attributed readings at coordinate {anchor.coordinate}: {values}"
            )
        agreed = next(iter(distinct))
        return {
            "agreed_quotient_value": agreed,
            "coordinate": anchor.coordinate,
            "ground_fingerprint": anchor.ground_fingerprint,
            "observer_values": values,
            "quotient_modulus": self.quotient_modulus,
            "receipt_digest": domain_digest("vld.observer-congruence.v1", {"coord": anchor.coordinate, "values": values}),
        }


__all__ = [
    "AttributedObserverReading",
    "CertifiedSimilarityDecomposition",
    "ComposedComponentSpec",
    "ComposedGroundNavigator",
    "ComposedGroundProjection",
    "ContradictoryObservationError",
    "CrossObserverCongruenceGate",
    "FiniteFieldOperatorChain",
    "ObserverOrbitError",
    "discover_similarity_blocks",
]
