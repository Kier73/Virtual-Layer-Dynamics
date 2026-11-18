r"""Phase 10 (`IDEA-006`) — Cellular Sheaf Cohomology & Global Observer Consistency (`vld.navigation.sheaf`).

Mathematical formulation:
Let $G = (V, E)$ be a directed multi-observer graph over exact field $\mathbb{K} \in \{\mathbb{Q}, \mathbb{F}_p\}$.
- Each observer $v \in V$ carries a local Ground stalk $\mathcal{F}(v) = \mathbb{K}^{d_v}$ ($N_0 = \sum_{v \in V} d_v$).
- Each directed overlap/bridge edge $e = (u \to v) \in E$ carries an agreement stalk $\mathcal{F}(e) = \mathbb{K}^{d_e}$
  ($N_1 = \sum_{e \in E} d_e$), linear restriction maps $R_{u \to e} \in \mathbb{K}^{d_e \times d_u}$ and
  $R_{v \to e} \in \mathbb{K}^{d_e \times d_v}$, and an observed relative offset $c_e \in \mathbb{K}^{d_e}$.
  (Direct unary anchor readings $A_v x_v = b_v$ are represented uniformly as self-edges $u = v$ with
  $R_{v \to e} - R_{u \to e} = A_v$, or via `add_anchor_observation`.)

The 0-coboundary operator $\delta^0: C^0(G; \mathcal{F}) \to C^1(G; \mathcal{F})$ acts blockwise on $x \in \mathbb{K}^{N_0}$ by:
$$(\delta^0 x)_e = R_{v \to e} x_v - R_{u \to e} x_u \quad (\text{or } A_v x_v \text{ on unary anchors}).$$

Exact Gauss-Jordan RREF on the augmented coboundary system $\bigl[\delta^0 \;\big|\; I_{N_1} \;\big|\; c\bigr]$ yields:
1. **Consistent Global Section ($[c] = 0 \in H^1(G; \mathcal{F})$):**
   Constructs $x^* \in \bigoplus_{v \in V} \mathcal{F}(v)$ satisfying $\delta^0 x^* = c$ identically across all edges,
   together with $\dim H^0(G; \mathcal{F}) = \dim \ker \delta^0$ and $\dim H^1(G; \mathcal{F}) = N_1 - \operatorname{rank}(\delta^0)$.
2. **Constructive 1-Cocycle / Fredholm Obstruction Witness ($[c] \neq 0 \in H^1(G; \mathcal{F})$):**
   Produces an exact parity-check functional $y \in \mathbb{K}^{N_1}$ satisfying:
   $$(\delta^0)^\top y = \mathbf{0} \quad \land \quad \langle y, c \rangle = \omega_{\text{holonomy}} \neq 0.$$
   Any untrusted verifier checks those two linear identities in $O(N_1)$ arithmetic without trusting RREF,
   while $\operatorname{supp}(y) = \{e \in E : y_e \neq \mathbf{0}\}$ localizes the minimal inconsistent observer cycle!
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from vld.canonical import domain_digest


class SheafConsistencyError(ValueError):
    """Raised on malformed stalk dimensions, non-prime finite field modulus, or invalid edge maps."""


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


@dataclass(frozen=True, slots=True)
class SheafEdgeSpec:
    """Directed restriction constraint `target_map @ x[target] - source_map @ x[source] = offset`."""

    edge_id: str
    source_observer: str
    target_observer: str
    edge_dim: int
    source_map: tuple[tuple[str, ...], ...]
    target_map: tuple[tuple[str, ...], ...]
    offset: tuple[str, ...]
    is_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_dim": self.edge_dim,
            "edge_id": self.edge_id,
            "is_anchor": self.is_anchor,
            "offset": list(self.offset),
            "source_map": [list(r) for r in self.source_map],
            "source_observer": self.source_observer,
            "target_map": [list(r) for r in self.target_map],
            "target_observer": self.target_observer,
        }


@dataclass(frozen=True, slots=True)
class SheafObstructionWitness:
    r"""Fredholm / 1-cocycle witness $y \in \ker((\delta^0)^\top)$ proving $\langle y, c \rangle \neq 0$."""

    dual_cocycle: tuple[str, ...]
    holonomy_residual: str
    conflicting_edge_ids: tuple[str, ...]
    witness_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.sheaf-obstruction-witness.v1",
            {
                "conflicting_edge_ids": list(self.conflicting_edge_ids),
                "dual_cocycle": list(self.dual_cocycle),
                "holonomy_residual": self.holonomy_residual,
            },
        )
        if self.witness_digest and self.witness_digest != comp:
            raise SheafConsistencyError("SheafObstructionWitness digest mismatch.")
        object.__setattr__(self, "witness_digest", comp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflicting_edge_ids": list(self.conflicting_edge_ids),
            "dual_cocycle": list(self.dual_cocycle),
            "holonomy_residual": self.holonomy_residual,
            "witness_digest": self.witness_digest,
        }


@dataclass(frozen=True, slots=True)
class SheafConsistencyReport:
    """Complete cohomological global-section or obstruction certificate for a `CellularSheaf`."""

    sheaf_id: str
    field: str
    modulus: int | None
    consistent: bool
    h0_dim: int
    h1_dim: int
    coboundary_rank: int
    global_section: Mapping[str, tuple[str, ...]] | None
    obstruction: SheafObstructionWitness | None
    sheaf_spec_dict: Mapping[str, Any]
    report_digest: str = ""

    def __post_init__(self) -> None:
        comp = domain_digest(
            "vld.sheaf-consistency-report.v1",
            {
                "coboundary_rank": self.coboundary_rank,
                "consistent": self.consistent,
                "field": self.field,
                "global_section": (
                    None
                    if self.global_section is None
                    else {k: list(v) for k, v in sorted(self.global_section.items())}
                ),
                "h0_dim": self.h0_dim,
                "h1_dim": self.h1_dim,
                "modulus": self.modulus,
                "obstruction": None if self.obstruction is None else self.obstruction.to_dict(),
                "sheaf_id": self.sheaf_id,
            },
        )
        if self.report_digest and self.report_digest != comp:
            raise SheafConsistencyError("SheafConsistencyReport digest mismatch.")
        object.__setattr__(self, "report_digest", comp)

    def to_bundle_dict(self) -> dict[str, Any]:
        """Export self-contained certificate verifiable by `verify_sheaf_obstruction_standalone`."""
        return {
            "coboundary_rank": self.coboundary_rank,
            "consistent": self.consistent,
            "field": self.field,
            "global_section": (
                None
                if self.global_section is None
                else {k: list(v) for k, v in sorted(self.global_section.items())}
            ),
            "h0_dim": self.h0_dim,
            "h1_dim": self.h1_dim,
            "modulus": self.modulus,
            "obstruction": None if self.obstruction is None else self.obstruction.to_dict(),
            "report_digest": self.report_digest,
            "schema": "vld.sheaf-consistency-bundle.v1",
            "sheaf": dict(self.sheaf_spec_dict),
            "sheaf_id": self.sheaf_id,
        }


class CellularSheaf:
    r"""Cellular sheaf $\mathcal{F}$ of local Virtual Ground stalks over exact $\mathbb{Q}$ or $\mathbb{F}_p$."""

    def __init__(
        self,
        sheaf_id: str,
        *,
        field: str = "Q",
        modulus: int | None = None,
    ) -> None:
        if not sheaf_id or not sheaf_id.strip():
            raise SheafConsistencyError("sheaf_id must be non-empty.")
        if field == "Q":
            if modulus is not None:
                raise SheafConsistencyError("modulus must be None when field == 'Q'.")
        elif field == "Fp":
            if modulus is None or isinstance(modulus, bool) or not _is_prime(int(modulus)):
                raise SheafConsistencyError("field == 'Fp' requires a prime integer modulus.")
            modulus = int(modulus)
        else:
            raise SheafConsistencyError("field must be 'Q' or 'Fp'.")

        self.sheaf_id = sheaf_id
        self.field = field
        self.modulus = modulus
        self._stalk_dims: dict[str, int] = {}
        self._edges: list[SheafEdgeSpec] = []
        self._edge_ids: set[str] = set()

    def _fmt_scalar(self, val: int | Fraction | str) -> str:
        if self.field == "Fp":
            assert self.modulus is not None
            iv = int(Fraction(val)) if not isinstance(val, int) else val
            return str(iv % self.modulus)
        f = Fraction(val)
        return f"{f.numerator}/{f.denominator}" if f.denominator != 1 else str(f.numerator)

    def add_observer(self, observer_id: str, stalk_dim: int) -> None:
        if not observer_id or not observer_id.strip():
            raise SheafConsistencyError("observer_id must be non-empty.")
        if isinstance(stalk_dim, bool) or not isinstance(stalk_dim, int) or stalk_dim <= 0:
            raise SheafConsistencyError("stalk_dim must be a positive integer.")
        if observer_id in self._stalk_dims and self._stalk_dims[observer_id] != stalk_dim:
            raise SheafConsistencyError(f"Conflicting stalk_dim for observer {observer_id!r}.")
        self._stalk_dims[observer_id] = stalk_dim

    def add_restriction_edge(
        self,
        edge_id: str,
        *,
        source_observer: str,
        target_observer: str,
        source_map: Sequence[Sequence[int | Fraction | str]],
        target_map: Sequence[Sequence[int | Fraction | str]],
        offset: Sequence[int | Fraction | str] | None = None,
    ) -> None:
        """Add overlap equation `target_map @ x[target] - source_map @ x[source] = offset`."""
        if not edge_id or edge_id in self._edge_ids:
            raise SheafConsistencyError(f"Invalid or duplicate edge_id {edge_id!r}.")
        if source_observer not in self._stalk_dims or target_observer not in self._stalk_dims:
            raise SheafConsistencyError("Both source_observer and target_observer must be registered first.")
        du = self._stalk_dims[source_observer]
        dv = self._stalk_dims[target_observer]
        de = len(source_map)
        if de <= 0 or len(target_map) != de:
            raise SheafConsistencyError("Edge restriction maps must have matching positive row count.")
        if any(len(r) != du for r in source_map) or any(len(r) != dv for r in target_map):
            raise SheafConsistencyError("Edge restriction map column count disagrees with vertex stalk_dim.")
        off_seq = tuple("0" for _ in range(de)) if offset is None else tuple(self._fmt_scalar(x) for x in offset)
        if len(off_seq) != de:
            raise SheafConsistencyError("Edge offset length disagrees with edge_dim.")

        spec = SheafEdgeSpec(
            edge_id=edge_id,
            source_observer=source_observer,
            target_observer=target_observer,
            edge_dim=de,
            source_map=tuple(tuple(self._fmt_scalar(v) for v in r) for r in source_map),
            target_map=tuple(tuple(self._fmt_scalar(v) for v in r) for r in target_map),
            offset=off_seq,
            is_anchor=False,
        )
        self._edges.append(spec)
        self._edge_ids.add(edge_id)

    def add_anchor_observation(
        self,
        anchor_id: str,
        *,
        observer_id: str,
        projection_map: Sequence[Sequence[int | Fraction | str]],
        reading: Sequence[int | Fraction | str],
    ) -> None:
        """Pin local projection `projection_map @ x[observer_id] = reading`."""
        if not anchor_id or anchor_id in self._edge_ids:
            raise SheafConsistencyError(f"Invalid or duplicate anchor_id {anchor_id!r}.")
        if observer_id not in self._stalk_dims:
            raise SheafConsistencyError(f"Unknown observer_id {observer_id!r}.")
        dv = self._stalk_dims[observer_id]
        de = len(projection_map)
        if de <= 0 or any(len(r) != dv for r in projection_map) or len(reading) != de:
            raise SheafConsistencyError("Malformed anchor projection_map or reading dimensions.")
        zero_src = tuple(tuple("0" for _ in range(dv)) for _ in range(de))
        spec = SheafEdgeSpec(
            edge_id=anchor_id,
            source_observer=observer_id,
            target_observer=observer_id,
            edge_dim=de,
            source_map=zero_src,
            target_map=tuple(tuple(self._fmt_scalar(v) for v in r) for r in projection_map),
            offset=tuple(self._fmt_scalar(x) for x in reading),
            is_anchor=True,
        )
        self._edges.append(spec)
        self._edge_ids.add(anchor_id)

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self._edges],
            "field": self.field,
            "modulus": self.modulus,
            "observers": dict(sorted(self._stalk_dims.items())),
            "sheaf_id": self.sheaf_id,
        }

    def solve_global_section(self) -> SheafConsistencyReport:
        """Solve $\\delta^0 x = c$ via exact Gauss-Jordan RREF and emit section or 1-cocycle obstruction."""
        if not self._stalk_dims or not self._edges:
            raise SheafConsistencyError("CellularSheaf requires at least one observer and one edge/anchor.")

        obs_sorted = sorted(self._stalk_dims.keys())
        v_offsets: dict[str, int] = {}
        n0 = 0
        for obs in obs_sorted:
            v_offsets[obs] = n0
            n0 += self._stalk_dims[obs]

        e_slices: list[tuple[str, int, int]] = []
        n1 = 0
        for e in self._edges:
            e_slices.append((e.edge_id, n1, n1 + e.edge_dim))
            n1 += e.edge_dim

        if self.field == "Fp":
            return self._solve_fp(obs_sorted, v_offsets, e_slices, n0, n1)
        return self._solve_q(obs_sorted, v_offsets, e_slices, n0, n1)

    def _solve_q(
        self,
        obs_sorted: list[str],
        v_offsets: Mapping[str, int],
        e_slices: Sequence[tuple[str, int, int]],
        n0: int,
        n1: int,
    ) -> SheafConsistencyReport:
        # Build augmented [delta0 (n1 x n0) | I_n1 (n1 x n1) | c (n1 x 1)] over exact Q
        total_cols = n0 + n1 + 1
        aug: list[list[Fraction]] = [[Fraction(0) for _ in range(total_cols)] for _ in range(n1)]
        row_cursor = 0
        for e in self._edges:
            u_off = v_offsets[e.source_observer]
            v_off = v_offsets[e.target_observer]
            du = self._stalk_dims[e.source_observer]
            dv = self._stalk_dims[e.target_observer]
            for r in range(e.edge_dim):
                row = aug[row_cursor]
                if not e.is_anchor:
                    for c_idx in range(du):
                        row[u_off + c_idx] -= Fraction(e.source_map[r][c_idx])
                for c_idx in range(dv):
                    row[v_off + c_idx] += Fraction(e.target_map[r][c_idx])
                row[n0 + row_cursor] = Fraction(1)
                row[n0 + n1] = Fraction(e.offset[r])
                row_cursor += 1

        piv_cols: list[int] = []
        piv_row = 0
        for col in range(n0):
            if piv_row >= n1:
                break
            sel = None
            for r in range(piv_row, n1):
                if aug[r][col] != 0:
                    sel = r
                    break
            if sel is None:
                continue
            aug[piv_row], aug[sel] = aug[sel], aug[piv_row]
            inv = Fraction(1) / aug[piv_row][col]
            aug[piv_row] = [val * inv for val in aug[piv_row]]
            for r in range(n1):
                if r != piv_row and aug[r][col] != 0:
                    fac = aug[r][col]
                    aug[r] = [aug[r][j] - fac * aug[piv_row][j] for j in range(total_cols)]
            piv_cols.append(col)
            piv_row += 1

        rank = len(piv_cols)
        h0_dim = n0 - rank
        h1_dim = n1 - rank

        # Check zero-coboundary rows r >= rank for non-zero holonomy residual aug[r][n0 + n1]
        obstruction_candidates: list[tuple[int, list[Fraction], Fraction, tuple[str, ...]]] = []
        for r in range(rank, n1):
            hol = aug[r][n0 + n1]
            if hol != 0:
                y_vec = aug[r][n0 : n0 + n1]
                supp_edges = tuple(
                    eid for eid, s_idx, e_idx in e_slices if any(y_vec[k] != 0 for k in range(s_idx, e_idx))
                )
                obstruction_candidates.append((len(supp_edges), y_vec, hol, supp_edges))

        if obstruction_candidates:
            obstruction_candidates.sort(key=lambda item: item[0])
            _, best_y, best_hol, best_edges = obstruction_candidates[0]
            obs_witness = SheafObstructionWitness(
                dual_cocycle=tuple(self._fmt_scalar(v) for v in best_y),
                holonomy_residual=self._fmt_scalar(best_hol),
                conflicting_edge_ids=best_edges,
            )
            return SheafConsistencyReport(
                sheaf_id=self.sheaf_id,
                field=self.field,
                modulus=None,
                consistent=False,
                h0_dim=h0_dim,
                h1_dim=h1_dim,
                coboundary_rank=rank,
                global_section=None,
                obstruction=obs_witness,
                sheaf_spec_dict=self.to_spec_dict(),
            )

        x_sol = [Fraction(0) for _ in range(n0)]
        for r_idx, p_col in enumerate(piv_cols):
            x_sol[p_col] = aug[r_idx][n0 + n1]
        sec_map = {
            obs: tuple(self._fmt_scalar(x_sol[v_offsets[obs] + k]) for k in range(self._stalk_dims[obs]))
            for obs in obs_sorted
        }
        return SheafConsistencyReport(
            sheaf_id=self.sheaf_id,
            field=self.field,
            modulus=None,
            consistent=True,
            h0_dim=h0_dim,
            h1_dim=h1_dim,
            coboundary_rank=rank,
            global_section=sec_map,
            obstruction=None,
            sheaf_spec_dict=self.to_spec_dict(),
        )

    def _solve_fp(
        self,
        obs_sorted: list[str],
        v_offsets: Mapping[str, int],
        e_slices: Sequence[tuple[str, int, int]],
        n0: int,
        n1: int,
    ) -> SheafConsistencyReport:
        assert self.modulus is not None
        p = self.modulus
        total_cols = n0 + n1 + 1
        aug: list[list[int]] = [[0 for _ in range(total_cols)] for _ in range(n1)]
        row_cursor = 0
        for e in self._edges:
            u_off = v_offsets[e.source_observer]
            v_off = v_offsets[e.target_observer]
            du = self._stalk_dims[e.source_observer]
            dv = self._stalk_dims[e.target_observer]
            for r in range(e.edge_dim):
                row = aug[row_cursor]
                if not e.is_anchor:
                    for c_idx in range(du):
                        row[u_off + c_idx] = (row[u_off + c_idx] - int(e.source_map[r][c_idx])) % p
                for c_idx in range(dv):
                    row[v_off + c_idx] = (row[v_off + c_idx] + int(e.target_map[r][c_idx])) % p
                row[n0 + row_cursor] = 1
                row[n0 + n1] = int(e.offset[r]) % p
                row_cursor += 1

        piv_cols: list[int] = []
        piv_row = 0
        for col in range(n0):
            if piv_row >= n1:
                break
            sel = None
            for r in range(piv_row, n1):
                if aug[r][col] % p != 0:
                    sel = r
                    break
            if sel is None:
                continue
            aug[piv_row], aug[sel] = aug[sel], aug[piv_row]
            inv = pow(aug[piv_row][col], p - 2, p)
            aug[piv_row] = [(val * inv) % p for val in aug[piv_row]]
            for r in range(n1):
                if r != piv_row and aug[r][col] != 0:
                    fac = aug[r][col]
                    aug[r] = [(aug[r][j] - fac * aug[piv_row][j]) % p for j in range(total_cols)]
            piv_cols.append(col)
            piv_row += 1

        rank = len(piv_cols)
        h0_dim = n0 - rank
        h1_dim = n1 - rank

        obstruction_candidates: list[tuple[int, list[int], int, tuple[str, ...]]] = []
        for r in range(rank, n1):
            hol = aug[r][n0 + n1] % p
            if hol != 0:
                y_vec = aug[r][n0 : n0 + n1]
                supp_edges = tuple(
                    eid for eid, s_idx, e_idx in e_slices if any(y_vec[k] % p != 0 for k in range(s_idx, e_idx))
                )
                obstruction_candidates.append((len(supp_edges), y_vec, hol, supp_edges))

        if obstruction_candidates:
            obstruction_candidates.sort(key=lambda item: item[0])
            _, best_y, best_hol, best_edges = obstruction_candidates[0]
            obs_witness = SheafObstructionWitness(
                dual_cocycle=tuple(str(v % p) for v in best_y),
                holonomy_residual=str(best_hol),
                conflicting_edge_ids=best_edges,
            )
            return SheafConsistencyReport(
                sheaf_id=self.sheaf_id,
                field=self.field,
                modulus=p,
                consistent=False,
                h0_dim=h0_dim,
                h1_dim=h1_dim,
                coboundary_rank=rank,
                global_section=None,
                obstruction=obs_witness,
                sheaf_spec_dict=self.to_spec_dict(),
            )

        x_sol = [0 for _ in range(n0)]
        for r_idx, p_col in enumerate(piv_cols):
            x_sol[p_col] = aug[r_idx][n0 + n1] % p
        sec_map = {
            obs: tuple(str(x_sol[v_offsets[obs] + k]) for k in range(self._stalk_dims[obs]))
            for obs in obs_sorted
        }
        return SheafConsistencyReport(
            sheaf_id=self.sheaf_id,
            field=self.field,
            modulus=p,
            consistent=True,
            h0_dim=h0_dim,
            h1_dim=h1_dim,
            coboundary_rank=rank,
            global_section=sec_map,
            obstruction=None,
            sheaf_spec_dict=self.to_spec_dict(),
        )


__all__ = [
    "CellularSheaf",
    "SheafConsistencyError",
    "SheafConsistencyReport",
    "SheafEdgeSpec",
    "SheafObstructionWitness",
]
