r"""Multi-rung deterministic Law Certifier, closed-form orbit jumpers, and exact inverse locators.

Mathematical ladder of generative state families:
1. `int`: $x_t \equiv (x_0 + \delta t) \pmod M$ — $O(1)$ forward & $O(1)$ $\gcd$-inverse lookup.
2. `lcg`: $x_{t+1} \equiv (a x_t + c) \pmod M$ — $O(\log t)$ dual-derivation forward jump;
   $O(\log M)$ 2-adic bit-lifting inverse for full-period $M = 2^r$, $O(\sqrt{M})$ BSGS otherwise.
3. `gf2`: $s_n + \sum_{i=1}^L c_i s_{n-i} \equiv 0 \pmod 2$ — Berlekamp-Massey minimal polynomial
   with $O(L^3 \log t)$ companion-matrix jump and $O(L)$ first-bit inverse lookup.
4. `poly`: $x_t = \sum_{k=0}^d \Delta^k x_0 \binom{t}{k} \pmod M$ (or over $\mathbb{Z}$) — $O(d)$
   exact Newton binomial evaluation at arbitrary coordinates $t \ge 0$.
5. `fgraph`: Eventual periodic finite orbit $(\mu, \lambda, \text{orbit})$ — $O(1)$ modular cycle read
   and $O(\mu + \lambda)$ exact inverse lookup.

Strict identifiability contract:
No family is certified (`GROUND`) without positive residual predictive degrees of freedom ($df \ge 1$)
verified against held-out observations and explicit domain bounds where unwrapped prefixes are ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from vld.canonical import canonical_json, domain_digest
from vld.ground import AffineOrbitLaw, propose_modular_progression
from vld.triad import EpistemicStatus


class LawCertificationRefused(ValueError):
    """Raised when observations are underdetermined, non-deterministic, or outside an admitted family."""


class HorizonExceededError(LawCertificationRefused):
    r"""Raised when navigating at or beyond a `PREFIX-GROUND` escape coordinate $t_{\text{horizon}}$."""


def berlekamp_massey_gf2(bits: Sequence[int]) -> tuple[tuple[int, ...], int]:
    r"""Compute minimal connection polynomial $C(D) = 1 + c_1 D + \dots + c_L D^L$ over $\mathbb{F}_2$."""
    for b in bits:
        if isinstance(b, bool) or not isinstance(b, int) or b not in (0, 1):
            raise LawCertificationRefused("GF(2) sequence must contain exact integers in {0, 1}.")
    c = [1]
    b_poly = [1]
    length = 0
    m = 1
    for n, sn in enumerate(bits):
        discrepancy = sn
        for i in range(1, length + 1):
            discrepancy ^= c[i] & bits[n - i]
        if discrepancy == 0:
            m += 1
        else:
            temp = list(c)
            need = len(b_poly) + m
            if len(c) < need:
                c.extend([0] * (need - len(c)))
            for j, bj in enumerate(b_poly):
                c[j + m] ^= bj
            if 2 * length <= n:
                length = n + 1 - length
                b_poly = temp
                m = 1
            else:
                m += 1
    return tuple(c[: length + 1]), length


def lfsr_jump_gf2(coeffs: Sequence[int], init_state: Sequence[int], coordinate: int) -> int:
    r"""Evaluate $s_t \in \{0, 1\}$ at coordinate $t \ge 0$ in $O(L^3 \log t)$ companion-matrix steps."""
    order = len(init_state)
    if order == 0:
        return 0
    if coordinate < order:
        return int(init_state[coordinate])

    # Companion matrix M in GF(2)^{L x L}: next_state[i] = state[i+1] for i < L-1,
    # next_state[L-1] = sum_{j=1..L} coeffs[j] * state[L-j] mod 2.
    # Packed row bitmasks accelerate GF(2) inner products in pure Python!
    def mat_mul_gf2(a_rows: list[int], b_rows: list[int]) -> list[int]:
        # Precompute transpose of b_rows
        b_cols = [0] * order
        for r_idx in range(order):
            row_val = b_rows[r_idx]
            for c_idx in range(order):
                if (row_val >> c_idx) & 1:
                    b_cols[c_idx] |= 1 << r_idx
        out_rows = [0] * order
        for r_idx in range(order):
            ar = a_rows[r_idx]
            acc_row = 0
            for c_idx in range(order):
                if ((ar & b_cols[c_idx]).bit_count()) & 1:
                    acc_row |= 1 << c_idx
            out_rows[r_idx] = acc_row
        return out_rows

    base_rows = [0] * order
    for i in range(order - 1):
        base_rows[i] = 1 << (i + 1)
    feedback_mask = 0
    for j in range(1, min(len(coeffs), order + 1)):
        if coeffs[j] & 1:
            feedback_mask |= 1 << (order - j)
    base_rows[order - 1] = feedback_mask

    acc_rows = [1 << i for i in range(order)]
    rem = coordinate
    while rem > 0:
        if rem & 1:
            acc_rows = mat_mul_gf2(acc_rows, base_rows)
        rem >>= 1
        if rem > 0:
            base_rows = mat_mul_gf2(base_rows, base_rows)

    state_mask = 0
    for idx, bit in enumerate(init_state):
        if bit & 1:
            state_mask |= 1 << idx
    # Row 0 of M^t dotted with init_state gives s_t!
    return (acc_rows[0] & state_mask).bit_count() & 1


def locate_int_progression(x0: int, step: int, mod: int, target: int) -> int | None:
    r"""Invert $x_0 + \text{step} \cdot t \equiv \text{target} \pmod{\text{mod}}$ for smallest $t \ge 0$ in $O(1)$."""
    if not (0 <= target < mod):
        return None
    diff = (target - x0) % mod
    if diff == 0:
        return 0
    if step % mod == 0:
        return None
    g = math.gcd(step, mod)
    if diff % g != 0:
        return None
    step_p = (step // g) % (mod // g)
    mod_p = mod // g
    return ((diff // g) * pow(step_p, -1, mod_p)) % mod_p


def locate_lcg_2adic(a: int, c: int, mod: int, x0: int, target: int) -> int | None:
    r"""Invert full-period power-of-two LCG ($M=2^r, a \equiv 1 \pmod 4, c \text{ odd}$) in $O(\log M)$ steps."""
    if mod < 8 or (mod & (mod - 1)) != 0 or (a % 4) != 1 or (c % 2) == 0:
        return None
    if not (0 <= target < mod):
        return None
    r = mod.bit_length() - 1
    order_bits = r - 2
    d0 = ((a - 1) * x0 + c) % mod
    dt = ((a - 1) * target + c) % mod
    if (d0 & 1) == 0 or (dt & 1) == 0:
        return None
    ratio = (dt * pow(d0, -1, mod)) % mod
    s = 0
    law = AffineOrbitLaw(a % mod, c % mod, mod)
    for k in range(order_bits):
        apk = pow(a, 1 << k, mod)
        mask = (1 << (k + 3)) - 1
        inv_as = pow(pow(a, s, mod), -1, mod)
        residue = ((ratio * inv_as) % mod) & mask
        if residue == (1 & mask):
            bit = 0
        elif residue == (apk & mask):
            bit = 1
        else:
            return None
        s += bit << k

    shift = 1 << order_bits
    candidates = [s + k * shift for k in range(4) if s + k * shift < mod]
    for cand_t in sorted(candidates):
        if law.navigate(x0 % mod, cand_t).value == target:
            return cand_t
    return None


def locate_lcg_bsgs(
    a: int,
    c: int,
    mod: int,
    x0: int,
    target: int,
    *,
    max_modulus: int = 10_000_000,
) -> int | None:
    r"""Invert coprime LCG ($\gcd(a, M) = 1$) via Baby-Step Giant-Step in $O(\sqrt{M})$ time."""
    if not (0 <= target < mod):
        return None
    if math.gcd(a, mod) != 1 or mod > max_modulus:
        return None
    law = AffineOrbitLaw(a % mod, c % mod, mod)
    step_sz = math.isqrt(mod) + 1
    a_inv = pow(a, -1, mod)
    c_inv = (-a_inv * c) % mod
    inv_law = AffineOrbitLaw(a_inv, c_inv, mod)

    baby: dict[int, int] = {}
    cur = x0 % mod
    for j in range(step_sz):
        baby.setdefault(cur, j)
        cur = (a * cur + c) % mod

    best: int | None = None
    for i in range(step_sz + 1):
        cand_state = inv_law.navigate(target, i * step_sz).value
        if cand_state in baby:
            cand_t = i * step_sz + baby[cand_state]
            if law.navigate(x0 % mod, cand_t).value == target:
                if best is None or cand_t < best:
                    best = cand_t
                break
    return best


def locate_gf2_bit(coeffs: Sequence[int], init_state: Sequence[int], target: int) -> int | None:
    r"""Locate smallest coordinate $t \ge 0$ emitting `target` $\in \{0, 1\}$ in $O(L)$ checks."""
    if isinstance(target, bool) or not isinstance(target, int) or target not in (0, 1) or not init_state:
        return None
    for idx, bit in enumerate(init_state):
        if bit == target:
            return idx
    # Initial state is uniform: either all-0 or all-1.
    if init_state[0] == 0:
        return None  # Zero state is an absorbing fixed point for homogeneous GF(2) LFSR.
    # All-1 initial state: next bit at coordinate L is sum_{j=1..L} coeffs[j] mod 2.
    order = len(init_state)
    if len(coeffs) < order + 1:
        return None
    next_bit = sum(coeffs[j] for j in range(1, order + 1)) & 1
    return order if next_bit == target else None


@dataclass(frozen=True, slots=True)
class CertifiedLaw:
    """Immutable deterministic generator law with closed-form `evaluate` and `locate_value`."""

    family: str
    params: Mapping[str, Any]
    order: int
    verdict: str = "GROUND"
    first_bad: int | None = None
    epistemic_status: EpistemicStatus = EpistemicStatus.CERTIFIED

    ADMITTED_FAMILIES = frozenset({"int", "lcg", "gf2", "poly", "fgraph"})

    def __post_init__(self) -> None:
        if self.family not in self.ADMITTED_FAMILIES:
            raise LawCertificationRefused(f"Unsupported law family {self.family!r}.")
        if self.verdict not in ("GROUND", "PREFIX-GROUND"):
            raise LawCertificationRefused(f"Verdict {self.verdict!r} cannot be instantiated as CertifiedLaw.")
        if self.verdict == "GROUND" and self.first_bad is not None:
            raise LawCertificationRefused("GROUND law cannot carry first_bad.")
        if self.verdict == "PREFIX-GROUND" and (
            isinstance(self.first_bad, bool) or not isinstance(self.first_bad, int) or self.first_bad < 1
        ):
            raise LawCertificationRefused("PREFIX-GROUND law requires integer first_bad >= 1.")
        clean_params = json_freeze(dict(self.params))
        object.__setattr__(self, "params", clean_params)

    @property
    def digest(self) -> str:
        return domain_digest("vld.certified-law.v1", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "epistemic_status": self.epistemic_status.value,
            "family": self.family,
            "first_bad": self.first_bad,
            "order": self.order,
            "params": json_thaw(self.params),
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CertifiedLaw:
        return cls(
            family=str(data["family"]),
            params=dict(data["params"]),
            order=int(data["order"]),
            verdict=str(data["verdict"]),
            first_bad=data.get("first_bad"),
            epistemic_status=EpistemicStatus(data.get("epistemic_status", EpistemicStatus.CERTIFIED.value)),
        )

    def evaluate(self, coordinate: int) -> int:
        """Evaluate $x_t$ directly at `coordinate` without expanding intermediate states."""
        if isinstance(coordinate, bool) or not isinstance(coordinate, int) or coordinate < 0:
            raise ValueError("coordinate must be a non-negative integer.")
        if self.verdict == "PREFIX-GROUND" and self.first_bad is not None and coordinate >= self.first_bad:
            raise HorizonExceededError(
                f"Coordinate {coordinate} >= PREFIX-GROUND escape horizon {self.first_bad}."
            )
        p = self.params
        if self.family == "int":
            mod = int(p["mod"])
            law = AffineOrbitLaw(1, int(p["step"]) % mod, mod)
            return law.navigate(int(p["x0"]) % mod, coordinate).value
        if self.family == "lcg":
            mod = int(p["mod"])
            law = AffineOrbitLaw(int(p["a"]) % mod, int(p["c"]) % mod, mod)
            return law.navigate(int(p["x0"]) % mod, coordinate).value
        if self.family == "gf2":
            return lfsr_jump_gf2(p["coeffs"], p["init"], coordinate)
        if self.family == "poly":
            coeffs = p["newton_coeffs"]
            total = 0
            for k, ck in enumerate(coeffs):
                total += int(ck) * math.comb(coordinate, k)
            mod = p.get("mod")
            return total % int(mod) if mod is not None else total
        if self.family == "fgraph":
            mu = int(p["mu"])
            lam = int(p["lam"])
            orbit = p["orbit"]
            if coordinate < mu:
                return int(orbit[coordinate])
            return int(orbit[mu + ((coordinate - mu) % lam)])
        raise LawCertificationRefused(f"Unsupported law family {self.family!r}.")

    def locate_value(self, target: int) -> int | None:
        r"""Return smallest valid coordinate $t \ge 0$ with $x_t = \text{target}$, or raise if non-invertible."""
        if isinstance(target, bool) or not isinstance(target, int):
            raise ValueError("target must be an exact int.")
        p = self.params
        res: int | None = None
        if self.family == "int":
            res = locate_int_progression(int(p["x0"]), int(p["step"]), int(p["mod"]), target)
        elif self.family == "lcg":
            a, c, mod, x0 = int(p["a"]), int(p["c"]), int(p["mod"]), int(p["x0"])
            res = locate_lcg_2adic(a, c, mod, x0, target)
            if res is None:
                if math.gcd(a, mod) == 1 and mod <= 10_000_000:
                    res = locate_lcg_bsgs(a, c, mod, x0, target)
                elif mod <= 65_536:
                    # Exact bounded orbit scan on small non-coprime rings
                    cur = x0 % mod
                    for t in range(mod + 1):
                        if cur == target:
                            res = t
                            break
                        cur = (a * cur + c) % mod
                else:
                    raise LawCertificationRefused(
                        f"LCG inverse lookup refused: non-2adic/large modulus {mod} lacks O(log M) or bounded BSGS witness."
                    )
        elif self.family == "gf2":
            res = locate_gf2_bit(p["coeffs"], p["init"], target)
        elif self.family == "fgraph":
            orbit = p["orbit"]
            for idx, val in enumerate(orbit):
                if int(val) == target:
                    res = idx
                    break
        else:
            raise LawCertificationRefused(f"Family {self.family!r} does not admit closed-form inverse lookup.")

        if res is not None and self.verdict == "PREFIX-GROUND" and self.first_bad is not None and res >= self.first_bad:
            return None
        return res


def json_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): json_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(json_freeze(v) for v in value)
    return value


def json_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: json_thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [json_thaw(v) for v in value]
    return value


def certify_sequence_law(
    samples: Sequence[int],
    *,
    declared_modulus: int | None = None,
    allow_unbounded_poly: bool = False,
    prefer_gf2: bool = False,
    max_poly_degree: int = 6,
) -> CertifiedLaw:
    """Recover and strictly certify the lowest-complexity deterministic law explaining `samples`.

    Refuses with `LawCertificationRefused` whenever observations lack held-out predictive degrees
    of freedom or when modulus/domain is unidentifiable.
    """
    if len(samples) < 4:
        raise LawCertificationRefused("At least 4 observations are required to certify a generative law.")
    if declared_modulus is not None and (
        isinstance(declared_modulus, bool) or not isinstance(declared_modulus, int) or declared_modulus < 2
    ):
        raise LawCertificationRefused("declared_modulus must be an exact int >= 2 when supplied.")
    for idx, x in enumerate(samples):
        if isinstance(x, bool) or not isinstance(x, int):
            raise LawCertificationRefused(f"Observation at index {idx} must be an exact int.")
        if declared_modulus is not None and not (0 <= x < declared_modulus):
            raise LawCertificationRefused("Observation falls outside declared_modulus.")

    # 1. Optional GF(2) binary stream rung when samples are bits and prefer_gf2 or declared_modulus == 2
    if (prefer_gf2 or declared_modulus == 2) and all(x in (0, 1) for x in samples):
        coeffs, order = berlekamp_massey_gf2(samples)
        if order == 0:
            # Constant 0 sequence
            return CertifiedLaw("int", {"mod": 2, "step": 0, "x0": samples[0]}, order=1)
        if len(samples) >= 2 * order + 2:
            init = list(samples[:order])
            cand = CertifiedLaw("gf2", {"coeffs": list(coeffs), "init": init}, order=order)
            if all(cand.evaluate(t) == samples[t] for t in range(len(samples))):
                return cand

    # 2. Rung 1: Modular arithmetic progression (`int`)
    if all(x >= 0 for x in samples):
        prop = propose_modular_progression(samples, declared_modulus=declared_modulus)
        if prop.status is EpistemicStatus.CERTIFIED and prop.law is not None:
            return CertifiedLaw(
                "int",
                {
                    "mod": prop.law.modulus,
                    "step": prop.law.increment,
                    "x0": samples[0] % prop.law.modulus,
                },
                order=1,
            )

    # 3. Rung 2: Modular affine recurrence (`lcg`) when declared_modulus >= 3 and N >= 5
    if declared_modulus is not None and declared_modulus >= 3 and len(samples) >= 5:
        m = declared_modulus
        if any(not (0 <= x < m) for x in samples):
            raise LawCertificationRefused("Observation falls outside declared_modulus.")
        # Find consecutive differences (d_t, d_{t+1}) with gcd(d_t, m) == 1 to uniquely identify `a mod m`
        for idx in range(len(samples) - 2):
            d0 = (samples[idx + 1] - samples[idx]) % m
            d1 = (samples[idx + 2] - samples[idx + 1]) % m
            g = math.gcd(d0, m)
            if g == 1:
                a = (d1 * pow(d0, -1, m)) % m
                c = (samples[1] - a * samples[0]) % m
                cand = CertifiedLaw("lcg", {"a": a, "c": c, "mod": m, "x0": samples[0]}, order=1)
                if all(cand.evaluate(t) == samples[t] for t in range(len(samples))):
                    return cand

    # 4. Rung 3: GF(2) LFSR fallback for binary streams even if prefer_gf2 was not set
    if all(x in (0, 1) for x in samples):
        coeffs, order = berlekamp_massey_gf2(samples)
        if 1 <= order and len(samples) >= 2 * order + 2:
            cand = CertifiedLaw("gf2", {"coeffs": list(coeffs), "init": list(samples[:order])}, order=order)
            if all(cand.evaluate(t) == samples[t] for t in range(len(samples))):
                return cand

    # Precompute eventual periodic finite-state candidate (`fgraph`) so bounded low-period cycles
    # are never mis-certified as high-degree unbounded polynomials on short prefixes.
    fgraph_cand: CertifiedLaw | None = None
    fgraph_complexity: int | None = None
    next_map: dict[int, int] = {}
    deterministic_map = True
    for i in range(len(samples) - 1):
        u, v = samples[i], samples[i + 1]
        if u in next_map and next_map[u] != v:
            deterministic_map = False
            break
        next_map[u] = v
    if deterministic_map:
        first_seen: dict[int, int] = {}
        mu: int | None = None
        lam: int | None = None
        for t, val in enumerate(samples):
            if val in first_seen:
                mu = first_seen[val]
                lam = t - mu
                break
            first_seen[val] = t
        if mu is not None and lam is not None and lam >= 1 and len(samples) >= mu + 2 * lam:
            orbit = list(samples[: mu + lam])
            cand_fg = CertifiedLaw("fgraph", {"lam": lam, "mu": mu, "orbit": orbit}, order=lam)
            if all(cand_fg.evaluate(t) == samples[t] for t in range(len(samples))):
                fgraph_cand = cand_fg
                fgraph_complexity = mu + lam

    # 5. Rung 4: Exact Newton Binomial Polynomial (`poly`) over Z or Z/MZ
    if allow_unbounded_poly or declared_modulus is not None:
        max_deg = min(max_poly_degree, len(samples) - 3)
        if allow_unbounded_poly and declared_modulus is None:
            diffs = [list(samples)]
            while len(diffs[-1]) > 1 and len(diffs) <= max_poly_degree + 2:
                prev = diffs[-1]
                diffs.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
            for deg in range(1, max_deg + 1):
                rem_diffs = diffs[deg + 1]
                if len(rem_diffs) >= 2 and all(d == 0 for d in rem_diffs):
                    if fgraph_cand is not None and fgraph_complexity is not None and fgraph_complexity <= deg:
                        return fgraph_cand
                    newton_coeffs = [diffs[k][0] for k in range(deg + 1)]
                    cand = CertifiedLaw("poly", {"newton_coeffs": newton_coeffs}, order=deg)
                    if all(cand.evaluate(t) == samples[t] for t in range(len(samples))):
                        return cand
        elif declared_modulus is not None:
            m = declared_modulus
            diffs_mod = [[x % m for x in samples]]
            while len(diffs_mod[-1]) > 1 and len(diffs_mod) <= max_poly_degree + 2:
                prev = diffs_mod[-1]
                diffs_mod.append([(prev[i + 1] - prev[i]) % m for i in range(len(prev) - 1)])
            for deg in range(1, max_deg + 1):
                rem_diffs = diffs_mod[deg + 1]
                if len(rem_diffs) >= 2 and all(d == 0 for d in rem_diffs):
                    newton_coeffs = [diffs_mod[k][0] for k in range(deg + 1)]
                    cand = CertifiedLaw("poly", {"mod": m, "newton_coeffs": newton_coeffs}, order=deg)
                    if all(cand.evaluate(t) == samples[t] for t in range(len(samples))):
                        return cand

    # 6. Rung 5: Eventual periodic finite state map (`fgraph`)
    if fgraph_cand is not None:
        return fgraph_cand

    _ = canonical_json(list(samples[:8]))
    raise LawCertificationRefused(
        "Observations do not uniquely identify an admitted deterministic Ground law with positive held-out degrees of freedom."
    )
