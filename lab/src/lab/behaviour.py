"""The historical (no-tool) model: how a person without the tool picks the next withdrawal amount.

`Demand` fits the demand `g`, `Theta` holds `pi`, `s`, `w` and the demand version, and
`Historical`/`GridHistorical` combine them into `P(x | r)`; the notation is that of
:mod:`lab`.  Sampling is exact and consumes the injected `rng` in a fixed order.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, Decimal, localcontext
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random
    from collections.abc import Mapping, Sequence

# ==== demand ====


@dataclass(frozen=True)
class Theta:
    """The historical model's `pi` (withdraw all), `s` (matched tail), `w` (uniform) and demand version `g`."""

    pi: float
    s: float
    w: float
    g: str


@cache
def log_edges(max_value: int) -> tuple[int, ...]:
    """Decade bins: `ceil(10 ** (k / 10))` for `k = 0, 1, ...` up to `max_value`.

    Args:
        max_value: The largest amount the bins must cover.

    Returns:
        Increasing bin edges, the last one above `max_value`. Cached.
    """
    edges: list[int] = []
    with localcontext() as ctx:
        ctx.prec, k = 50, 0
        while True:
            value = int((Decimal(10) ** (Decimal(k) / 10)).to_integral_value(rounding=ROUND_CEILING))
            if not edges or value != edges[-1]:
                edges.append(value)
                if value > max_value:
                    break
            k += 1
    return tuple(edges)


def _bin_index(edges: Sequence[int], value: int) -> int | None:
    k = bisect_right(edges, value) - 1
    return k if 0 <= k < len(edges) - 1 else None


class Demand:
    """The demand `g`: an empirical withdrawal-amount distribution on `[1, max_deposit]`.

    An atom is an amount withdrawn at least twice in training (after the `lo` removal); it keeps
    its own probability.  Every other amount is counted in its decade bin (:func:`log_edges`), each
    bin from `step` to `max_deposit` gets an extra count of 1/2. A non-atom integer receives
    its bin count divided by `z` and the full bin width; atom positions retain their own mass.
    `z` is the total count used as the normaliser.
    """

    __slots__ = ("_bin_count", "_mass", "_pc_hi", "_pc_lo", "_widths", "_z", "atoms", "edges",
                 "max_deposit", "step", "version")

    def __init__(self, version: str, step: int, atoms: frozenset[int], edges: tuple[int, ...],
                 max_deposit: int, mass: dict[int, float], bin_count: dict[int, float],
                 pc_lo: int, pc_hi: int, z: float) -> None:
        self.version, self.step, self.atoms, self.edges = version, step, atoms, edges
        self.max_deposit, self._mass, self._bin_count = max_deposit, dict(mass), dict(bin_count)
        self._widths = {k: edges[k + 1] - edges[k] for k in range(len(edges) - 1)}
        self._pc_lo, self._pc_hi, self._z = pc_lo, pc_hi, z

    @classmethod
    def from_counts(cls, gross_counts: Mapping[int, int], deposit_counts: Mapping[int, int],
                    version: str, step: int, max_deposit: int) -> Demand:
        """Fit the demand from per-amount counts.

        Args:
            gross_counts: How many times each amount was withdrawn in training.
            deposit_counts: How many earlier deposits each amount had.
            version: `hi` counts every withdrawal; `lo` first removes one per earlier deposit.
            step: The demand grid `G`, in the asset's smallest unit.
            max_deposit: The largest balance the demand must cover.

        Returns:
            The fitted demand.
        """
        edges = log_edges(max(max(gross_counts, default=0), step, max_deposit))
        mass: dict[int, float] = {}
        bin_count: dict[int, float] = defaultdict(float)
        for value, count in gross_counts.items():
            residual = count if version == "hi" else count - deposit_counts.get(value, 0)
            if residual >= 2:
                mass[value] = float(residual)
            else:
                bin_count[_bin_index(edges, value)] += (1.0 if version == "hi"
                                                        else float(max(residual, 0)) if count >= 2 else 1.0)
        pc_lo, pc_hi = _bin_index(edges, step), _bin_index(edges, max(step, max_deposit))
        for k in range(pc_lo, pc_hi + 1):
            bin_count[k] += 0.5
        return cls(version, step, frozenset(mass), edges, max_deposit, mass, bin_count, pc_lo, pc_hi,
                   sum(mass.values()) + sum(bin_count.values()))

    @classmethod
    def build(cls, prior_grosses: Sequence[int], deposit_values: Sequence[int], version: str,
              step: int, max_deposit: int) -> Demand:
        """Fit the demand from one entry per training withdrawal and per deposit.

        Args:
            prior_grosses: The training withdrawal amounts, one entry per withdrawal.
            deposit_values: The deposit amounts, one entry per deposit.
            version: `hi` counts every withdrawal; `lo` first removes one per earlier deposit.
            step: The demand grid `G`, in the asset's smallest unit.
            max_deposit: The largest balance the demand must cover.

        Returns:
            The fitted demand.
        """
        counts: dict[int, int] = defaultdict(int)
        for value in prior_grosses:
            counts[value] += 1
        deposits: dict[int, int] = defaultdict(int)
        for value in deposit_values:
            deposits[value] += 1
        return cls.from_counts(counts, deposits, version, step, max_deposit)

    @property
    def z(self) -> float:
        """The total count, the normaliser of the demand."""
        return self._z

    def density(self, value: int) -> float:
        """The atom mass or non-atom bin density at `value`.

        Args:
            value: The amount, in the asset's smallest unit.

        Returns:
            An atom's own mass divided by `z`, or its bin's per-integer density otherwise.
            Zero for an empty demand, a non-positive `value` or a non-atom outside counted bins.
        """
        if self._z <= 0 or value <= 0:
            return 0.0
        if value in self._mass:
            return self._mass[value] / self._z
        k = _bin_index(self.edges, value)
        if k is None or k < self._pc_lo or k > self._pc_hi:
            return 0.0
        count = self._bin_count.get(k, 0.0)
        return 0.0 if count == 0.0 else count / (self._z * self._widths[k])

    def pmf(self, value: int, ell: int) -> float:
        """Probability of `value`: an atom's own, else the bin density times `ell` integers.

        Args:
            value: The amount, in the asset's smallest unit.
            ell: How many integers the bin density stands for (1 for a single amount).

        Returns:
            The demand probability, zero for a non-positive `value` or an empty demand.
        """
        if value <= 0 or self._z <= 0:
            return 0.0
        mass = self._mass.get(value)
        return mass / self._z if mass is not None else self.density(value) * ell


# ==== exact interval sums ====


def _tail_count(lo: int, hi: int, r: int, u: int) -> int:
    """Count `v` in `[lo, hi]` with `v < r`, `v % u == r % u` and `v % u != 0`."""
    if u <= 0 or r % u == 0:
        return 0
    hi2 = min(hi, r - 1)
    if hi2 < lo:
        return 0
    residue = r % u
    first = lo + ((residue - lo) % u)
    if first > hi2:
        return 0
    return (hi2 - first) // u + 1


def _band_g_sum(demand: Demand, atoms: tuple[int, ...], edges: tuple[int, ...],
                bands: Sequence[tuple[int, int]]) -> float:
    """Exact `sum_{v in bands} g(v)` with `atoms`/`edges` already sorted."""
    if not bands:
        return 0.0
    total = 0.0
    for lo, hi in bands:
        if hi < 1:
            continue
        lo = max(lo, 1)
        if hi < lo:
            continue
        i = bisect_left(atoms, lo)
        j = bisect_right(atoms, hi)
        for a in atoms[i:j]:
            total += demand.pmf(a, 1)
        k = max(bisect_right(edges, lo) - 1, 0)
        while k < len(edges) - 1:
            seg_lo = max(lo, edges[k])
            seg_hi = min(hi, edges[k + 1] - 1)
            if seg_hi >= seg_lo:
                points = seg_hi - seg_lo + 1
                atom_pts = bisect_right(atoms, seg_hi) - bisect_left(atoms, seg_lo)
                non_atom = points - atom_pts
                if non_atom:
                    rep = seg_lo
                    while rep <= seg_hi and bisect_left(atoms, rep) < bisect_right(atoms, rep):
                        rep += 1
                    total += demand.density(rep) * non_atom
            if edges[k + 1] > hi:
                break
            k += 1
    return total


def demand_band_sum(demand: Demand, bands: Sequence[tuple[int, int]]) -> float:
    """Exact `sum_{v in bands} g(v)` over the integers of `bands`.

    Args:
        demand: The fitted demand.
        bands: Inclusive `(lo, hi)` ranges in any order; each is clipped to start at 1.

    Returns:
        The summed demand probability.
    """
    return _band_g_sum(demand, tuple(sorted(demand.atoms)), tuple(demand.edges), bands)


def _band_sum_likelihood(bands: Sequence[tuple[int, int]], r: int, theta: Theta, demand: Demand,
                         unit: int) -> float:
    """`sum_{v in bands} L(v | r)` under `theta`, by interval and atom counts."""
    if r <= 0:
        return 0.0
    base = r // unit
    total = 0.0
    for lo, hi in bands:
        lo = max(lo, 1)
        if hi < lo:
            continue
        point = lo <= r <= hi
        clipped_hi = min(hi, r)
        if clipped_hi >= lo:
            n = clipped_hi - lo + 1
            gsum = demand_band_sum(demand, [(lo, clipped_hi)])
            prop = (1.0 - theta.w) * gsum + theta.w * n / r
            total += (1.0 - theta.pi) * (1.0 - theta.s) * prop
            if theta.s and base > 0:
                total += (1.0 - theta.pi) * theta.s * _tail_count(lo, clipped_hi, r, unit) / base
        if point:
            total += theta.pi
    return total


def channel_mass(r: int, theta: Theta, demand: Demand, unit: int) -> float:
    """`Z_theta(r) = sum_{x=1}^{r} L_theta(x | r)`, the channel mass under `theta`.

    Args:
        r: The remaining balance, in the asset's smallest unit.
        theta: The historical model's `pi`, `s` and `w`.
        demand: The demand `g`.
        unit: The tail unit of `s`.

    Returns:
        The unnormalised mass, never below zero.
    """
    return max(float(_band_sum_likelihood([(1, r)], r, theta, demand, unit)), 0.0)


# ==== component sampler ====


def _demand_segments(demand: Demand, r: int) -> list[tuple[float, str, object]]:
    """Restrict `g` to `[1, r]` as `(mass, kind, payload)` components (atoms first)."""
    if r < 1:
        return []
    z = float(demand.z)
    if z <= 0.0:
        return []
    atoms = tuple(sorted(int(a) for a in demand.atoms))
    segments: list[tuple[float, str, object]] = [
        (float(demand.pmf(a, 1)), "atom", a) for a in atoms if 1 <= a <= r
    ]
    edges = tuple(int(e) for e in demand.edges)
    atom_set = set(atoms)
    for k in range(len(edges) - 1):
        lo = max(edges[k], 1)
        hi = min(edges[k + 1] - 1, r)
        if lo > hi:
            continue
        rep = None
        v = lo
        while v <= hi:
            if v not in atom_set:
                rep = v
                break
            v += 1
        if rep is None:
            continue
        density = float(demand.density(rep))
        if density <= 0.0:
            continue
        atoms_here = tuple(a for a in atoms if lo <= a <= hi)
        non_atom = (hi - lo + 1) - len(atoms_here)
        if non_atom <= 0:
            continue
        segments.append((density * non_atom, "bin", (lo, hi, atoms_here)))
    return segments


def _draw_demand_point(rng: random.Random, segments: list[tuple[float, str, object]],
                       total: float) -> int:
    u = rng.random() * total
    acc = 0.0
    for mass, kind, payload in segments:
        acc += mass
        if u < acc:
            if kind == "atom":
                return int(payload)
            lo, hi, atoms_here = payload
            atom_set = set(atoms_here)
            while True:
                v = lo + rng.randrange(hi - lo + 1)
                if v not in atom_set:
                    return v
    mass, kind, payload = segments[-1]
    return int(payload) if kind == "atom" else payload[0]


def _sample_tail(r: int, unit: int, rng: random.Random) -> int:
    """A uniform matched-tail amount: `x in [1, r-1]` with `x % unit == r % unit != 0`."""
    return r % unit + rng.randrange(r // unit) * unit


def sample_input(r: int, theta: Theta, demand: Demand, unit: int, rng: random.Random) -> int:
    """Draw `x in [1, r]` exactly from the `plain` model's `P(x | r)`.

    Args:
        r: The remaining balance, at least 1.
        theta: The historical model's `pi`, `s` and `w`.
        demand: The demand `g`.
        unit: The tail unit of `s`.
        rng: The source of randomness; consumed in a fixed component order.

    Returns:
        The drawn need `x`.
    """
    pi, s, w, _g = theta.pi, theta.s, theta.w, theta.g
    residue = r % unit
    base = r // unit
    segments = _demand_segments(demand, r)
    demand_mass = math.fsum(mass for mass, _k, _p in segments)
    w_atom = pi
    w_tail = (1.0 - pi) * s if (residue != 0 and base > 0) else 0.0
    w_demand = (1.0 - pi) * (1.0 - s) * (1.0 - w) * demand_mass
    w_uniform = (1.0 - pi) * (1.0 - s) * w
    z = w_atom + w_tail + w_demand + w_uniform
    u = rng.random()
    comps = ((w_atom, "atom"), (w_tail, "tail"), (w_demand, "demand"), (w_uniform, "uniform"))
    chosen = max(comps, key=lambda c: c[0])[1]
    acc = 0.0
    for mass, name in comps:
        acc += mass / z
        if u < acc:
            chosen = name
            break
    if chosen == "atom":
        return r
    if chosen == "tail":
        return _sample_tail(r, unit, rng)
    if chosen == "uniform":
        return 1 + rng.randrange(r)
    if demand_mass <= 0.0 or not segments:
        return r
    return _draw_demand_point(rng, segments, demand_mass)


# ==== historical models ====


@dataclass(frozen=True)
class Historical:
    """The `plain` historical model: `P(x | r) = L(x | r) / Z(r)` over every integer of `[1, r]`."""

    theta: Theta
    demand: Demand
    unit: int
    _z: dict = field(default_factory=dict, compare=False, repr=False)

    def mass(self, y: int, r: int) -> float:
        """`P(next withdrawal is y | remaining r)` under the normalised historical model.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.

        Returns:
            The probability, zero when `r < 1` or `y > r`. The channel mass is cached per `r`.
        """
        if r < 1 or y > r:
            return 0.0
        if r not in self._z:
            self._z[r] = channel_mass(r, self.theta, self.demand, self.unit)
        return float(_band_sum_likelihood([(y, y)], r, self.theta, self.demand, self.unit)) / self._z[r]

    def active(self, r: int) -> bool:
        return r >= 1

    def sample(self, r: int, rng: random.Random) -> tuple[int, int]:
        """Draw `x` from `P(x | r)` and return `(x, y)` with `y = x`.

        Args:
            r: The remaining balance.
            rng: The source of randomness, consumed in a fixed order.

        Returns:
            `(x, y)`, both the need, since the person does not use the tool.
        """
        x = sample_input(r, self.theta, self.demand, self.unit, rng)
        return x, x


def _mult_count(lo: int, hi: int, grid: int) -> int:
    """Number of multiples of `grid` in `[lo, hi]`."""
    return hi // grid - (lo - 1) // grid if hi >= lo else 0


def _atom_multiples(atoms: tuple[int, ...], lo: int, hi: int, grid: int) -> list[int]:
    """Grid multiples in `[lo, hi]` that are demand atoms (they carry the atom pmf)."""
    return [a for a in atoms[bisect_left(atoms, lo):bisect_right(atoms, hi)] if a % grid == 0]


def _nth_multiple(first: int, grid: int, index: int, atom_multiples: list[int]) -> int:
    """The `index`-th (0-based) multiple of `grid` from `first` that is not an atom."""
    position = index
    while True:
        value = first + position * grid
        skipped = bisect_right(atom_multiples, value)
        if skipped + index == position:
            return value
        position = skipped + index


@dataclass(frozen=True)
class GridHistorical:
    """The `grid` historical model: the demand and the uniform part sit on the grid `G = step`.

    Atoms keep their probability; a bin's count is spread over its non-atom multiples of `G` and
    is zero elsewhere, and the uniform part picks a multiple of `G` in `[1, r]`.
    """

    theta: Theta
    demand: Demand
    unit: int
    _z: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_atoms", tuple(sorted(self.demand.atoms)))

    def _grid_density(self, y: int) -> float:
        """The demand on the grid at `y`: an atom's probability, else its bin's share if `G` divides `y`."""
        demand = self.demand
        if y in demand.atoms:
            return float(demand.pmf(y, 1))
        grid = demand.step
        edges = demand.edges
        if y % grid:
            return 0.0
        k = bisect_right(edges, y) - 1
        if k < 0 or k >= len(edges) - 1 or k < demand._pc_lo or k > demand._pc_hi:
            return 0.0
        count = demand._bin_count.get(k, 0.0)
        points = _mult_count(edges[k], edges[k + 1] - 1, grid)
        return 0.0 if count == 0.0 or points == 0 else count / (float(demand.z) * points)

    def _channel_sum(self, r: int) -> float:
        """`Z(r) = sum_{v=1..r} L(v | r)` by atom and per-bin grid arithmetic."""
        demand, grid = self.demand, self.demand.step
        pi, s, w = self.theta.pi, self.theta.s, self.theta.w
        edges = demand.edges
        points = float(demand.z)
        atoms = self._atoms
        gsum = math.fsum(float(demand.pmf(a, 1)) for a in atoms[:bisect_right(atoms, r)])
        for k in range(len(edges) - 1):
            count = demand._bin_count.get(k, 0.0)
            if count == 0.0 or k < demand._pc_lo or k > demand._pc_hi:
                continue
            span = _mult_count(edges[k], edges[k + 1] - 1, grid)
            lo, hi = max(edges[k], 1), min(edges[k + 1] - 1, r)
            if span == 0 or hi < lo:
                continue
            non_atom = _mult_count(lo, hi, grid) - len(_atom_multiples(atoms, lo, hi, grid))
            gsum += non_atom * count / (points * span)
        unit = self.unit
        base = r // unit
        tail = _tail_count(1, r, r, unit) / base if base > 0 else 0.0
        return pi + (1.0 - pi) * ((1.0 - s) * ((1.0 - w) * gsum + w * (1.0 if r >= grid else 0.0))
                                  + s * tail)

    def mass(self, y: int, r: int) -> float:
        """`P(next withdrawal is y | remaining r)` under the grid-normalised historical model.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.

        Returns:
            The probability, zero when `r < 1` or `y > r`. The channel mass is cached per `r`.
        """
        if r < 1 or y > r:
            return 0.0
        pi, s, w = self.theta.pi, self.theta.s, self.theta.w
        grid = self.demand.step
        unit = self.unit
        base = r // unit
        grid_base = r // grid
        grid_term = 1.0 / grid_base if grid_base > 0 and y % grid == 0 else 0.0
        tail = (1.0 / base if base > 0 and y < r and y % unit != 0 and y % unit == r % unit
                else 0.0)
        likelihood = ((1.0 if y == r else 0.0) * pi + (1.0 - pi) * (
            (1.0 - s) * ((1.0 - w) * self._grid_density(y) + w * grid_term) + s * tail))
        z = self._z.get(r)
        if z is None:
            z = self._z[r] = self._channel_sum(r)
        return likelihood / z

    def active(self, r: int) -> bool:
        return r >= 1

    def sample(self, r: int, rng: random.Random) -> tuple[int, int]:
        """Draw `x` exactly from `P(x | r)` and return `(x, y)` with `y = x`.

        Args:
            r: The remaining balance.
            rng: One draw selects a component. A tail, bin or grid component then draws its
                value; self and atom components need no further randomness.

        Returns:
            `(x, y)`, both the need, since the person does not use the tool.
        """
        demand, grid = self.demand, self.demand.step
        pi, s, w = self.theta.pi, self.theta.s, self.theta.w
        edges, points = demand.edges, float(demand.z)
        atoms = self._atoms
        unit = self.unit
        scale = (1.0 - pi) * (1.0 - s) * (1.0 - w)
        components: list[tuple[float, str, object]] = [(pi, "self", None)]
        base = r // unit
        if base > 0:
            components.append(((1.0 - pi) * s * _tail_count(1, r, r, unit) / base, "tail", None))
        components.extend((scale * float(demand.pmf(a, 1)), "atom", a)
                          for a in atoms[:bisect_right(atoms, r)])
        for k in range(len(edges) - 1):
            count = demand._bin_count.get(k, 0.0)
            if count == 0.0 or k < demand._pc_lo or k > demand._pc_hi:
                continue
            span = _mult_count(edges[k], edges[k + 1] - 1, grid)
            lo, hi = max(edges[k], 1), min(edges[k + 1] - 1, r)
            if span == 0 or hi < lo:
                continue
            multiples = _atom_multiples(atoms, lo, hi, grid)
            non_atom = _mult_count(lo, hi, grid) - len(multiples)
            if non_atom > 0:
                components.append((scale * non_atom * count / (points * span), "bin",
                                   (lo, hi, multiples)))
        grid_base = r // grid
        if grid_base > 0:
            components.append(((1.0 - pi) * (1.0 - s) * w, "grid", grid_base))
        cumulative, acc = [], 0.0
        for weight, _kind, _payload in components:
            acc += weight
            cumulative.append(acc)
        _weight, kind, payload = components[bisect_right(cumulative, rng.random() * acc)]
        if kind == "self":
            x = r
        elif kind == "tail":
            x = _sample_tail(r, unit, rng)
        elif kind == "atom":
            x = int(payload)
        elif kind == "bin":
            lo, hi, multiples = payload
            first = -(-lo // grid) * grid
            index = rng.randrange(_mult_count(lo, hi, grid) - len(multiples))
            x = _nth_multiple(first, grid, index, multiples)
        else:
            x = grid * rng.randrange(1, payload + 1)
        return x, x
