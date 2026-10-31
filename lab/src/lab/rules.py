"""The tool rules, the adopter behaviour and the exact probability of an adopter's next amount.

The notation is that of :mod:`lab`.  An adopter's `P(y | r)` sums the walk probability of `y`
over the needs `x` on the `quantum` grid.  Needs whose first rung `c0` lies in one piece share
the rungs above `c0`, so the sum runs over pieces and the maximums at which the walk changes, not
over every `x`.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING

from withdrawchan import P, Q, decimal_rungs, walk

if TYPE_CHECKING:
    from random import Random

__all__ = [
    "Adopter",
    "DecimalLadder",
    "Need",
    "PopularLadder",
    "model",
    "popular_counts",
]

#: `rng(n)`: a uniform integer in `[0, n)`.
Rng = Callable[[int], int]

#: A piece `(lo, hi, tail)`: first rungs `c0` (multiples of `d`) in `[lo, hi]` whose rungs above
#: `c0`, before the maximum cuts them, are all `tail`.
Piece = tuple[int, int, tuple[int, ...]]


def _ceil_to(v: int, unit: int) -> int:
    return -(-v // unit) * unit


def _floor_to(v: int, unit: int) -> int:
    return v // unit * unit


def _bucket(top: int, d: int) -> int:
    """The bound pieces are built to: the power of ten above `top`, rounded up to `d`."""
    return _ceil_to(10 ** len(str(top)), d)


# ==== rules ====


@dataclass(frozen=True)
class Need:
    """Recommend exactly the need: `y = x`."""

    def output(self, x: int, m: int, rng: Rng) -> int:
        """Return the recommended amount, which is the need itself.

        Args:
            x: The need, the least amount to withdraw.
            m: The maximum, the largest amount to withdraw; unused.
            rng: A callable returning a uniform integer in `[0, n)`; unused.

        Returns:
            `x`.
        """
        return x

    def mass(self, y: int, r: int, quantum: int, m_model: str) -> float:
        """`P(y | r)` with `y = x`: `1/n` on each multiple of `quantum`, `n = r // quantum`.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.
            quantum: The adopter need grid, also the spacing of the reachable `x`.
            m_model: The maximum model name; unused.

        Returns:
            The probability.
        """
        return 1.0 / (r // quantum) if y % quantum == 0 else 0.0


@dataclass(frozen=True)
class DecimalLadder:
    """Walk up the decimal rungs: `c0 = ceil_d(x)`, `c0` rounded up to one significant digit, then
    each higher power of ten (:func:`withdrawchan.decimal_rungs`)."""

    d: int
    q: Fraction = Q
    p: Fraction = P
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def rungs(self, c0: int, m: int) -> list[int]:
        return decimal_rungs(c0, self.d, m)

    def pieces(self, top: int) -> tuple[Piece, ...]:
        """Pieces partitioning the `c0` multiples of `d` in `[d, top]` with one common tail.

        Args:
            top: The upper bound of the partition.

        Returns:
            Ascending disjoint pieces; every `c0` multiple of `d` in `[d, top]` lies in exactly one,
            and the rungs above it are the piece's tail up to the maximum. Cached per `top`.
        """
        key = ("pieces", top)
        cached = self._cache.get(key)
        if cached is None:
            cached = _decimal_pieces(self.d, top)
            self._cache[key] = cached
        return cached

    def output(self, x: int, m: int, rng: Rng) -> int:
        """Draw one recommended amount from the decimal ladder.

        Args:
            x: The need, the least amount to withdraw.
            m: The maximum, the largest amount to withdraw.
            rng: A callable returning a uniform integer in `[0, n)`; consumed in walk order.

        Returns:
            The recommended amount.
        """
        return walk(self.rungs(_ceil_to(x, self.d), m), m, d=self.d, q=self.q, p=self.p, rng=rng)

    def mass(self, y: int, r: int, quantum: int, m_model: str) -> float:
        """`P(y | r)` for the decimal ladder, summed over the quantum grid.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.
            quantum: The adopter need grid.
            m_model: The maximum model, `double` or `balance`.

        Returns:
            The probability.
        """
        return _ladder_mass(self, y, r, quantum, m_model)


@dataclass(frozen=True)
class PopularLadder:
    """Walk up popular amounts: `amounts` were each withdrawn at least twice in training, in
    increasing order, and `counts` says how often; a rung above `c0` must be more popular than the
    rung before."""

    amounts: tuple[int, ...]
    counts: tuple[int, ...]
    d: int
    q: Fraction = Q
    p: Fraction = P
    _cache: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_nxt", _next_greater(self.counts))

    def rungs(self, c0: int, m: int) -> list[int]:
        """`c0`, the first popular amount above it, then each later one with a larger count than the last rung.

        Args:
            c0: The first rung, `x` rounded up to `d`.
            m: The maximum, the largest amount to withdraw.

        Returns:
            Increasing rungs: `c0` and the popular amounts above it with rising counts, up to `m`.
        """
        i = bisect_left(self.amounts, c0)
        if i < len(self.amounts) and self.amounts[i] == c0:
            i = self._nxt[i]
        rungs = [c0]
        while i < len(self.amounts) and self.amounts[i] <= m:
            rungs.append(self.amounts[i])
            i = self._nxt[i]
        return rungs

    def pieces(self, top: int) -> tuple[Piece, ...]:
        """Pieces partitioning the `c0` multiples of `d` in `[d, top]` with one common tail.

        Args:
            top: The upper bound of the partition.

        Returns:
            Ascending disjoint pieces; every `c0` multiple of `d` in `[d, top]` lies in exactly one,
            and the rungs above it are the piece's tail up to the maximum. Cached per `top`.
        """
        key = ("pieces", top)
        cached = self._cache.get(key)
        if cached is None:
            cached = _popular_pieces(self.amounts, self._nxt, self.d, top)
            self._cache[key] = cached
        return cached

    def output(self, x: int, m: int, rng: Rng) -> int:
        """Draw one recommended amount from the popular ladder.

        Args:
            x: The need, the least amount to withdraw.
            m: The maximum, the largest amount to withdraw.
            rng: A callable returning a uniform integer in `[0, n)`; consumed in walk order.

        Returns:
            The recommended amount.
        """
        return walk(self.rungs(_ceil_to(x, self.d), m), m, d=self.d, q=self.q, p=self.p, rng=rng)

    def mass(self, y: int, r: int, quantum: int, m_model: str) -> float:
        """`P(y | r)` for the popular ladder, summed over the quantum grid.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.
            quantum: The adopter need grid.
            m_model: The maximum model, `double` or `balance`.

        Returns:
            The probability.
        """
        return _ladder_mass(self, y, r, quantum, m_model)


def _next_greater(weights: tuple[int, ...]) -> tuple[int, ...]:
    """Index of the first later weight strictly above each weight (`len` when there is none)."""
    out = [len(weights)] * len(weights)
    stack: list[int] = []
    for i, w in enumerate(weights):
        while stack and weights[stack[-1]] < w:
            out[stack.pop()] = i
        stack.append(i)
    return tuple(out)


def popular_counts(gross_amounts: Iterable[int], min_count: int = 2) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`(amounts, counts)` of the historical amounts seen at least `min_count` times, sorted.

    Args:
        gross_amounts: The training withdrawal amounts, one entry per withdrawal.
        min_count: The smallest count an amount must reach to be kept.

    Returns:
        `amounts` in increasing order and, in the same order, their counts.
    """
    counts = Counter(gross_amounts)
    amounts = tuple(sorted(a for a, c in counts.items() if c >= min_count))
    return amounts, tuple(counts[a] for a in amounts)


# ==== pieces ====


def _decimal_pieces(d: int, top: int) -> tuple[Piece, ...]:
    """Piece the `c0` multiples of `d` in `[d, top]` by their number of digits and one-digit ceiling."""
    sig: set[int] = set()
    p10 = 1
    while p10 <= top:
        for j in range(1, 10):
            if j * p10 <= top:
                sig.add(j * p10)
        p10 *= 10
    cuts = sorted(sig)
    spans = [(s, s) for s in cuts]
    spans += [(a + 1, b - 1) for a, b in zip(cuts, cuts[1:], strict=False)]
    spans += [(cuts[-1] + 1, top)]
    out: list[Piece] = []
    for lo0, hi0 in spans:
        lo = _ceil_to(lo0, d)
        hi = min(_floor_to(hi0, d), top)
        if lo > hi or lo > top:
            continue
        e = len(str(lo)) - 1
        one = _ceil_to(lo, 10**e)
        powers: list[int] = []
        v = 10 ** (e + 1)
        while v <= 2 * top:
            powers.append(v)
            v *= 10
        tail = tuple(sorted({t for t in (one, *powers) if lo < t}))
        out.append((lo, hi, tail))
    return tuple(sorted(out))


def _popular_pieces(amounts: tuple[int, ...], nxt: tuple[int, ...], d: int, top: int) -> tuple[Piece, ...]:
    """Piece the `c0` multiples of `d` in `[d, top]` where the first popular amount above `c0` changes."""
    starts: set[int] = {d}
    for a in amounts:
        for t in (_ceil_to(a, d), a, a + d) if a % d == 0 else (_ceil_to(a, d),):
            if d < t <= top:
                starts.add(t)
    ordered = sorted(starts)
    out: list[Piece] = []
    for i, lo in enumerate(ordered):
        hi = min(ordered[i + 1] - d, top) if i + 1 < len(ordered) else top
        if lo > hi:
            continue
        j = bisect_left(amounts, lo)
        if j < len(amounts) and amounts[j] == lo:
            j = nxt[j]
        tail: list[int] = []
        while j < len(amounts) and amounts[j] <= 2 * top:
            tail.append(amounts[j])
            j = nxt[j]
        out.append((lo, hi, tuple(tail)))
    return tuple(sorted(out))


# ==== exact ladder mass ====


@dataclass(frozen=True)
class _PiecePlan:
    """One piece that can produce `y`.

    `lo`/`hi` bound its `c0` values below `y`; `bounds` are the needs `x` from which the
    `double` maximum `ceil_d(2x)` reaches `y` or a rung; from maximum `bps[j]` on, the walk
    probability of `y` is `nums[j]` over the plan's `den`.
    """

    lo: int
    hi: int
    tail: tuple[int, ...]
    bounds: tuple[int, ...]
    bps: tuple[int, ...]
    nums: tuple[int, ...]


@dataclass(frozen=True)
class _MassPlan:
    """The part of `P(y | r)` that does not depend on `r`, for one rule, `y`, bucket and maximum model.

    Probabilities are integer numerators over `den = qden**(depth+1) * pden * d`, where
    `q = qnum/qden`, `p = pnum/pden` and `depth` is the longest tail.  The `first_*` fields
    cover the needs whose first rung is `floor_d(y)`: under `double` they are split by maximum into
    `first_splits` (maximum, need count), under `balance` they are `first_count` needs.
    """

    y: int
    d: int
    quantum: int
    m_model: str
    den: int
    ceil_y: int
    fallback_ok: bool
    first_present: bool
    first_splits: tuple[tuple[int, int], ...]
    first_count: int
    first_bps: tuple[int, ...]
    first_nums: tuple[int, ...]
    pieces: tuple[_PiecePlan, ...]


def _factor(y: int, d: int, pnum: int, pden: int, v: int, m: int, *, has_p: bool) -> int:
    """Numerator over `pden * d` of the probability that stopping at rung `v` gives `y` at maximum `m`."""
    top = min(v + d - 1, m) if has_p else v
    if y == v:
        return pden * d - pnum * (top - v)
    if v < y <= top:
        return pnum
    return 0


def _walk_num(c0: int, tail: tuple[int, ...], m: int, *, y: int, d: int, qstay: int, pnum: int,
              pden: int, has_p: bool, depth: int, qnum_pow: tuple[int, ...], qden_pow: tuple[int, ...]) -> int:
    """The probability of `y` under `withdrawchan.walk_distribution([c0, *(t for t in tail if t <= m)], m)`,
    as a numerator over the plan's `den`.

    `qstay = qden - qnum` is the numerator of `1 - q`; `qnum_pow`/`qden_pow` are powers of
    `qnum`/`qden`.
    """
    k = bisect_right(tail, m)
    coef = _factor(y, d, pnum, pden, c0, m, has_p=has_p)
    total = qnum_pow[0] * coef * qden_pow[depth + 1] if k == 0 else qnum_pow[0] * qstay * coef * qden_pow[depth]
    for i in range(k):
        coef = _factor(y, d, pnum, pden, tail[i], m, has_p=has_p)
        j = i + 1
        total += (qnum_pow[j] * coef * qden_pow[depth + 1 - j] if i == k - 1
                  else qnum_pow[j] * qstay * coef * qden_pow[depth - j])
    return total


def _step_table(c0: int, tail: tuple[int, ...], *, y: int, d: int, qstay: int, pnum: int, pden: int,
                has_p: bool, depth: int, qnum_pow: tuple[int, ...], qden_pow: tuple[int, ...],
                ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """`(bps, nums)`: `_walk_num` at each maximum `m` where the mass can change, as `m` grows."""
    bps = tuple(sorted({_ceil_to(v, d) for v in (c0, *tail)}
                       | {_ceil_to(v + d - 1, d) for v in (c0, *tail)}
                       | {_ceil_to(y, d)}))
    nums = tuple(_walk_num(c0, tail, m, y=y, d=d, qstay=qstay, pnum=pnum, pden=pden, has_p=has_p,
                           depth=depth, qnum_pow=qnum_pow, qden_pow=qden_pow) for m in bps)
    return bps, nums


def _lookup(bps: tuple[int, ...], nums: tuple[int, ...], m: int) -> int:
    """The step numerator in force at maximum `m`."""
    return nums[bisect_right(bps, m) - 1]


@lru_cache(maxsize=4096)
def _plan(rule: DecimalLadder | PopularLadder, y: int, bucket: int, quantum: int,
          m_model: str) -> _MassPlan:
    """The :class:`_MassPlan` of `y` for pieces built up to `bucket`; cached, bounded."""
    d = rule.d
    q, p = rule.q, rule.p
    qnum, qden = q.numerator, q.denominator
    pnum, pden = p.numerator, p.denominator
    has_p = pnum != 0
    qstay = qden - qnum
    lo_w = y - (d if has_p else 1) + 1
    c0_a = _floor_to(y, d)
    pieces_all = rule.pieces(bucket)
    first_chain: tuple[int, ...] = ()
    if c0_a >= d:
        for lo, hi, tail in pieces_all:
            if lo <= c0_a <= hi:
                first_chain = tail
    floor_w = _floor_to(lo_w - 1, d)
    clipped: list[tuple[int, int, tuple[int, ...]]] = []
    depth = len(first_chain)
    for lo, hi0, tail in pieces_all:
        hi = min(hi0, floor_w)
        if lo > hi or not any(lo_w <= t <= y for t in tail):
            continue
        clipped.append((lo, hi, tail))
        depth = max(depth, len(tail))
    qnum_pow = tuple(qnum**j for j in range(depth + 2))
    qden_pow = tuple(qden**j for j in range(depth + 2))
    pieces = tuple(_PiecePlan(lo, hi, tail,
                              tuple(sorted({(_ceil_to(t, d) - d) // 2 + 1
                                            for t in (y, y + d - 1, *tail)})),
                              *_step_table(lo, tail, y=y, d=d, qstay=qstay, pnum=pnum, pden=pden, has_p=has_p,
                                           depth=depth, qnum_pow=qnum_pow, qden_pow=qden_pow))
                   for lo, hi, tail in clipped)
    first_present = c0_a >= d and c0_a >= lo_w
    first_splits: tuple[tuple[int, int], ...] = ()
    first_count = 0
    first_bps: tuple[int, ...] = ()
    first_nums: tuple[int, ...] = ()
    if first_present:
        mul = c0_a // d
        flo = quantum * (((mul - 1) * d) // quantum + 1)
        fhi = quantum * (c0_a // quantum)
        if flo > fhi:
            first_present = False
        else:
            first_bps, first_nums = _step_table(c0_a, first_chain, y=y, d=d, qstay=qstay, pnum=pnum, pden=pden,
                                                has_p=has_p, depth=depth, qnum_pow=qnum_pow, qden_pow=qden_pow)
            if m_model == "balance":
                first_count = (fhi - flo) // quantum + 1
            else:
                cut = ((2 * mul - 1) * d) // 2
                first_splits = tuple((bn * d, (shi - slo) // quantum + 1)
                                     for slo, shi, bn in ((flo, min(fhi, cut), 2 * mul - 1),
                                                          (max(flo, cut + 1), fhi, 2 * mul))
                                     if slo <= shi)
    return _MassPlan(y=y, d=d, quantum=quantum, m_model=m_model, den=qden ** (depth + 1) * pden * d,
                     ceil_y=_ceil_to(y, d), fallback_ok=y % quantum == 0 and quantum <= y,
                     first_present=first_present, first_splits=first_splits, first_count=first_count,
                     first_bps=first_bps, first_nums=first_nums, pieces=pieces)


def _piece_num(plan: _MassPlan, piece: _PiecePlan, r: int, m_bal: int) -> int:
    """The needs of `piece` at balance `r`: their summed probability of `y` over `plan.den`.

    `m_bal = floor_d(r)` is the `balance` maximum and the upper limit of the `double` maximum.
    """
    d = plan.d
    xa = max(plan.quantum, piece.lo - d + 1)
    if plan.m_model == "balance":
        xb = min(piece.hi, r, m_bal)
        if xa > xb:
            return 0
        count = xb // plan.quantum - (xa - 1) // plan.quantum
        return count * _lookup(piece.bps, piece.nums, m_bal)
    xb = min(piece.hi, r)
    if xa > xb:
        return 0
    clamp = m_bal // 2 + 1
    stops = [xa]
    if xa < clamp <= xb:
        stops.append(clamp)
    stops += [x0 for x0 in piece.bounds if xa < x0 <= xb]
    stops.sort()
    stops.append(xb + 1)
    total = 0
    for a0, b0 in zip(stops, stops[1:], strict=False):
        xr = _ceil_to(a0, plan.quantum)
        if xr > b0 - 1:
            continue
        count = (b0 - 1) // plan.quantum - (xr - 1) // plan.quantum
        total += count * _lookup(piece.bps, piece.nums, min(_ceil_to(2 * xr, d), m_bal))
    return total


def _ladder_fraction(rule: DecimalLadder | PopularLadder, y: int, r: int, quantum: int,
                     m_model: str) -> Fraction:
    """Exact `P(y | r)` of an adopter using `rule`; the cost does not grow with `r / d`."""
    n = r // quantum
    if n < 1 or y > r:
        return Fraction(0)
    d = rule.d
    plan = _plan(rule, y, _bucket(_ceil_to(r, d), d), quantum, m_model)
    m_bal = _floor_to(r, d)
    total = 0
    if plan.first_present:
        if m_model == "balance":
            total += plan.first_count * _lookup(plan.first_bps, plan.first_nums, m_bal)
        else:
            for cap, count in plan.first_splits:
                total += count * _lookup(plan.first_bps, plan.first_nums, min(cap, m_bal))
    for piece in plan.pieces:
        total += _piece_num(plan, piece, r, m_bal)
    if plan.fallback_ok and m_bal < plan.ceil_y:
        total += plan.den
    return Fraction(total, plan.den * n)


def _ladder_mass(rule: DecimalLadder | PopularLadder, y: int, r: int, quantum: int,
                 m_model: str) -> float:
    return float(_ladder_fraction(rule, y, r, quantum, m_model))


# ==== adopter behaviour ====


@dataclass(frozen=True)
class Adopter:
    """An adopter, `KEEP` or (`drain=True`) `DRAIN`: applies `rule` with the `m_model` maximum while
    `r >= threshold`."""

    rule: Need | DecimalLadder | PopularLadder
    threshold: int
    quantum: int
    d: int
    drain: bool
    m_model: str

    def active(self, r: int) -> bool:
        """Whether the adopter still uses the tool at remaining balance `r`.

        A `KEEP` adopter stops below the threshold; a `DRAIN` adopter stays active to withdraw the
        whole remainder at once.
        """
        return r >= self.threshold or (self.drain and r >= 1)

    def output(self, x: int, r: int, rng: Rng) -> int:
        """The recommended amount for need `x` and remaining balance `r`.

        The maximum is `min(ceil_d(2x), floor_d(r))` under `double` and `floor_d(r)` under
        `balance`; when the maximum is below `ceil_d(x)` the tool cannot be used and the output is
        `x`.

        Args:
            x: The need, the least amount to withdraw.
            r: The remaining balance.
            rng: A callable returning a uniform integer in `[0, n)`; consumed only when the tool is
                used.

        Returns:
            The recommended amount.
        """
        m = _floor_to(r, self.d)
        if self.m_model == "double":
            m = min(_ceil_to(2 * x, self.d), m)
        if m < _ceil_to(x, self.d):
            return x
        return self.rule.output(x, m, rng)

    def mass(self, y: int, r: int) -> float:
        """`P(next withdrawal is y | remaining r)`; zero when `y > r`.

        Args:
            y: The candidate amount, in the asset's smallest unit.
            r: The remaining balance.

        Returns:
            The probability: the rule's mass while `r >= threshold`, else the drain's atom at `r`.
        """
        if r < 1 or y > r:
            return 0.0
        if r < self.threshold:
            return 1.0 if self.drain and y == r else 0.0
        return self.rule.mass(y, r, self.quantum, self.m_model)

    def sample(self, r: int, rng: Random) -> tuple[int, int]:
        """Draw `x` on the quantum grid and return `(x, y)`; consumes `rng` in walk order.

        Args:
            r: The remaining balance.
            rng: The source of randomness, consumed in walk order.

        Returns:
            `(x, y)`, the need on the quantum grid and the recommended amount. Below the threshold
            this method returns `(r, r)` without drawing; the simulator calls it only for active adopters.
        """
        if r < self.threshold:
            return r, r
        n = r // self.quantum
        x = self.quantum * rng.randrange(1, n + 1)
        return x, self.output(x, r, rng.randrange)


def model(
    historical: object,
    rule: Need | DecimalLadder | PopularLadder,
    threshold: int,
    quantum: int,
    d: int,
    m_model: str = "double",
) -> tuple[object, Adopter, Adopter]:
    """The behaviour tuple `(historical, keep, drain)`, indexed by `NON`, `KEEP`, `DRAIN`.

    Args:
        historical: The non-adopter behaviour of :mod:`lab.behaviour`, exposing `active`/`sample`.
        rule: The adopter rule for both `KEEP` and `DRAIN`.
        threshold: The tool-use threshold, in the asset's smallest unit.
        quantum: The adopter need grid.
        d: The rounding unit.
        m_model: The maximum model, `double` or `balance`.

    Returns:
        The three behaviours in `NON`, `KEEP`, `DRAIN` order.
    """
    return (
        historical,
        Adopter(rule, threshold, quantum, d, drain=False, m_model=m_model),
        Adopter(rule, threshold, quantum, d, drain=True, m_model=m_model),
    )
