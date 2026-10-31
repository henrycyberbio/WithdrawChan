"""The `group` observer: a particle posterior over which deposits have already withdrawn.

It knows the rule and the adopter kind of the path (hence also the path's `KEEP`/`DRAIN` stopping
policy) but not the adopter count, the adopter identities, the event kinds or the true sources.  It is
built and scored only when a config's `observers` list names `group`.  A particle is

- `k`: its own candidate adopter count;
- its identified adopters and identified non-adopters, as `(i, r_i)` pairs sorted by `i`, where
  `r_i` is the remaining balance the particle attributes to deposit `i`.  Every other deposit is
  free.

The source of the next event is uniform over a particle's active deposits, a free deposit being an
adopter with probability `(k - |adopters|) / free`; the event's public amount `y` then moves the
identified deposits through the adopter's or the historical `mass(y, r)`.  An event with more
children than `max_states` is resampled down to that budget, by one of two schemes that differ only
in how many uniforms they draw: the streaming path shifts all positions by a single offset
(systematic), while the fallback draws one uniform per slot (stratified).  A public amount is always
at least 1 -- a ladder output never falls below `ceil_d(x)`, a drained remainder is the balance itself,
and a non-adopter's amount is positive -- so `y <= 0` never reaches the observer.

The candidate counts arrive as one `k` per `rho` entry and may repeat: the prior is uniform over
the entries, so a repeated `k` receives the sum of its weights, and particles carrying different
`k` never merge.

Two implementation points, both keeping the fixed numerical behaviour
----------------------------------------------------------------------
- Staging keeps each particle's positive masses as two index/value arrays and resampling is one
  `cumsum` plus one `searchsorted`.  The products `factor * mass` are still formed, and the sums
  still accumulated left to right, in the plain Python loop's order; each `math.fsum` returns one
  rounded value for the floats it is given, which bounds the rounding error but is still a
  floating-point total, not an exact real-valued sum.  What is preserved is the same operations on
  the same float64 values in the same order, hence the same results, particle order and RNG draw
  count as the reference loop.
- The fallback (no particle can explain `y`) must merge children of *different* parents, because
  `(parent, source) -> child` is not injective for `y >= 1`: balances `(30, 5, 20)` with `y = 2` give
  the same child from a parent that already tracks deposit 0 at `b_0 - y` and from one that inserts
  it there.  A child is therefore described by `(parent, source, remaining, branch)` and keyed by a
  fingerprint of its state.  The fingerprint only picks the dict bucket; `_ChildKey.__eq__` rebuilds
  both children and compares them exactly, so collisions of any arity still merge correctly.  The
  builtin `dict` supplies first-insertion order, which is also the reference loop's order.

Notation
--------
- `b_i`: the initial public balance of deposit `i`.
- `v`: an amount to score; `y`: an event's actual public amount.
- `n`: the number of deposits; `P = max_states`: the particle budget.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    import random
    from collections.abc import Iterator, Sequence

__all__ = ["Behaviour", "GroupObserver"]


class Behaviour(Protocol):
    """A person kind's next-withdrawal law: `mass(y, r)` and whether `r` is active."""

    def mass(self, y: int, r: int) -> float:
        """`P(next withdrawal is y | remaining r)`."""
        pass

    def active(self, r: int) -> bool:
        """Whether a person with remaining `r` can still act."""
        pass


#: A tracked deposit: `(i, r_i)`, its index and remaining balance.
Entry = tuple[int, int]
#: A particle: `(k, identified adopters, identified non-adopters)`, the entries sorted by `i`.
State = tuple[int, tuple[Entry, ...], tuple[Entry, ...]]
#: `(i, mass)` pairs with positive mass.
Pairs = list[tuple[int, float]]

#: Where the one changed coordinate of a fallback child lives.
_SHIFT_ADOPTER = 0
_SHIFT_OTHER = 1
_INSERT_ADOPTER = 2
_INSERT_OTHER = 3

_MASK = (1 << 63) - 1


def _shift(entries: tuple[Entry, ...], i: int, y: int) -> tuple[Entry, ...]:
    """`entries` with deposit `i`'s remaining balance reduced by `y`."""
    return tuple((j, rem - y if j == i else rem) for j, rem in entries)


def _insert(entries: tuple[Entry, ...], i: int, rem: int) -> tuple[Entry, ...]:
    """`entries` plus `(i, rem)`, kept sorted by `i` (`i` is not already present)."""
    pos = bisect_left(entries, (i, rem))
    return entries[:pos] + ((i, rem),) + entries[pos:]


def _triple_hash(group: int, i: int, rem: int) -> int:
    """Fingerprint of one tracked triple (or of `k`, with `group = 2`); arbitrary-size integers."""
    return hash((group, i, rem)) & _MASK


class _ChildKey:
    """One fallback child, described instead of materialised.

    Args:
        parent: The parent state tuple (shared, not copied).
        i: The source deposit's index.
        rem: The child's remaining balance for `i`.
        branch: Which coordinate changed, one of `_SHIFT_*`/`_INSERT_*`.
        fp: The child's fingerprint.

    The object is immutable by construction; the dict only needs `__hash__` and `__eq__`.
    """

    __slots__ = ("parent", "i", "rem", "branch", "fp")

    def __init__(self, parent: State, i: int, rem: int, branch: int, fp: int) -> None:
        self.parent = parent
        self.i = i
        self.rem = rem
        self.branch = branch
        self.fp = fp

    def __hash__(self) -> int:
        return self.fp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ChildKey):
            return NotImplemented
        if self.fp != other.fp:
            return False
        if (self.parent is other.parent and self.branch == other.branch
                and self.i == other.i and self.rem == other.rem):
            return True
        return self.child() == other.child()

    def child(self) -> State:
        """The child state this key describes."""
        k, adopters, others = self.parent
        branch, i, rem = self.branch, self.i, self.rem
        if branch == _SHIFT_ADOPTER:
            return (k, tuple((j, rem if j == i else r) for j, r in adopters), others)
        if branch == _SHIFT_OTHER:
            return (k, adopters, tuple((j, rem if j == i else r) for j, r in others))
        if branch == _INSERT_ADOPTER:
            return (k, _insert(adopters, i, rem), others)
        return (k, adopters, _insert(others, i, rem))


@dataclass
class _Scratch:
    """One event's derived work, reused by `update` when the next amount and states are unchanged."""

    v: int
    adopt: np.ndarray
    hist: np.ndarray
    memo_ad: dict[int, float]
    memo_hist: dict[int, float]
    actives: list[int]
    states: list[tuple[State, float]]


def _prior(ks: Sequence[int]) -> list[tuple[State, float]]:
    """One particle per distinct candidate `K`, weighted by how many candidates give that `K`.

    The caller passes one candidate per `rho` (`K = floor(rho * N + 1/2)`), so repeated values
    accumulate: the prior is uniform over `rho`, not over the distinct counts.
    """
    counts: dict[int, int] = {}
    for k in ks:
        counts[k] = counts.get(k, 0) + 1
    total = len(ks)
    return [((k, (), ()), count / total) for k, count in counts.items()]


# ==== group ====


class GroupObserver:
    """A particle posterior over who has withdrawn, knowing the rule and the adopter kind.

    A particle assigns every deposit that has withdrawn so far to its identified adopters (at most
    its own `k`) or its identified non-adopters, with the remaining balance it attributes to it; the
    other deposits are free.  The source of the next event is uniform over the particle's active
    deposits, and a free deposit is an adopter with probability
    `(k - identified adopters) / free`.  Particles beyond `max_states` are resampled down to it:
    systematic from a single offset on the streaming path, stratified one uniform per slot in the
    fallback.

    Args:
        historical: The non-adopter behaviour, supplying `mass(y, r)` and `active(r)`.
        adopter: This path's adopter behaviour, with the same two methods; it carries the
            `KEEP`/`DRAIN` stopping policy.
        balances: The initial public balances `b_i`, one per deposit.
        ks: One candidate adopter count per `rho` entry; repeats are allowed and weighted.
        max_states: The particle budget `P`.
        rng: The source of randomness; consumed only by resampling.
    """

    def __init__(
        self,
        historical: Behaviour,
        adopter: Behaviour,
        balances: Sequence[int],
        ks: Sequence[int],
        max_states: int,
        rng: random.Random,
    ) -> None:
        self.historical = historical
        self.adopter = adopter
        self.balances = tuple(balances)
        self.n = len(self.balances)
        self.ks = tuple(ks)
        self.max_states = max_states
        self.rng = rng
        self.fallback_count = 0
        self._index = 0
        self._static = _StaticObserver(historical, self.balances)
        self._states: list[tuple[State, float]] = _prior(self.ks)
        self._scratch: _Scratch | None = None

    def _active(self, state: State) -> int:
        """Active deposits of a particle: identified ones by their kind's `active`, plus every free
        deposit.

        Counting a free deposit as active presumes every initial balance lies in the adopter's
        active range: `b_i >= threshold` for `KEEP`, while `DRAIN` and the historical model accept
        any `b_i >= 1`.  The counting arithmetic is kept as frozen; it does not re-check the
        adopter's `active` for a free deposit.
        """
        _k, adopters, others = state
        return (self.n - len(adopters) - len(others)
                + sum(1 for _i, rem in adopters if self.adopter.active(rem))
                + sum(1 for _i, rem in others if self.historical.active(rem)))

    def _masses(self, v: int) -> tuple[np.ndarray, np.ndarray]:
        """Per-deposit `(adopter P(v | b_i), historical P(v | b_i))` at the initial balances."""
        adopt = np.array([self.adopter.mass(v, b) for b in self.balances])
        hist = np.array([self.historical.mass(v, b) for b in self.balances])
        return adopt, hist

    def _parts(
        self,
        state: State,
        v: int,
        adopt: np.ndarray,
        hist: np.ndarray,
        memo_ad: dict[int, float],
        memo_hist: dict[int, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-deposit `(adopter part, non-adopter part)` of `P(v | source = i)` for one particle.

        A free deposit's parts are weighted by `share` and `1 - share`, `share` being the
        probability that a free deposit is an adopter.
        """
        k, adopters, others = state
        free = self.n - len(adopters) - len(others)
        share = (k - len(adopters)) / free if free else 0.0
        ap = share * adopt
        op = (1.0 - share) * hist
        for i, rem in adopters:
            if rem not in memo_ad:
                memo_ad[rem] = self.adopter.mass(v, rem)
            ap[i] = memo_ad[rem]
            op[i] = 0.0
        for i, rem in others:
            if rem not in memo_hist:
                memo_hist[rem] = self.historical.mass(v, rem)
            op[i] = memo_hist[rem]
            ap[i] = 0.0
        return ap, op

    def posterior(self, v: int) -> np.ndarray:
        """`P(source = i | earlier public amounts, v)` for every deposit; the particles are unchanged.

        Args:
            v: The next event's public amount.

        Returns:
            A length-`n` distribution over the deposits; the historical posterior on the initial
            balances when no particle can produce `v`.
        """
        adopt, hist = self._masses(v)
        memo_ad: dict[int, float] = {}
        memo_hist: dict[int, float] = {}
        joint = np.zeros(self.n, dtype=np.float64)
        actives: list[int] = []
        for state, weight in self._states:
            active = self._active(state)
            actives.append(active)
            if active == 0:
                continue
            ap, op = self._parts(state, v, adopt, hist, memo_ad, memo_hist)
            joint += weight / active * (ap + op)
        self._scratch = _Scratch(v, adopt, hist, memo_ad, memo_hist, actives, self._states)
        total = float(joint.sum())
        if total == 0.0:
            return self._static.posterior(v)
        return joint * (1.0 / total)

    def update(self, y: int) -> None:
        """Condition the particles on the event's public amount `y` and advance the event index.

        Args:
            y: The event's public amount.
        """
        scratch = self._scratch
        if scratch is not None and scratch.v == y and scratch.states is self._states:
            adopt, hist = scratch.adopt, scratch.hist
            memo_ad, memo_hist = scratch.memo_ad, scratch.memo_hist
            actives: list[int] | None = scratch.actives
        else:
            adopt, hist = self._masses(y)
            memo_ad = {}
            memo_hist = {}
            actives = None
        staged: list[tuple[State, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        count = 0
        total = 0.0
        for slot, (state, weight) in enumerate(self._states):
            active = actives[slot] if actives is not None else self._active(state)
            if active == 0:
                continue
            ap, op = self._parts(state, y, adopt, hist, memo_ad, memo_hist)
            factor = weight / active
            ma = ap > 0.0
            mo = op > 0.0
            ia = np.flatnonzero(ma)
            io = np.flatnonzero(mo)
            wa = ap[ma]
            wo = op[mo]
            staged.append((state, factor, ia, wa, io, wo))
            count += ia.size + io.size
            total += factor * (math.fsum(wa.tolist()) + math.fsum(wo.tolist()))
        if count == 0:
            items = self._fallback_items(y)
        elif count <= self.max_states:
            merged: dict[State, float] = {}
            for state, factor, ia, wa, io, wo in staged:
                pa = list(zip(ia.tolist(), wa.tolist(), strict=False))
                po = list(zip(io.tolist(), wo.tolist(), strict=False))
                for key, weight in self._children(state, factor, y, pa, po):
                    merged[key] = merged.get(key, 0.0) + weight
            total = math.fsum(merged.values())
            items = [(key, weight / total) for key, weight in merged.items()]
        else:
            items = self._stream_np(staged, y, total)
        self._states = items
        self._scratch = None
        self._index += 1

    def _children(
        self, state: State, factor: float, y: int, pa: Pairs, po: Pairs
    ) -> Iterator[tuple[State, float]]:
        """Yield `(child, weight)` for one particle from its positive `(i, mass)` pairs."""
        k, adopters, others = state
        adopter_ids = {i for i, _ in adopters}
        other_ids = {i for i, _ in others}
        for branch, pairs in ((True, pa), (False, po)):
            for i, mass in pairs:
                yield (self._child(k, adopters, others, adopter_ids, other_ids, i, y, branch=branch),
                       factor * mass)

    def _stream_np(
        self,
        staged: list[tuple[State, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        y: int,
        total: float,
    ) -> list[tuple[State, float]]:
        """Systematic resampling to `max_states` from a single random offset, streaming the children
        without building them all.

        One `u` shifts every landing position, `(j + u) / max_states`, so the whole step costs one
        uniform, unlike the fallback's one uniform per slot.  One `cumsum` over the concatenated
        products and one `searchsorted` locate every landing position; a state is built only when a
        position lands in it.
        """
        n_particles = self.max_states
        scale = total / n_particles
        u = self.rng.random()
        targets = (np.arange(n_particles, dtype=np.float64) + u) * scale

        size = 0
        for _state, _factor, ia, _wa, io, _wo in staged:
            size += ia.size + io.size
        flat = np.empty(size, dtype=np.float64)
        blocks: list[tuple[int, int, int, bool]] = []
        pos = 0
        for slot, (_state, factor, _ia, wa, _io, wo) in enumerate(staged):
            if wa.size:
                start = pos
                np.multiply(wa, factor, out=flat[pos:pos + wa.size])
                pos += wa.size
                blocks.append((start, pos, slot, True))
            if wo.size:
                start = pos
                np.multiply(wo, factor, out=flat[pos:pos + wo.size])
                pos += wo.size
                blocks.append((start, pos, slot, False))
        # sequential left-to-right float64 accumulation, as in the plain loop
        np.cumsum(flat, out=flat)
        # first position whose cumulative total reaches each target; clamping to the last position
        # reproduces the trailing "emit the last child again" iterations
        hit = np.searchsorted(flat, targets, side="left")
        np.minimum(hit, size - 1, out=hit)

        starts = [b[0] for b in blocks]
        per_slot = []
        for state, _factor, _ia, _wa, _io, _wo in staged:
            k, adopters, others = state
            per_slot.append((k, adopters, others,
                             {i for i, _ in adopters}, {i for i, _ in others}))

        children: dict[int, State] = {}
        inv = 1.0 / n_particles
        out: list[tuple[State, float]] = []
        for h in hit.tolist():
            child = children.get(h)
            if child is None:
                start, _stop, slot, branch = blocks[bisect_right(starts, h) - 1]
                k, adopters, others, adopter_ids, other_ids = per_slot[slot]
                i = int((staged[slot][2] if branch else staged[slot][4])[h - start])
                child = self._child(k, adopters, others, adopter_ids, other_ids, i, y,
                                    branch=branch)
                children[h] = child
            out.append((child, inv))
        return out

    def _child(
        self,
        k: int,
        adopters: tuple[Entry, ...],
        others: tuple[Entry, ...],
        adopter_ids: set[int],
        other_ids: set[int],
        i: int,
        y: int,
        *,
        branch: bool,
    ) -> State:
        """The particle after deposit `i` withdraws `y`, as an adopter if `branch`: debit `i` if
        tracked, else add it to that group."""
        if branch:
            return (k, _shift(adopters, i, y), others) if i in adopter_ids \
                else (k, _insert(adopters, i, self.balances[i] - y), others)
        return (k, adopters, _shift(others, i, y)) if i in other_ids \
            else (k, adopters, _insert(others, i, self.balances[i] - y))

    def _fallback_items(self, y: int) -> list[tuple[State, float]]:
        """The fallback children, normalised and (if over budget) resampled.

        Children of different parents are merged exactly through `_ChildKey`, and the over-budget
        branch draws one uniform per slot, unlike `_stream_np`'s single offset.
        """
        self.fallback_count += 1
        static = self._static.posterior(y)
        nz = np.flatnonzero(static > 0.0)
        ids = nz.tolist()
        svals = [float(static[i]) for i in ids]
        balances = self.balances

        merged: dict[_ChildKey, float] = {}
        fp_cache: dict[int, int] = {}
        get = merged.get
        miss = object()

        for state, weight in self._states:
            k, adopters, others = state
            fp0 = fp_cache.get(id(state))
            if fp0 is None:
                fp0 = _triple_hash(2, k, 0)
                for i, rem in adopters:
                    fp0 ^= _triple_hash(0, i, rem)
                for i, rem in others:
                    fp0 ^= _triple_hash(1, i, rem)
                fp_cache[id(state)] = fp0
            adopter_ids = {i for i, _ in adopters}
            other_ids = {i for i, _ in others}
            adopter_rems = dict(adopters)
            other_rems = dict(others)
            free_is_adopter = len(adopters) < k
            for j, i in enumerate(ids):
                if i in adopter_ids:
                    old = adopter_rems[i]
                    rem = old - y
                    branch = _SHIFT_ADOPTER
                    fp = fp0 ^ _triple_hash(0, i, old) ^ _triple_hash(0, i, rem)
                elif i in other_ids:
                    old = other_rems[i]
                    rem = old - y
                    branch = _SHIFT_OTHER
                    fp = fp0 ^ _triple_hash(1, i, old) ^ _triple_hash(1, i, rem)
                elif free_is_adopter:
                    rem = balances[i] - y
                    branch = _INSERT_ADOPTER
                    fp = fp0 ^ _triple_hash(0, i, rem)
                else:
                    rem = balances[i] - y
                    branch = _INSERT_OTHER
                    fp = fp0 ^ _triple_hash(1, i, rem)
                key = _ChildKey(state, i, rem, branch, fp)
                add = weight * svals[j]
                prev = get(key, miss)
                merged[key] = add if prev is miss else prev + add

        keys = list(merged)
        m = len(keys)
        if m == 0:
            return []
        total = math.fsum(merged.values())
        if m <= self.max_states:
            return [(key.child(), merged[key] / total) for key in keys]

        n_particles = self.max_states
        positions = [(slot + self.rng.random()) / n_particles for slot in range(n_particles)]
        cdf = np.cumsum(np.fromiter((merged[key] / total for key in keys),
                                    dtype=np.float64, count=m))
        cdf[-1] = 1.0
        # `cdf[:-1]` is non-decreasing, so a position below 1 can only reach the forced 1.0 at the
        # end when every earlier entry is smaller -- and then the plain loop lands on the last item
        hit = np.searchsorted(cdf[:-1], np.array(positions, dtype=np.float64), side="left")
        inv = 1.0 / n_particles
        cache: dict[int, State] = {}
        out: list[tuple[State, float]] = []
        for h in hit.tolist():
            child = cache.get(h)
            if child is None:
                child = keys[h].child()
                cache[h] = child
            out.append((child, inv))
        return out


class _StaticObserver:
    """The historical (no-tool) posterior: `P(source = i | v)` proportional to historical `P(v | b_i)`.

    When the historical model gives `v` zero mass at every balance, the posterior is uniform over
    the deposits with `b_i >= v`.
    """

    def __init__(self, historical: Behaviour, balances: Sequence[int]) -> None:
        self.historical = historical
        self.balances = tuple(balances)
        self.fallback_count = 0

    def posterior(self, v: int) -> np.ndarray:
        joint = np.array([self.historical.mass(v, b) for b in self.balances])
        if joint.sum() == 0.0:
            self.fallback_count += 1
            eligible = np.array([1.0 if b >= v else 0.0 for b in self.balances])
            return eligible / eligible.sum()
        return joint / joint.sum()
