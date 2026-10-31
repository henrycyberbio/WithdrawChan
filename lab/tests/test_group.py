"""The `group` observer: unknown adopter count, KEEP/DRAIN kinds, the exact fallback merge, the
particle budget and the RNG draw count."""

from __future__ import annotations

import math
import random

import pytest

from lab.observers.group import GroupObserver


class _Flat:
    """Duck-typed behaviour: `value` for every amount in `amounts` with `y <= r`, else zero."""

    def __init__(self, amounts: tuple[int, ...], value: float = 1.0, min_active: int = 1) -> None:
        self.amounts = frozenset(amounts)
        self.value = value
        self.min_active = min_active

    def mass(self, y: int, r: int) -> float:
        """`value` for the listed amounts that fit, else zero."""
        return self.value if y in self.amounts and 0 <= y <= r else 0.0

    def active(self, r: int) -> bool:
        """Active while `r >= min_active`."""
        return r >= self.min_active


class _Keep:
    """A `KEEP` adopter: the rule's amounts at or above `threshold`, nothing below it."""

    def __init__(self, amounts: tuple[int, ...], threshold: int = 5) -> None:
        self.amounts = frozenset(amounts)
        self.threshold = threshold

    def mass(self, y: int, r: int) -> float:
        """The rule's amounts while active, else zero."""
        return 1.0 if r >= self.threshold and y in self.amounts and y <= r else 0.0

    def active(self, r: int) -> bool:
        """Active while `r >= threshold`."""
        return r >= self.threshold


class _Drain:
    """A `DRAIN` adopter: the rule's amounts while active, the whole remainder once below it."""

    def __init__(self, amounts: tuple[int, ...], threshold: int = 5) -> None:
        self.amounts = frozenset(amounts)
        self.threshold = threshold

    def mass(self, y: int, r: int) -> float:
        """The rule's amounts above the threshold, the atom at `r` below it."""
        if r < 1 or y > r:
            return 0.0
        if r < self.threshold:
            return 1.0 if y == r else 0.0
        return 1.0 if y in self.amounts else 0.0

    def active(self, r: int) -> bool:
        """Active while any balance remains."""
        return r >= 1


class _CountingRng(random.Random):
    """A `random.Random` that counts its `random()` calls."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed)
        self.draws = 0

    def random(self) -> float:
        """One uniform draw, counted."""
        self.draws += 1
        return super().random()


def _observer(
    balances: tuple[int, ...],
    ks: tuple[int, ...],
    max_states: int = 64,
    *,
    historical: object | None = None,
    adopter: object | None = None,
    seed: int = 0,
) -> GroupObserver:
    """A `GroupObserver` over the given balances and candidate counts."""
    if historical is None:
        historical = _Flat((1, 2, 3))
    if adopter is None:
        adopter = _Flat((1, 2, 3))
    return GroupObserver(historical, adopter, balances, ks, max_states, random.Random(seed))


# ==== prior over the candidate counts ====


def test_prior_is_uniform_over_the_rho_entries_and_keeps_duplicates() -> None:
    # GIVEN: Three candidate counts for two distinct K values.
    observer = _observer((10, 10), (2, 2, 5))
    # WHEN: The particles are built.
    states = observer._states
    # THEN: Each distinct K appears once, weighted by how many rho entries give it.
    assert [state for state, _weight in states] == [(2, (), ()), (5, (), ())]
    assert [weight for _state, weight in states] == [2 / 3, 1 / 3]


def test_state_count_follows_the_budget_after_an_event() -> None:
    # GIVEN: Four deposits, one candidate count and a budget of three particles.
    observer = _observer((10, 10, 10, 10), (1,), max_states=3)
    # WHEN: The first event is conditioned on.
    observer.update(1)
    # THEN: The particle count is held at the budget, with equal weights.
    assert len(observer._states) == 3
    assert [weight for _state, weight in observer._states] == [1 / 3] * 3


def test_unknown_k_with_duplicate_counts_keeps_the_prior_mass_per_k() -> None:
    # GIVEN: Candidate counts with a repeat, so K = 1 has twice the prior weight of K = 2.
    observer = _observer((10, 10, 10), (1, 1, 2), max_states=16)
    # WHEN: One event is conditioned on.
    observer.update(1)
    # THEN: Every particle carries one candidate count, and the K masses follow the prior.
    mass: dict[int, float] = {}
    for state, weight in observer._states:
        assert state[0] in {1, 2}
        mass[state[0]] = mass.get(state[0], 0.0) + weight
    assert mass[1] == pytest.approx(2 / 3, rel=1e-15)
    assert mass[2] == pytest.approx(1 / 3, rel=1e-15)


# ==== posterior and kinds ====


def test_posterior_splits_a_free_deposit_between_the_two_kinds() -> None:
    # GIVEN: A single particle with K = 1 and two free deposits.
    observer = _observer((10, 10), (1,))
    # WHEN: Scoring an amount both kinds can produce.
    q = observer.posterior(5)
    # THEN: The two deposits are equally likely.
    assert q.tolist() == [0.5, 0.5]


def test_posterior_falls_back_to_the_unaware_distribution_when_no_particle_matches() -> None:
    # GIVEN: An amount no kind produces.
    observer = _observer((10, 10), (1,))
    # WHEN: Scoring it.
    q = observer.posterior(7)
    # THEN: The result is the historical posterior over the eligible deposits.
    assert q.tolist() == [0.5, 0.5]


def test_keep_and_drain_adopters_differ_only_below_the_threshold() -> None:
    # GIVEN: One identified adopter with `r = 3` and one free deposit, threshold 5.
    balances, ks = (10, 10), (1,)
    keep = _observer(balances, ks, adopter=_Keep((1,)))
    drain = _observer(balances, ks, adopter=_Drain((1,)))
    state = (1, ((0, 3),), ())
    # WHEN: Counting the particle's active deposits.
    # THEN: The KEEP depositor has stopped; the DRAIN depositor still takes the remainder once.
    assert keep._active(state) == 1
    assert drain._active(state) == 2


def test_a_keep_adopter_below_the_threshold_is_dropped_from_the_posterior() -> None:
    # GIVEN: A particle whose adopter has stopped and one free deposit.
    observer = _observer((10, 10), (1,), adopter=_Keep((1, 2, 3)))
    observer._states = [((1, ((0, 3),), ()), 1.0)]
    # WHEN: Scoring an amount deposit 1 can still produce.
    q = observer.posterior(2)
    # THEN: Only the free deposit is a candidate.
    assert q.tolist() == [0.0, 1.0]


# ==== exact fallback merge ====


def test_children_of_two_parents_merge_to_one_state() -> None:
    # GIVEN: The non-injective pair: balances (30, 5, 20), y = 2, and two parents that both reach
    # the child (2, ((0, 28), (1, 1)), ((2, 13),)) -- one by shifting deposit 2, one by inserting 0.
    observer = _observer((30, 5, 20), (2,), historical=_Flat(()), adopter=_Flat(()))
    p1 = (2, ((0, 28), (1, 1)), ((2, 15),))
    p2 = (2, ((1, 1),), ((2, 13),))
    observer._states = [(p1, 0.6), (p2, 0.4)]
    # WHEN: The event is conditioned on, forcing the fallback (no kind produces 2).
    observer.update(2)
    # THEN: Six children collapse to five, the shared one carrying the folded weight.
    states = dict(observer._states)
    assert len(observer._states) == 5
    assert len(states) == 5
    assert observer.fallback_count == 1
    shared = (2, ((0, 28), (1, 1)), ((2, 13),))
    assert states[shared] == pytest.approx(1 / 3, rel=1e-15)
    assert states[(2, ((0, 26), (1, 1)), ((2, 15),))] == pytest.approx(0.2, rel=1e-15)
    assert states[(2, ((1, 1),), ((2, 11),))] == pytest.approx(0.4 / 3, rel=1e-15)
    assert math.fsum(states.values()) == 1.0


def test_fallback_with_duplicate_parents_folds_every_copy() -> None:
    # GIVEN: Two copies of the same parent, so every child is folded twice.
    observer = _observer((30, 5, 20), (2,), historical=_Flat(()), adopter=_Flat(()))
    parent = (2, ((0, 28), (1, 1)), ((2, 15),))
    observer._states = [(parent, 0.5), (parent, 0.5)]
    # WHEN: The fallback merges their children.
    observer.update(2)
    # THEN: Each of the three children keeps weight 1/3 and no copy is duplicated.
    assert len(observer._states) == 3
    assert [weight for _state, weight in observer._states] == pytest.approx([1 / 3] * 3, rel=1e-15)


def test_fallback_above_the_budget_resamples_and_consumes_one_draw_per_slot() -> None:
    # GIVEN: A fallback that produces three children with a budget of two particles.
    rng = _CountingRng(3)
    observer = GroupObserver(_Flat(()), _Flat(()), (30, 5, 20), (2,), 2, rng)
    observer._states = [((2, ((0, 28), (1, 1)), ((2, 15),)), 1.0)]
    # WHEN: The event is conditioned on.
    observer.update(2)
    # THEN: The particles are resampled to the budget, one draw per slot.
    assert len(observer._states) == 2
    assert [weight for _state, weight in observer._states] == [0.5, 0.5]
    assert rng.draws == 2


# ==== large integers ====


def test_balances_beyond_int64_are_kept_exact() -> None:
    # GIVEN: Two deposits whose balances exceed int64 and an amount far past int64 as well.
    big = 2**100
    y = big - 3
    observer = _observer((big, big + 7), (1,), max_states=2, historical=_Flat((y,)),
                         adopter=_Flat((y,)))
    # WHEN: One event is conditioned on and resampled.
    observer.update(y)
    # THEN: Each particle tracks exactly one deposit at the exact integer remainder.
    assert len(observer._states) == 2
    for state, _weight in observer._states:
        k, adopters, others = state
        assert k == 1
        tracked = dict(adopters) | dict(others)
        assert len(tracked) == 1
        i, rem = next(iter(tracked.items()))
        assert rem == observer.balances[i] - y
        assert rem in (3, 10)


# ==== RNG ====


def test_resampling_draws_exactly_one_uniform_per_slot() -> None:
    # GIVEN: A budget of three and enough children to need streaming resampling.
    rng = _CountingRng(11)
    observer = GroupObserver(_Flat((1,)), _Flat((1,)), (10, 10, 10, 10), (1,), 3, rng)
    # WHEN: One event is conditioned on.
    observer.update(1)
    # THEN: The streamed systematic resampler consumed one offset draw, not one per slot.
    assert len(observer._states) == 3
    assert rng.draws == 1


def test_same_seed_reproduces_states_and_weights() -> None:
    # GIVEN: Two observers built identically, with a repeated candidate count.
    events = (1, 2, 1, 3, 1)
    first = _observer((10, 10, 10), (1, 2), max_states=4, seed=5)
    second = _observer((10, 10, 10), (1, 2), max_states=4, seed=5)
    # WHEN: Both are conditioned on the same stream.
    for y in events:
        assert first.posterior(y).tolist() == second.posterior(y).tolist()
        first.update(y)
        second.update(y)
    # THEN: The final particles and weights agree bit for bit.
    assert [(state, weight.hex()) for state, weight in first._states] == \
        [(state, weight.hex()) for state, weight in second._states]
