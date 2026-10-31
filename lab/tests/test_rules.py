from __future__ import annotations

import random
from fractions import Fraction

import pytest
from withdrawchan import distribution, walk_distribution

from lab.rules import (
    Adopter,
    DecimalLadder,
    Need,
    PopularLadder,
    _ceil_to,
    _floor_to,
    _ladder_fraction,
)

QUANTUM = 10
UNIT = 100
R = 2000
POPULAR_AMOUNTS = (200, 500, 1000, 10000)
POPULAR_COUNTS = (3, 5, 7, 9)


def _rule(rule_name: str, q: Fraction, p: Fraction, d: int = UNIT) -> DecimalLadder | PopularLadder:
    if rule_name == "decimal":
        return DecimalLadder(d, q, p)
    return PopularLadder(POPULAR_AMOUNTS, POPULAR_COUNTS, d, q, p)


# ==== published distribution comparison ====


@pytest.mark.parametrize("rule_name", ["decimal", "popular"])
@pytest.mark.parametrize("m_model", ["double", "balance"])
@pytest.mark.parametrize(("q", "p"), [
    (Fraction(1), Fraction(0)),
    (Fraction(1, 2), Fraction(0)),
    (Fraction(1, 2), Fraction(1, 2)),
])
def test_ladder_mass_matches_brute_force(rule_name: str, m_model: str, q: Fraction,
                                         p: Fraction) -> None:
    # GIVEN: A ladder rule whose rungs start at ceil_d(x), on a small grid with a coarse unit d.
    rule = _rule(rule_name, q, p)
    adopter = Adopter(rule, 0, QUANTUM, UNIT, drain=False, m_model=m_model)
    # WHEN: The exact adopter mass is compared with the per-x grid sum of the published distribution.
    expected = _brute(rule, R, QUANTUM, UNIT, m_model)
    # THEN: Every output amount agrees exactly and the mass sums to one, so dividing the grid into
    # one walk per (ceil_d(x), m) class loses no probability.
    for y in range(1, R + 1):
        assert adopter.mass(y, R) == float(expected.get(y, Fraction(0)))
    assert sum(expected.values()) == 1


def _brute(rule: DecimalLadder | PopularLadder, r: int, quantum: int, d: int,
           m_model: str) -> dict[int, Fraction]:
    """Exact `P(Y = y | r)` by summing per-`x` walk distributions with the fallback applied."""
    result: dict[int, Fraction] = {}
    n = r // quantum
    for k in range(1, n + 1):
        x = quantum * k
        c0 = _ceil_to(x, d)
        m = _floor_to(r, d)
        if m_model == "double":
            m = min(_ceil_to(2 * x, d), m)
        if m < c0:
            result[x] = result.get(x, Fraction(0)) + 1
            continue
        dist = (distribution(x, m, d=d, q=rule.q, p=rule.p) if isinstance(rule, DecimalLadder)
                else walk_distribution(rule.rungs(c0, m), m, d=d, q=rule.q, p=rule.p))
        for y, v in dist.items():
            result[y] = result.get(y, Fraction(0)) + v
    return {y: v / n for y, v in result.items()}


def test_need_mass_is_uniform_on_the_quantum_grid() -> None:
    # GIVEN: The need rule, which recommends exactly x on the quantum grid.
    adopter = Adopter(Need(), 0, QUANTUM, UNIT, drain=False, m_model="double")
    # WHEN: Asking for the mass of one grid amount and of an off-grid amount.
    on_grid, off_grid = adopter.mass(50, R), adopter.mass(55, R)
    # THEN: The grid amount has probability 1/n and the off-grid amount is impossible.
    assert (on_grid, off_grid) == (1.0 / (R // QUANTUM), 0.0)


# ==== segment reference (the implementation the closed form replaces) ====


def _segments(r: int, quantum: int, d: int, m_model: str) -> list[tuple[int, int, int]]:
    """`(c0, m, count)` classes of the quantum grid `x in [quantum, r]`."""
    balance = m_model == "balance"
    m = _floor_to(r, d)
    segments: list[tuple[int, int, int]] = []
    for a in range(1, -(-r // d) + 1):
        c0 = a * d
        lo = quantum * (((a - 1) * d) // quantum + 1)
        hi = quantum * (min(c0, r) // quantum)
        if lo > hi:
            continue
        if balance:
            segments.append((c0, m, (hi - lo) // quantum + 1))
            continue
        cut = ((2 * a - 1) * d) // 2
        for slo, shi, b in ((lo, min(hi, cut), 2 * a - 1), (max(lo, cut + 1), hi, 2 * a)):
            if slo <= shi:
                segments.append((c0, min(b * d, m), (shi - slo) // quantum + 1))
    return segments


def _group_key(y: int, r: int, d: int, m_model: str) -> tuple[int, int]:
    if m_model == "balance":
        return (_ceil_to(y, d), _floor_to(r, d))
    return (_ceil_to(y, d), min(_ceil_to(2 * y, d), _floor_to(r, d)))


_REFERENCE: dict = {}


def _segment_mass(rule: DecimalLadder | PopularLadder, y: int, r: int, quantum: int,
                  m_model: str) -> Fraction:
    """Exact `P(Y = y | r)` by enumerating every `(ceil_d(x), m)` class and its walk distribution."""
    key = (repr(rule), r, quantum, m_model)
    entry = _REFERENCE.get(key)
    if entry is None:
        n = r // quantum
        base: dict[int, Fraction] = {}
        fallback: dict[tuple[int, int], Fraction] = {}
        for c0, m, count in _segments(r, quantum, rule.d, m_model):
            if m < c0:
                fallback[(c0, m)] = Fraction(1)
                continue
            dist = walk_distribution(rule.rungs(c0, m), m, d=rule.d, q=rule.q, p=rule.p)
            for yy, v in dist.items():
                base[yy] = base.get(yy, Fraction(0)) + v * count
        entry = (base, fallback, n)
        _REFERENCE[key] = entry
    base, fallback, n = entry
    if n < 1 or y > r:
        return Fraction(0)
    total = base.get(y, Fraction(0))
    if y % quantum == 0 and quantum <= y <= r:
        total += fallback.get(_group_key(y, r, rule.d, m_model), Fraction(0))
    return total / n


# ==== pieces ====


def _piece_rules() -> list[tuple[str, DecimalLadder | PopularLadder]]:
    return [
        ("decimal", DecimalLadder(UNIT, Fraction(1, 2), Fraction(0))),
        ("popular", PopularLadder(POPULAR_AMOUNTS, POPULAR_COUNTS, UNIT, Fraction(1, 2), Fraction(0))),
        ("popular_close", PopularLadder((150, 155, 300, 305, 1000, 1007), (4, 6, 7, 9, 11, 12),
                                        UNIT, Fraction(1, 2), Fraction(0))),
        ("popular_multiple", PopularLadder((100, 200, 400), (2, 3, 4), UNIT, Fraction(1, 2), Fraction(0))),
        ("popular_small_d", PopularLadder((10, 20, 35, 40, 50), (2, 3, 4, 5, 6),
                                          10, Fraction(1, 2), Fraction(0))),
    ]


@pytest.mark.parametrize(("name", "rule"), _piece_rules())
def test_pieces_partition_and_reproduce_rungs(name: str, rule: DecimalLadder | PopularLadder) -> None:
    # GIVEN: A ladder and maximums that are multiples of d, with amounts that are multiples of d, not
    # multiples, and within one d of each other.
    d = rule.d
    for top in (d, 3 * d, 10 * d, 25 * d, 37 * d):
        pieces = rule.pieces(top)
        # WHEN: The pieces are read as c0 ranges and their tails compared with rungs() for every m.
        covered: list[int] = []
        previous = 0
        for lo, hi, tail in pieces:
            assert lo % d == 0
            assert hi % d == 0
            assert lo <= hi
            assert lo > previous
            previous = hi
            assert all(t <= 2 * top for t in tail)
            covered += list(range(lo, hi + d, d))
        # THEN: The pieces partition the c0 multiples of d in [d, top].
        assert covered == list(range(d, top + 1, d))
        for lo, hi, tail in pieces:
            for c0 in range(lo, hi + d, d):
                for m in range(c0, 2 * top + 1, d):
                    assert rule.rungs(c0, m) == [c0] + [t for t in tail if t <= m]


# ==== closed form vs the segment reference ====


def _mass_ys(rule: DecimalLadder | PopularLadder, r: int, quantum: int, d: int,
             rng: random.Random, *, full: bool) -> range | list[int]:
    """All `y` for small `r`; rung neighbourhoods, grid points and random draws for large `r`."""
    if full:
        return range(1, r + 1)
    ys: set[int] = set()
    top = _ceil_to(r, d)
    for c0 in range(d, top + 1, d):
        for t in rule.rungs(c0, 2 * top)[1:]:
            ys.update(range(max(1, t - d), min(r, t + d) + 1))
    step = quantum * max(1, r // (20 * quantum))
    ys.update(range(quantum, r + 1, step))
    ys.update(rng.randrange(1, r + 1) for _ in range(25))
    return sorted(y for y in ys if 1 <= y <= r)


@pytest.mark.parametrize("rule_name", ["decimal", "popular"])
@pytest.mark.parametrize("m_model", ["double", "balance"])
@pytest.mark.parametrize(("d", "quantum"), [(10, 7), (100, 25)])
@pytest.mark.parametrize(("q", "p"), [(Fraction(1), Fraction(0)), (Fraction(1, 3), Fraction(1, 10))])
def test_closed_form_matches_segment_reference(rule_name: str, m_model: str, q: Fraction,
                                               p: Fraction, d: int, quantum: int) -> None:
    # GIVEN: A ladder, both maximum models, an odd grid that does not divide d and a finer one on a
    # larger d.
    rule = _rule(rule_name, q, p, d)
    rng = random.Random(20241009)
    # WHEN: The closed form Fraction is compared with the segment reference for many y and r.
    for r in (d, 3 * d + 1, 12 * d, 150 * d, 150 * d + 37):
        for y in _mass_ys(rule, r, quantum, d, rng, full=r <= 3 * d):
            # THEN: Every probability is identical as an exact Fraction.
            assert _ladder_fraction(rule, y, r, quantum, m_model) == _segment_mass(rule, y, r, quantum, m_model)


@pytest.mark.parametrize("m_model", ["double", "balance"])
@pytest.mark.parametrize("q", [Fraction(1), Fraction(1, 3)])
@pytest.mark.parametrize("p", [Fraction(0), Fraction(1, 10)])
def test_closed_form_matches_reference_for_close_and_non_multiple_amounts(
        m_model: str, q: Fraction, p: Fraction) -> None:
    # GIVEN: A popular ladder whose amounts are not multiples of d and lie within one d of each other.
    rule = PopularLadder((150, 155, 300, 305, 1000, 1007), (4, 6, 7, 9, 11, 12),
                         UNIT, q, p)
    rng = random.Random(13)
    # WHEN: The closed form is compared with the reference for every y on a few maximums.
    for r in (UNIT, 5 * UNIT, 30 * UNIT):
        for y in _mass_ys(rule, r, UNIT, UNIT, rng, full=r <= 5 * UNIT):
            # THEN: The amounts inside and around d-buckets are accounted for exactly.
            assert _ladder_fraction(rule, y, r, UNIT, m_model) == _segment_mass(rule, y, r, UNIT, m_model)


# ==== adopter output ====


def test_adopter_output_lowers_the_default_cap_to_the_balance() -> None:
    # GIVEN: A USDT-scale decimal ladder (d = 1e6, q = 1) and a balance under the default maximum.
    rule = DecimalLadder(10**6, Fraction(1), Fraction(0))
    adopter = Adopter(rule, 0, 10**6, 10**6, drain=False, m_model="double")
    # WHEN: A 437 USDT need meets a 700.5 USDT balance: ceil(2x) = 874 exceeds the balance.
    # THEN: The maximum drops to floor(700.5) = 700 and the walk climbs 437 -> 500, since 1000 > 700.
    assert adopter.output(437 * 10**6, 700_500_000, random.Random(0).randrange) == 500 * 10**6


def test_adopter_output_is_x_when_the_cap_is_below_ceil_of_the_need() -> None:
    # GIVEN: A USDT-scale decimal ladder (d = 1e6, q = 1) and a balance just under ceil_d(x).
    rule = DecimalLadder(10**6, Fraction(1), Fraction(0))
    rng = random.Random(0).randrange
    # WHEN: Ceil(437.2) = 438 exceeds floor(437.4) = 437, so the tool cannot be used.
    # THEN: Both maximum models output the need itself.
    assert Adopter(rule, 0, 10**6, 10**6, drain=False, m_model="double").output(
        437_200_000, 437_400_000, rng) == 437_200_000
    assert Adopter(rule, 0, 10**6, 10**6, drain=False, m_model="balance").output(
        437_200_000, 437_400_000, rng) == 437_200_000
