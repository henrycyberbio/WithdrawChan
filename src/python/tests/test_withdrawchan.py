"""Tests for the withdrawal recommendation logic.

The names and behaviour of these tests match the TypeScript suite in the JavaScript package, apart
from the tests that only make sense in one language (vector regeneration here, fraction reduction
there).
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

import withdrawchan as wc
from tests.vectors import EXAMPLES, USDT, VECTORS_PATH, build, example_amount

# ==== helpers ====


def _vectors() -> list[dict]:
    """Return the committed vector payload from the JSON file."""
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]


def _inputs(v: dict) -> tuple[int, int | None, dict]:
    """Parse a vector row into (`x`, `m`, `options`).

    Args:
        v: The raw vector entry from the serialized JSON file.

    Returns:
        A tuple of (`x`, `m`, `options`), where `options` contains the rounding unit and
        probabilities.
    """
    m = None if v["m"] is None else int(v["m"])
    return int(v["x"]), m, {"d": int(v["d"]), "q": Fraction(v["q"]), "p": Fraction(v["p"])}


# ==== vectors ====


def test_committed_vectors_equal_a_fresh_enumeration() -> None:
    # GIVEN: The vector file both implementations are tested against.
    # WHEN: Rebuilding it by exhaustive enumeration of every `rng` outcome.
    rebuilt = build()
    # THEN: The committed file is current, so a TypeScript pass means agreement with this code.
    assert rebuilt == _vectors()


def test_closed_form_distribution_equals_the_vectors() -> None:
    # GIVEN: Vectors whose distributions come from running `walk` on every `rng` outcome.
    for v in _vectors():
        x, m, opts = _inputs(v)
        # WHEN: Computing the closed-form distribution.
        dist = wc.distribution(x, m, **opts)
        # THEN: It matches outcome by outcome, so the audit formula describes the sampler.
        assert {str(k): f"{f.numerator}/{f.denominator}" for k, f in dist.items()} == v["distribution"]


def test_scripted_rng_reproduces_the_recorded_draw() -> None:
    # GIVEN: Each vector's recorded `rng` calls (`n`, `value`) and output.
    for v in _vectors():
        x, m, opts = _inputs(v)
        script = [(int(n), int(value)) for n, value in v["rng"]]
        calls: list[tuple[int, int]] = []

        def rng(n: int, calls: list[tuple[int, int]] = calls, script: list[tuple[int, int]] = script) -> int:
            calls.append((n, script[len(calls)][1]))
            return calls[-1][1]

        # WHEN: Replaying them.
        y = wc.recommend(x, m, rng=rng, **opts)
        # THEN: The same calls happen in the same order and give the same amount.
        assert (calls, y) == (script, int(v["y"]))


# ==== worked examples ====


@pytest.mark.parametrize("example", EXAMPLES["examples"], ids=lambda e: f"{e['x']}-{e['m']}")
def test_worked_example_has_the_tabulated_distribution(example: dict) -> None:
    # GIVEN: A hand-computed row of `../../examples.json` (USDT, `d` = 1 USDT, `q` = 1/2, `p` = 0).
    opts = {"d": example_amount(EXAMPLES["d"]), "q": Fraction(EXAMPLES["q"]), "p": Fraction(EXAMPLES["p"])}
    m = None if example["m"] is None else example_amount(example["m"])
    # WHEN: Computing its distribution.
    dist = wc.distribution(example_amount(example["x"]), m, **opts)
    # THEN: It matches the row, an oracle written independently of the code.
    assert dist == {example_amount(y): Fraction(f) for y, f in example["distribution"].items()}


# ==== privacy properties ====


def test_any_two_feasible_maximums_change_an_output_probability_at_most_twofold() -> None:
    # GIVEN: Needs across 1..300 and every maximum from `ceil(x)` to `10^4` in rounding unit 10, `q` = 1/2.
    d = 10
    worst = Fraction(0)
    for x in range(1, 301, 7):
        dists = {m: wc.distribution(x, m, d=d) for m in range(-(-x // d) * d, 10**4 + 1, d)}
        # WHEN: Comparing `P(y | m)` over all maximums `m >= y`.
        for y in set().union(*dists.values()):
            probs = [dist.get(y, Fraction(0)) for m, dist in dists.items() if m >= y]
            worst = max(worst, max(probs) / min(probs))
    # THEN: The ratio never exceeds `1 / (1 - q) = 2`: a maximum tied to the balance leaks at most 1 bit.
    assert worst == 2


@pytest.mark.parametrize(("x", "m"), [(1, 100), (37, 1000), (99, 100), (100, 1000), (1200, 12000)])
def test_output_never_falls_below_the_need_rounded_up(x: int, m: int) -> None:
    # GIVEN: A need below one rounding unit (1, 37, 99), on a rung (100, 1200), and a maximum from
    # the rung up to ten times `c0`.
    # WHEN: Listing possible outputs.
    low = min(wc.distribution(x, m, d=100, p=Fraction(1, 2)))
    # THEN: The raw need with its part below the rounding unit is never sent, which is why `m`
    # must be whole units.
    assert low >= -(-x // 100) * 100


# ==== defaults ====


def test_default_secure_rng_draws_an_amount_the_distribution_allows() -> None:
    # GIVEN: `x` = 520 USDT with the default maximum and the system's secure randomness.
    support = wc.distribution(520 * USDT, d=USDT)
    # WHEN: Drawing many times.
    seen = {wc.recommend(520 * USDT, d=USDT) for _ in range(200)}
    # THEN: Every draw is a possible output and all three rungs (520, 600, 1000) appear.
    assert seen == set(support)


# ==== units ====


def test_unit_scales_whole_tokens_to_the_smallest_unit() -> None:
    # GIVEN: Whole-token units of 1 (USD stablecoins) and 0.001 (ETH).
    # WHEN: Converting them to an asset's smallest unit.
    # THEN: The scale multiplies the unit, and an impossible scale or an unknown code raises.
    assert (wc.unit("USD-stable", 6), wc.unit("USD-stable", 18), wc.unit("ETH", 18)) == (10**6, 10**18, 10**15)
    with pytest.raises(KeyError):
        wc.unit("ETH-cents", 6)
    with pytest.raises(ValueError, match="not a whole number"):
        wc.unit("ETH", 2)


# ==== input contract ====


@pytest.mark.parametrize(("x", "m", "d"), [(0, None, 100), (150, 250, 100), (150, 100, 100), (150, None, 25)])
def test_invalid_need_maximum_or_rounding_unit_is_rejected(x: int, m: int | None, d: int) -> None:
    # GIVEN: A non-positive need, a maximum off the rounding unit, a maximum below the need, or a
    # rounding unit that is not a power of ten.
    # WHEN: Asking for a recommendation.
    # THEN: It raises instead of silently correcting the input.
    with pytest.raises(wc.AmountError):
        wc.recommend(x, m, d=d)


@pytest.mark.parametrize(("q", "p"), [(Fraction(5, 4), Fraction(0)), (Fraction(1, 2), Fraction(-1, 4))])
def test_probabilities_outside_the_unit_interval_are_rejected(q: Fraction, p: Fraction) -> None:
    # GIVEN: A climb probability above 1 and a decimal probability below 0.
    # WHEN: Asking for a recommendation, its distribution, or a walk on given rungs.
    # THEN: Each raises instead of returning negative probabilities.
    with pytest.raises(wc.AmountError):
        wc.recommend(520, 1040, d=10, q=q, p=p)
    with pytest.raises(wc.AmountError):
        wc.distribution(520, 1040, d=10, q=q, p=p)
    with pytest.raises(wc.AmountError):
        wc.walk([10, 20], 20, d=10, q=q, p=p)
    with pytest.raises(wc.AmountError):
        wc.walk_distribution([10, 20], 20, d=10, q=q, p=p)


def test_empty_ladder_is_rejected() -> None:
    # GIVEN: A walk with no rungs.
    # WHEN: Drawing an amount.
    # THEN: It raises the package error instead of indexing into the empty ladder.
    with pytest.raises(wc.AmountError, match="rungs must not be empty"):
        wc.walk([], 10, d=10)
