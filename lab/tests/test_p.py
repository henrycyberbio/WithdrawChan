"""Tests for the config's `p` axis: the optional field and the rules it reaches."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from lab.rules import DecimalLadder, Need, PopularLadder
from lab.schema import ExperimentConfig
from lab.study import rule_object

CONFIGS = Path(__file__).parents[1] / "configs"


def _load(name: str) -> tuple[dict, ExperimentConfig]:
    data = json.loads((CONFIGS / name).read_text(encoding="utf-8"))
    return data, ExperimentConfig.from_json(data)


def test_config_p_defaults_to_zero_and_decodes_a_fraction() -> None:
    # GIVEN: A config with an explicit p and one that omits it.
    _, explicit = _load("usdt-p-1-2.json")
    _, legacy = _load("usdt.json")
    # WHEN: Reading the field.
    # THEN: The explicit fraction survives and the omitted key falls back to "0".
    assert explicit.p == "1/2"
    assert Fraction(explicit.p) == Fraction(1, 2)
    assert legacy.p == "0"


def test_p_reaches_every_rounding_rule_and_not_need() -> None:
    # GIVEN: The four rounding rules and the need rule, built at p = 1/2.
    p = Fraction(1, 2)
    # WHEN: Building each rule object.
    rules = {label: rule_object(label, 10, (40, 60, 50, 50), p)
             for label in ("decstep", "decsmooth", "popstep", "popsmooth", "need")}
    # THEN: Every rounding rule carries the probability and the need rule has no random decimal.
    assert all(rules[label].p == p for label in ("decstep", "decsmooth", "popstep", "popsmooth"))
    assert isinstance(rules["decstep"], DecimalLadder)
    assert isinstance(rules["popstep"], PopularLadder)
    assert isinstance(rules["need"], Need)


def test_p_one_adds_a_uniform_tail_and_p_zero_stops_on_the_rung() -> None:
    # GIVEN: A decimal rule at p = 1 and at p = 0, on a maximum one whole unit above the only rung.
    def rng(_n: int) -> int:
        return 7

    tail = rule_object("decstep", 10, (), Fraction(1))
    plain = rule_object("decstep", 10, (), Fraction(0))
    # WHEN: The walk's random-decimal coin lands 7 below one whole unit.
    # THEN: With `p = 1` the random decimal is added (possibly zero); `p = 0` stops on the rung (10).
    assert tail.output(10, 20, rng) == 17
    assert plain.output(10, 20, rng) == 10
