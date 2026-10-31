"""Tests for the `Ratio` schema type: fraction text validated once at decode and kept verbatim, with
`value` re-parsed from the text on each access."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from lab.schema import ExperimentConfig, Ratio, SchemaError

CONFIGS = Path(__file__).parents[1] / "configs"
CONFIG_NAMES = sorted(path.name for path in CONFIGS.glob("*.json"))


def _raw(name: str) -> dict:
    """The config file as the JSON object it holds."""
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def _with(raw: dict, key: str, value: object) -> dict:
    """The JSON object with a dotted key replaced, so a nested field can be tampered with."""
    head, _, tail = key.partition(".")
    if not tail:
        return {**raw, head: value}
    return {**raw, head: {**raw[head], tail: value}}


def _fraction_fields(config: ExperimentConfig) -> dict[str, Ratio]:
    """Every single-valued fraction field of a config, by dotted name."""
    return {
        "theta.pi": config.theta.pi,
        "theta.s": config.theta.s,
        "theta.w": config.theta.w,
        "logit_train.l2": config.logit_train.l2,
        "logit_train.pi_lo": config.logit_train.pi_lo,
        "logit_train.pi_hi": config.logit_train.pi_hi,
        "logit_train.s_hi": config.logit_train.s_hi,
        "logit_train.w_hi": config.logit_train.w_hi,
        "combinatorial_tolerance": config.combinatorial_tolerance,
        "p": config.p,
    }


# ==== round-trip ====


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_a_published_config_round_trips_its_fraction_text(name: str) -> None:
    # GIVEN: A published config file, several of which hold non-canonical fraction text ("422/740",
    # "0.001", "0.6055").
    raw = _raw(name)
    # WHEN: Decoding it through the schema and encoding it back.
    config = ExperimentConfig.from_json(raw)
    # THEN: The JSON object comes out identical, so a run's config record keeps the text it read.
    assert config.to_json() == raw


# ==== decoded values ====


def test_fraction_fields_decode_to_ratios_of_their_exact_value() -> None:
    # GIVEN: The ETH config, whose theta and training ranges are written as text.
    config = ExperimentConfig.from_json(_raw("eth.json"))
    # WHEN: Reading every fraction-typed field.
    fields = _fraction_fields(config)
    # THEN: Each one is a Ratio carrying its text and the exact rational it names: "422/740" is not
    # reduced to 211/370 and "0.001" is not rewritten as 1/1000.
    assert all(type(value) is Ratio for value in fields.values())
    assert fields["theta.pi"] == "422/740"
    assert fields["theta.pi"].value == Fraction(422, 740)
    assert fields["theta.s"] == "3/263"
    assert fields["theta.s"].value == Fraction(3, 263)
    assert fields["theta.w"].value == Fraction(0)
    assert fields["logit_train.l2"] == "0.001"
    assert fields["logit_train.l2"].value == Fraction(1, 1000)
    assert fields["logit_train.pi_lo"].value == Fraction(5343, 10000)
    assert fields["logit_train.pi_hi"].value == Fraction(6055, 10000)
    assert fields["logit_train.s_hi"].value == Fraction(33, 1000)
    assert fields["logit_train.w_hi"].value == Fraction(1)
    assert fields["combinatorial_tolerance"].value == Fraction(0)
    assert fields["p"].value == Fraction(0)


def test_the_rho_axis_decodes_every_entry_to_a_ratio() -> None:
    # GIVEN: The ETH config's adoption ratios.
    config = ExperimentConfig.from_json(_raw("eth.json"))
    # WHEN: Reading the tuple.
    # THEN: Each entry is a Ratio of its own text, the plain "0" and "1" included.
    assert all(type(rho) is Ratio for rho in config.rhos)
    assert [str(rho) for rho in config.rhos] == ["0", "1/100", "1/40", "1/10", "1/4", "1/2", "3/4", "1"]
    assert [rho.value for rho in config.rhos] == [Fraction(0), Fraction(1, 100), Fraction(1, 40),
                                                  Fraction(1, 10), Fraction(1, 4), Fraction(1, 2),
                                                  Fraction(3, 4), Fraction(1)]


def test_an_explicit_p_decodes_to_its_own_ratio() -> None:
    # GIVEN: A config that sets p.
    config = ExperimentConfig.from_json(_raw("usdc-p-1-2.json"))
    # WHEN: Reading the field.
    # THEN: The Ratio keeps the text and value of "1/2".
    assert type(config.p) is Ratio
    assert config.p == "1/2"
    assert config.p.value == Fraction(1, 2)


# ==== rejection at the boundary ====


@pytest.mark.parametrize(("key", "value"), [
    ("p", "abc"),
    ("p", "1/0"),
    ("combinatorial_tolerance", "abc"),
    ("theta.pi", "1/0"),
    ("theta.s", "abc"),
    ("logit_train.l2", "1/0"),
])
def test_a_malformed_fraction_is_rejected_at_decode(key: str, value: object) -> None:
    # GIVEN: A config whose fraction field is not a number ("abc") or divides by zero ("1/0").
    raw = _with(_raw("eth.json"), key, value)
    # WHEN: Decoding it.
    # THEN: The boundary check refuses it, so no later code sees an unparsable fraction.
    with pytest.raises(SchemaError):
        ExperimentConfig.from_json(raw)


@pytest.mark.parametrize(("key", "value"), [
    ("p", 1),
    ("p", 0.5),
    ("combinatorial_tolerance", 0),
    ("theta.pi", 1),
    ("rhos", [0, "1/2"]),
])
def test_a_non_string_fraction_is_rejected_at_decode(key: str, value: object) -> None:
    # GIVEN: A config whose fraction field is a JSON number or a number inside the rho array.
    raw = _with(_raw("eth.json"), key, value)
    # WHEN: Decoding it.
    # THEN: The boundary check refuses it, so a float never silently becomes a Ratio.
    with pytest.raises(SchemaError):
        ExperimentConfig.from_json(raw)
