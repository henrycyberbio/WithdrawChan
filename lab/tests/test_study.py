from __future__ import annotations

from fractions import Fraction

import pytest

from lab.schema import Deposit, ExperimentConfig, LogitTrainSpec, Manifest, SchemaError, Snapshot, ThetaSpec, Withdrawal
from lab.sim import derive_seed
from lab.study import inputs_from, theta_of

MANIFEST = Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, 6, 10**6, 1, 99,
                    "0x" + "00" * 32, None)
SNAPSHOT = Snapshot(
    deposits=(
        Deposit(10, 0, "0xa", "0xd", "1", "2", 100, "3"),
        Deposit(15, 0, "0xa", "0xd", "1", "2", 0, "3"),
        Deposit(20, 0, "0xa", "0xd", "1", "2", 200, "3"),
        Deposit(50, 0, "0xa", "0xd", "1", "2", 300, "3"),
    ),
    withdrawals=(
        Withdrawal(10, 1, "0xb", 50, 0, "0xr", None, "1", "2"),
        Withdrawal(20, 1, "0xb", 60, 0, "0xr", None, "1", "2"),
        Withdrawal(25, 1, "0xb", 0, 0, "0xr", None, "1", "2"),
        Withdrawal(30, 1, "0xb", 70, 0, "0xr", None, "1", "2"),
    ),
    ragequits=(),
    roots=(),
)
THETA = ThetaSpec("9/22", "0", "0", "hi", 10**6, "P2 anchors, USDT (Wilson 0.23-0.61)")
LOGIT_TRAIN = LogitTrainSpec(60, "0.001", "0.2325581619647187", "0.6126518152368488", "0.2991450484195441",
                       "1", "synthetic ranges")


def test_inputs_from_uses_every_positive_deposit_before_the_split_once() -> None:
    # GIVEN: A snapshot with a zero-value deposit before the split and a positive deposit at it.
    config = ExperimentConfig("manifest.json", "651b808f", 30, THETA, 5, ("0", "1/40", "1/10"),
                              ("need",), ("double", "balance"), ("keep", "drain"), ("historical",),
                              "0", 10**6, 1, 10**4, "plain", (10**6, 10**7), LOGIT_TRAIN, "usdt_s0")
    # WHEN: Preparing the study inputs.
    inputs = inputs_from(config, MANIFEST, SNAPSHOT, "0" * 64)
    # THEN: Only positive deposits strictly before the split are participants, once each in
    # (block, log_index) order, and the rounding unit is the manifest's `d`.
    assert inputs.train_gross == (50, 60)
    assert inputs.d == 10**6
    assert inputs.balances == (100, 200)


def test_experiment_config_round_trips_with_a_nested_theta() -> None:
    # GIVEN: A config carrying a nested theta, adoption ratios as fraction strings and tail units.
    config = ExperimentConfig("manifest.json", "651b808f", 25603700, THETA, 5, ("0", "1/40", "1/10"),
                              ("need",), ("double", "balance"), ("keep", "drain"), ("historical",), "0",
                              10**6, 1, 10**4, "plain", (10**6, 10**7), LOGIT_TRAIN, "usdt_s0")
    # WHEN: It is encoded and decoded.
    restored = ExperimentConfig.from_json(config.to_json())
    # THEN: The nested theta, the ratio strings and the tail units survive unchanged.
    assert restored == config
    assert restored.theta == THETA
    assert restored.logit_units == (10**6, 10**7)
    assert Fraction(restored.rhos[2]) == Fraction(1, 10)


def test_experiment_config_rejects_a_legacy_n_key() -> None:
    # GIVEN: A config JSON object that still carries an `n` key.
    config = ExperimentConfig("manifest.json", "651b808f", 25603700, THETA, 5, ("0",), ("need",),
                              ("double",), ("keep",), ("historical",), "0", 10**6, 1, 10**4, "plain",
                              (10**6,), LOGIT_TRAIN, "usdt_s0")
    data = config.to_json()
    data["n"] = 200
    # WHEN: Decoding it.
    # THEN: The leftover key is rejected rather than silently ignored.
    with pytest.raises(SchemaError):
        ExperimentConfig.from_json(data)


def test_theta_of_converts_fraction_strings_exactly() -> None:
    # GIVEN: A theta with a fraction share and a decimal string share.
    spec = ThetaSpec("9/22", "0", "0.2305", "hi", 10**6, "P2 anchors, USDT")
    # WHEN: Converting to the behaviour model.
    theta = theta_of(spec)
    # THEN: The shares are the floats of the exact fractions.
    assert theta.pi == 0.4090909090909091
    assert theta.pi == float(Fraction(9, 22))
    assert theta.w == 0.2305
    assert theta.s == 0.0
    assert theta.g == "hi"


def test_derive_seed_is_stable() -> None:
    # GIVEN: The published derivation of a path seed.
    # WHEN: Deriving it twice.
    first, second = derive_seed("path", "usdt_s0", 200, 0), derive_seed("path", "usdt_s0", 200, 0)
    # THEN: It is deterministic and distinct across replicates.
    assert first == second
    assert first != derive_seed("path", "usdt_s0", 200, 1)
