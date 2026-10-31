"""Regression tests binding the trained-logit features to the rounding unit `d`.

The ETH rounding unit is `d = 0.001 ETH = 10**15 wei`, not one whole ether (`10**18`), so the
divisibility features must divide by `d` and the tail units are `d`-relative.  The stablecoin
case keeps one whole coin as `d` and its feature matrix must not move.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lab.observers import logit
from lab.observers.logit import feature_names, features
from lab.run import grid, run
from lab.schema import (
    Deposit,
    ExperimentConfig,
    LogitTrainSpec,
    Manifest,
    PinError,
    Snapshot,
    ThetaSpec,
    Withdrawal,
    canonical,
    sha256,
)
from lab.study import inputs_from
from lab.train import train

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# ==== synthetic ETH-like data: D is 0.001 ETH, not one whole ether ====

D = 10**15
DECIMALS = 18
STEP = D // 100
QUANTUM = D // 100
THRESHOLD = D
LOGIT_UNITS = (D // 100, D, 10 * D, 100 * D, 1000 * D)
THETA = ThetaSpec("9/22", "0", "0", "hi", 10 * D, "synthetic")
LOGIT_TRAIN = LogitTrainSpec(2, "0.001", "1/4", "3/4", "1/2", "1", "synthetic")
DEPOSIT_VALUES = (5 * D, 8 * D + 1, 12 * D, 20 * D, 40 * D, 33 * D, 60 * D, 15 * D, 7 * D, 25 * D,
                  9 * D + 3, 50 * D)
WITHDRAWAL_BLOCKS = (15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185,
                     195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305)
WITHDRAWAL_GROSS = (2 * D, 3 * D, 5 * D, D, 4 * D + 1, 6 * D, 2 * D + 7, 8 * D, 3 * D + 2, 7 * D,
                    9 * D, 5 * D + 4, 11 * D, 2 * D, 6 * D + 3, 4 * D, 12 * D, 3 * D, 8 * D + 5, 10 * D,
                    2 * D, 5 * D, 13 * D, 7 * D + 1, 4 * D, 9 * D, 6 * D, 15 * D, 3 * D, 11 * D)


def _snapshot() -> Snapshot:
    deposits = tuple(Deposit(10 + 10 * i, 0, "0xaa", "0xdd", str(i + 1), "0xx", value, "0xp")
                     for i, value in enumerate(DEPOSIT_VALUES))
    withdrawals = tuple(Withdrawal(block, 1, "0xbb", gross, 0, "0xrr", None, str(i), "0xnn")
                        for i, (block, gross) in enumerate(zip(WITHDRAWAL_BLOCKS, WITHDRAWAL_GROSS,
                                                               strict=True)))
    return Snapshot(deposits=deposits, withdrawals=withdrawals, ragequits=(), roots=())


def _manifest(digest: str | None) -> Manifest:
    return Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, DECIMALS, D, 1, 1000,
                    "0x" + "00" * 32, digest)


def _config(snapshot_sha256: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "manifest": "manifest.json", "snapshot_sha256": snapshot_sha256, "split_block": 200,
        "theta": THETA, "replicates": 1, "rhos": ("1/2",), "rules": ("need",),
        "m_models": ("double",), "adopters": ("keep",), "observers": ("trained-logit",),
        "combinatorial_tolerance": "0", "threshold": THRESHOLD, "quantum": QUANTUM, "step": STEP,
        "historical": "plain",
        "logit_units": LOGIT_UNITS, "logit_train": LOGIT_TRAIN, "seed_root": "eth_logit_s0",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _setup(tmp_path: Path) -> tuple[str, str, ExperimentConfig]:
    """Write the snapshot, manifest and config; return their paths and the config."""
    snapshot = _snapshot()
    digest = sha256(snapshot)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(digest).to_json()), encoding="utf-8")
    folder = tmp_path / "data" / "manifest"
    folder.mkdir(parents=True)
    (folder / "snapshot.json").write_bytes(canonical(snapshot))
    config = _config(digest)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_json()), encoding="utf-8")
    return str(config_path), str(tmp_path / "data"), config


def _oracle(y: int, balances: Sequence[int], units: Sequence[int],
            whole_unit: int) -> tuple[list[list[float]], list[bool]]:
    """The expected features by explicit integer arithmetic, independent of the numpy implementation."""
    matrix: list[list[float]] = []
    eligible: list[bool] = []
    for b in balances:
        row = [0.0] * (4 + len(units))
        ok = b >= y
        eligible.append(ok)
        if ok:
            row[0] = math.log(b) - math.log(y)
            row[1] = 1.0 if b == y else 0.0
            for k, u in enumerate(units):
                residue = y % u
                row[2 + k] = 1.0 if residue != 0 and residue == b % u else 0.0
            whole = b % whole_unit == 0
            row[2 + len(units)] = 1.0 if whole else 0.0
            row[3 + len(units)] = 1.0 if whole and y % whole_unit == 0 else 0.0
        matrix.append(row)
    return matrix, eligible


# ==== the rounding unit reaches the logit input ====


def test_inputs_from_binds_whole_unit_to_the_manifest_rounding_unit() -> None:
    # GIVEN: An ETH-like manifest whose rounding unit D is 0.001 ether, not one whole ether.
    manifest = _manifest(None)
    # WHEN: Preparing the study inputs.
    inputs = inputs_from(_config("0" * 64), manifest, _snapshot(), "0" * 64)
    # THEN: The logit amount unit is the manifest's D, not 10**decimals.
    assert inputs.d == D
    assert inputs.d != 10**DECIMALS


def test_stablecoin_whole_unit_stays_one_whole_coin() -> None:
    # GIVEN: A stablecoin manifest whose rounding unit D is one whole coin.
    manifest = Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, 6, 10**6, 1, 1000,
                        "0x" + "00" * 32, None)
    # WHEN: Preparing the study inputs.
    inputs = inputs_from(_config("0" * 64, step=10**4), manifest, _snapshot(), "0" * 64)
    # THEN: The logit unit is still one whole coin, so the stablecoin features cannot move.
    assert inputs.d == 10**6 == 10 ** manifest.decimals


# ==== the divisibility features use D ====


def test_features_judge_d_multiples_not_one_whole_ether() -> None:
    # GIVEN: Balances that are multiples of D but not of one ether, and one that is a whole ether.
    y = 7 * D
    balances = [7 * D, 7 * D + 1, 9 * D, 9 * D + 1, 1000 * D, 10**18 + 1]
    # WHEN: Building the feature matrix.
    matrix, _eligible = features(y, balances, LOGIT_UNITS, D)
    # THEN: `b_whole` marks multiples of D (9*D qualifies; one whole ether does not, since it is not a
    # multiple of D when D is 0.001 ether) and y itself is a D multiple.
    assert matrix[:, -2].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert matrix[:, -1].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_features_match_a_pure_integer_oracle() -> None:
    # GIVEN: A public amount and balances with mixed tails around D.
    y = 7 * D + 5
    balances = [5 * D, 7 * D + 5, 7 * D + 6, 9 * D, 9 * D + 1, 1000 * D, 10**18 + 1]
    # WHEN: Building the feature matrix.
    matrix, eligible = features(y, balances, LOGIT_UNITS, D)
    # THEN: Every column matches the explicit integer oracle.
    expected, expected_eligible = _oracle(y, balances, LOGIT_UNITS, D)
    assert eligible.tolist() == expected_eligible
    assert matrix[:, 1:].tolist() == [row[1:] for row in expected]
    assert matrix[:, 0] == pytest.approx([row[0] for row in expected])


def test_features_are_exact_for_adjacent_wei_above_2_53() -> None:
    # GIVEN: Amounts above 2**53 whose floats compare equal.
    y = 10**21 + 13
    balances = [y - 1, y, y + 1, 2 * y, 2 * y + 1]
    # WHEN: Building the feature matrix.
    matrix, eligible = features(y, balances, LOGIT_UNITS, D)
    # THEN: Eligibility and the exact-equality feature separate adjacent wei.
    assert eligible.tolist() == [False, True, True, True, True]
    assert matrix[:, 1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert np.isfinite(matrix).all()
    # AND the exact integer comparison is load-bearing: as floats the neighbours are equal.
    assert float(y) == float(y + 1)


def test_features_are_invariant_when_amounts_units_and_d_scale_together() -> None:
    # GIVEN: The same amounts expressed in a unit k times finer, with D and the tail units scaled too.
    k = 1000
    y = 7 * D
    balances = [5 * D, 7 * D, 7 * D + 1, 9 * D + 4, 20 * D]
    # WHEN: Building both feature matrices.
    base, base_eligible = features(y, balances, LOGIT_UNITS, D)
    scaled, scaled_eligible = features(y * k, [b * k for b in balances], [u * k for u in LOGIT_UNITS], D * k)
    # THEN: The divisibility and tail features are unchanged and log_ratio is the same ratio.
    assert scaled_eligible.tolist() == base_eligible.tolist()
    assert scaled[:, 1:].tolist() == base[:, 1:].tolist()
    assert scaled[:, 0] == pytest.approx(base[:, 0])


# ==== the stablecoin matrix and predictions are unchanged ====


def test_stablecoin_features_and_predictions_are_unchanged() -> None:
    # GIVEN: A stablecoin pool where one whole coin equals D, and its tail axis.
    d = 10**6
    units = (d // 100, d, 10 * d, 100 * d, 1000 * d)
    y = 20 * d
    balances = [3 * d, 7 * d + 1, 20 * d, 20 * d + 1, 1001 * d]
    # WHEN: Building the feature matrix.
    matrix, eligible = features(y, balances, units, d)
    # THEN: It equals the frozen oracle and b_whole marks whole-coin multiples.
    expected, expected_eligible = _oracle(y, balances, units, d)
    assert eligible.tolist() == expected_eligible
    assert matrix[:, 1:].tolist() == [row[1:] for row in expected]
    assert matrix[:, -2].tolist() == [0.0, 0.0, 1.0, 0.0, 1.0]
    # AND the observer's posterior is the softmax of those same rows over the eligible deposits.
    beta = np.linspace(-0.4, 0.4, matrix.shape[1])
    posterior = logit.predict(beta, matrix, eligible)
    logits = np.array([sum(row[j] * beta[j] for j in range(len(row))) if ok else -np.inf
                       for row, ok in zip(expected, expected_eligible, strict=True)])
    weights = np.exp(logits - logits[expected_eligible].max())
    weights[~eligible] = 0.0
    assert posterior == pytest.approx(weights / weights.sum())
    assert (posterior[~eligible] == 0.0).all()


# ==== the trained artifact pins D and the tail units, and the runner enforces them ====


def test_train_pins_the_rounding_unit_and_units_in_the_artifact(tmp_path: Path) -> None:
    # GIVEN: A synthetic ETH-like snapshot and config.
    config_path, data, _config_obj = _setup(tmp_path)
    out = tmp_path / "beta"
    # WHEN: Training the observer.
    train(config_path, data=data, out=str(out))
    # THEN: The artifact records D (not one whole ether) and the tail units.
    payload = json.loads((out / "logit_beta.json").read_text(encoding="utf-8"))
    assert payload["whole_unit"] == D
    assert payload["whole_unit"] != 10**DECIMALS
    assert payload["logit_units"] == list(LOGIT_UNITS)
    # WHEN: The runner reads that artifact back.
    run_out = tmp_path / "run"
    run(config_path, data=data, out=str(run_out), workers=1, logit_beta_path=str(out / "logit_beta.json"))
    # THEN: The trained-logit observer scored every path.
    for spec in grid(_config_obj):
        assert (run_out / spec.key / "path.json").is_file()


def test_run_rejects_a_bare_beta_array_without_unit_provenance(tmp_path: Path) -> None:
    # GIVEN: An artifact that is a bare coefficient array with no D or tail-unit record.
    config_path, data, _config_obj = _setup(tmp_path)
    beta = (0.1,) * len(feature_names(LOGIT_UNITS))
    artifact = tmp_path / "beta.json"
    artifact.write_text(json.dumps(list(beta)), encoding="utf-8")
    # WHEN: Running with it.
    # THEN: It is rejected rather than read blindly.
    with pytest.raises(PinError):
        run(config_path, data=data, out=str(tmp_path / "out"), workers=1, logit_beta_path=str(artifact))


def test_run_rejects_an_artifact_missing_the_unit_provenance(tmp_path: Path) -> None:
    # GIVEN: An artifact carrying coefficients but no whole_unit or logit_units.
    config_path, data, _config_obj = _setup(tmp_path)
    beta = (0.1,) * len(feature_names(LOGIT_UNITS))
    artifact = tmp_path / "beta.json"
    artifact.write_text(json.dumps({"beta": list(beta), "commit": "deadbeef"}), encoding="utf-8")
    # WHEN: Running with it.
    # THEN: The missing provenance is rejected.
    with pytest.raises(PinError):
        run(config_path, data=data, out=str(tmp_path / "out"), workers=1, logit_beta_path=str(artifact))
