"""Regression tests that pin the trained-logit artifact to its rounding unit, tail axis and snapshot.

:func:`lab.run.run` must read an artifact's `whole_unit`/`logit_units`/`snapshot_sha256`
exactly: a fractional or boolean unit cannot be truncated into a match, the tail axis must have
the configured length and order, the coefficients must be a finite array of the feature width, and
an artifact fitted on another snapshot is rejected.  Every malformed field raises
:class:`~lab.schema.PinError`, never a bare `TypeError`/`ValueError`.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from lab.observers.logit import feature_names
from lab.run import run
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
from lab.train import train_logit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from lab.study import StudyInputs

# ==== synthetic ETH-like data: D is 0.001 ETH = 10**15 wei, not one whole ether ====

D = 10**15
DECIMALS = 18
STEP = D // 100
QUANTUM = D // 100
THRESHOLD = D
LOGIT_UNITS = (D // 100, D, 10 * D, 100 * D, 1000 * D)
THETA = ThetaSpec("9/22", "0", "0", "hi", 10 * D, "synthetic")
LOGIT_TRAIN = LogitTrainSpec(2, "0.001", "1/4", "3/4", "1/2", "1", "synthetic")
PAIRS = (("need", "double"),)
RHOS = (Fraction(1, 2),)
ADOPTERS = ("keep", "drain")
DEPOSIT_VALUES = (5 * D, 8 * D + 1, 12 * D, 20 * D, 40 * D, 33 * D, 60 * D, 15 * D, 7 * D, 25 * D,
                  9 * D + 3, 50 * D)
WITHDRAWAL_BLOCKS = (15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185,
                     195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305)
WITHDRAWAL_GROSS = (2 * D, 3 * D, 5 * D, D, 4 * D + 1, 6 * D, 2 * D + 7, 8 * D, 3 * D + 2, 7 * D,
                    9 * D, 5 * D + 4, 11 * D, 2 * D, 6 * D + 3, 4 * D, 12 * D, 3 * D, 8 * D + 5, 10 * D,
                    2 * D, 5 * D, 13 * D, 7 * D + 1, 4 * D, 9 * D, 6 * D, 15 * D, 3 * D, 11 * D)
BETA = (0.1,) * len(feature_names(LOGIT_UNITS))


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


def _setup(tmp_path: Path, **overrides: object) -> tuple[str, str, str]:
    """Write the snapshot, manifest and config; return the config path, the data dir and the digest."""
    snapshot = _snapshot()
    digest = sha256(snapshot)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(digest).to_json()), encoding="utf-8")
    folder = tmp_path / "data" / "manifest"
    folder.mkdir(parents=True)
    (folder / "snapshot.json").write_bytes(canonical(snapshot))
    config = _config(digest, **overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_json()), encoding="utf-8")
    return str(config_path), str(tmp_path / "data"), digest


def _inputs() -> StudyInputs:
    snapshot = _snapshot()
    return inputs_from(_config("0" * 64), _manifest(sha256(snapshot)), snapshot, sha256(snapshot))


def _artifact(beta: Sequence[object], whole_unit: object, units: Sequence[object],
              snapshot_sha256: object) -> dict:
    return {"beta": list(beta), "whole_unit": whole_unit, "logit_units": list(units),
            "feature_names": list(feature_names(LOGIT_UNITS)), "snapshot_sha256": snapshot_sha256}


def _write(tmp_path: Path, artifact: dict) -> str:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return str(path)


def _run(tmp_path: Path, config_path: str, data: str, artifact: dict) -> dict:
    return run(config_path, data=data, out=str(tmp_path / "out"), workers=1,
               logit_beta_path=_write(tmp_path, artifact))


# ==== train_logit records the rounding unit of its inputs ====


def test_train_logit_records_the_rounding_unit_of_its_inputs() -> None:
    # GIVEN: The study inputs, whose D is the manifest's rounding unit.
    inputs = _inputs()
    # WHEN: Training on two paths.
    result = train_logit(inputs, paths=2, tag="tag_a", pairs=PAIRS, rhos=RHOS, adopters=ADOPTERS)
    # THEN: The fit runs and the artifact records that D.
    assert result["whole_unit"] == inputs.d == D
    assert result["n_paths"] == 2
    assert len(result["beta"]) == len(feature_names(LOGIT_UNITS))


# ==== run rejects a fractional or boolean unit instead of truncating it ====


def test_run_rejects_a_float_whole_unit_that_truncates_to_d(tmp_path: Path) -> None:
    # GIVEN: An artifact whose whole_unit is the float 1e15, which int() would truncate to D.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, float(D), LOGIT_UNITS, digest)
    # WHEN: Running with it.
    # THEN: The fractional unit is rejected rather than silently truncated.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_float_tail_unit_that_truncates_to_the_configured_axis(tmp_path: Path) -> None:
    # GIVEN: An artifact whose first tail unit is a float that truncates to the configured value.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, [float(LOGIT_UNITS[0]), *LOGIT_UNITS[1:]], digest)
    # WHEN: Running with it.
    # THEN: The fractional tail unit is rejected.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_bool_tail_unit_that_would_truncate_to_one(tmp_path: Path) -> None:
    # GIVEN: A config whose first tail unit is 1, and an artifact that writes it as the boolean true.
    units = (1, D, 10 * D, 100 * D, 1000 * D)
    config_path, data, digest = _setup(tmp_path, logit_units=units)
    artifact = _artifact(BETA, D, [True, *units[1:]], digest)
    # WHEN: Running with it.
    # THEN: The boolean is rejected, not read as the integer 1.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_accepts_decimal_string_units_consistent_with_the_amount_schema(tmp_path: Path) -> None:
    # GIVEN: An artifact whose whole_unit and tail units are Amount-style decimal strings.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, str(D), [str(u) for u in LOGIT_UNITS], digest)
    # WHEN: Running with it.
    summary = _run(tmp_path, config_path, data, artifact)
    # THEN: The decimal strings are accepted like any other integer amount.
    assert summary["written"] == summary["paths"] > 0


# ==== run rejects a bad coefficient array ====


def test_run_rejects_a_beta_of_the_wrong_width(tmp_path: Path) -> None:
    # GIVEN: An artifact whose beta is one coefficient shorter than the feature width.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA[:-1], D, LOGIT_UNITS, digest)
    # WHEN: Running with it.
    # THEN: The wrong width is rejected up front.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_non_finite_coefficient(tmp_path: Path) -> None:
    # GIVEN: An artifact with a NaN coefficient (JSON permits it).
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact([*BETA[:-1], float("nan")], D, LOGIT_UNITS, digest)
    # WHEN: Running with it.
    # THEN: The non-finite coefficient is rejected.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_non_numeric_coefficient(tmp_path: Path) -> None:
    # GIVEN: An artifact whose first coefficient is a string.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(["0.1", *BETA[1:]], D, LOGIT_UNITS, digest)
    # WHEN: Running with it.
    # THEN: The non-numeric coefficient is rejected.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_beta_that_is_not_an_array(tmp_path: Path) -> None:
    # GIVEN: An artifact whose beta is a bare number.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, LOGIT_UNITS, digest)
    artifact["beta"] = 0.1
    # WHEN: Running with it.
    # THEN: The shape error is a PinError, not a TypeError.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


# ==== run rejects malformed provenance fields as PinError ====


def test_run_rejects_a_whole_unit_that_is_null(tmp_path: Path) -> None:
    # GIVEN: An artifact whose whole_unit is null.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, None, LOGIT_UNITS, digest)
    # WHEN: Running with it.
    # THEN: The malformed field is a PinError, not a TypeError.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_logit_units_that_is_not_an_array(tmp_path: Path) -> None:
    # GIVEN: An artifact whose logit_units is a bare number.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, LOGIT_UNITS, digest)
    artifact["logit_units"] = D
    # WHEN: Running with it.
    # THEN: The shape error is a PinError, not a TypeError.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_rejects_a_tail_axis_of_the_wrong_length(tmp_path: Path) -> None:
    # GIVEN: An artifact whose tail axis is one unit shorter than the config's.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, LOGIT_UNITS[:-1], digest)
    # WHEN: Running with it.
    # THEN: The mismatched axis is rejected.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


# ==== run rejects an artifact fitted on another snapshot ====


def test_run_rejects_an_artifact_fitted_on_another_snapshot(tmp_path: Path) -> None:
    # GIVEN: An artifact with the right D and tail axis but another pool's snapshot hash.
    config_path, data, _digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, LOGIT_UNITS, "f" * 64)
    # WHEN: Running with it.
    # THEN: It is rejected, so another pool's same-axis coefficients cannot be mixed in.
    with pytest.raises(PinError):
        _run(tmp_path, config_path, data, artifact)


def test_run_accepts_an_artifact_that_pins_this_snapshot(tmp_path: Path) -> None:
    # GIVEN: An artifact whose snapshot hash is this run's own.
    config_path, data, digest = _setup(tmp_path)
    artifact = _artifact(BETA, D, LOGIT_UNITS, digest)
    # WHEN: Running with it.
    summary = _run(tmp_path, config_path, data, artifact)
    # THEN: It is accepted and every path is scored.
    assert summary["written"] == summary["paths"] > 0
