"""Tests for the trained-logit observer trainer, on a synthetic snapshot."""

from __future__ import annotations

import json
from fractions import Fraction
from typing import TYPE_CHECKING

from lab import sim
from lab.observers.logit import feature_names
from lab.run import grid, run
from lab.schema import (
    Deposit,
    ExperimentConfig,
    LogitTrainSpec,
    Manifest,
    Snapshot,
    ThetaSpec,
    Withdrawal,
    canonical,
    sha256,
)
from lab.score import read_rows
from lab.study import inputs_from
from lab.train import train, train_logit

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from lab.study import StudyInputs

# ==== synthetic data ====

D = 10
STEP = 1
QUANTUM = 10
THRESHOLD = 50
UNIT = 10
LOGIT_UNITS = (10,)
THETA = ThetaSpec("9/22", "0", "0", "hi", UNIT, "synthetic")
LOGIT_TRAIN = LogitTrainSpec(3, "0.001", "1/4", "3/4", "1/2", "1", "synthetic")
PAIRS = (("need", "double"),)
RHOS = (Fraction(1, 2),)
ADOPTERS = ("keep", "drain")
DEPOSIT_VALUES = (300, 450, 600, 350, 500, 700, 250, 800, 400, 550, 650, 300)
WITHDRAWAL_BLOCKS = (15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185,
                     195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305)
WITHDRAWAL_GROSS = (40, 60, 50, 80, 30, 70, 90, 45, 55, 65, 35, 75, 85, 25, 95, 50, 60, 40, 70,
                    110, 120, 80, 100, 90, 130, 60, 70, 140, 100, 80)


def _snapshot() -> Snapshot:
    deposits = tuple(Deposit(10 + 10 * i, 0, "0xaa", "0xdd", str(i + 1), "0xx", value, "0xp")
                     for i, value in enumerate(DEPOSIT_VALUES))
    withdrawals = tuple(Withdrawal(block, 1, "0xbb", gross, 0, "0xrr", None, str(i), "0xnn")
                        for i, (block, gross) in enumerate(zip(WITHDRAWAL_BLOCKS, WITHDRAWAL_GROSS,
                                                               strict=True)))
    return Snapshot(deposits=deposits, withdrawals=withdrawals, ragequits=(), roots=())


def _manifest(digest: str) -> Manifest:
    return Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, 0, D, 1, 1000,
                    "0x" + "00" * 32, digest)


def _config(snapshot_sha256: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "manifest": "manifest.json", "snapshot_sha256": snapshot_sha256, "split_block": 200,
        "theta": THETA, "replicates": 1, "rhos": ("0", "1/2"), "rules": ("need",),
        "m_models": ("double",), "adopters": ("keep", "drain"), "observers": ("trained-logit",),
        "combinatorial_tolerance": "0", "threshold": THRESHOLD, "quantum": QUANTUM, "step": STEP,
        "historical": "plain",
        "logit_units": LOGIT_UNITS, "logit_train": LOGIT_TRAIN, "seed_root": "synthetic_s0",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _setup(tmp_path: Path, **overrides: object) -> tuple[str, str, str, ExperimentConfig]:
    """Write the snapshot, manifest and config; return their paths, the digest and the config."""
    snapshot = _snapshot()
    digest = sha256(snapshot)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(digest).to_json()), encoding="utf-8")
    folder = tmp_path / "data" / "manifest"
    folder.mkdir(parents=True)
    (folder / "snapshot.json").write_bytes(canonical(snapshot))
    config = _config(overrides.pop("snapshot_sha256", digest), **overrides)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_json()), encoding="utf-8")
    return str(config_path), str(tmp_path / "data"), digest, config


def _inputs(**overrides: object) -> StudyInputs:
    snapshot = _snapshot()
    return inputs_from(_config("0" * 64, **overrides), _manifest(sha256(snapshot)), snapshot, sha256(snapshot))


# ==== inner layer ====


def test_train_logit_reproduces_for_a_fixed_tag() -> None:
    # GIVEN: The study inputs and a fixed config.
    inputs = _inputs()
    # WHEN: Training twice with the same tag.
    first = train_logit(inputs, paths=LOGIT_TRAIN.paths, tag="tag_a", pairs=PAIRS, rhos=RHOS,
                        adopters=ADOPTERS)
    second = train_logit(inputs, paths=LOGIT_TRAIN.paths, tag="tag_a", pairs=PAIRS, rhos=RHOS,
                         adopters=ADOPTERS)
    # THEN: Every coefficient reproduces exactly and the path count is the requested count.
    assert first["beta"] == second["beta"]
    assert first["n_groups"] == second["n_groups"]
    assert first["n_paths"] == LOGIT_TRAIN.paths


def test_train_logit_tag_moves_the_coefficients() -> None:
    # GIVEN: The same inputs under two tags.
    inputs = _inputs()
    # WHEN: Training once per tag.
    a = train_logit(inputs, paths=LOGIT_TRAIN.paths, tag="tag_a", pairs=PAIRS, rhos=RHOS, adopters=ADOPTERS)
    b = train_logit(inputs, paths=LOGIT_TRAIN.paths, tag="tag_b", pairs=PAIRS, rhos=RHOS, adopters=ADOPTERS)
    # THEN: The differently seeded draws give different coefficients.
    assert a["beta"] != b["beta"]


def test_train_logit_group_count_is_the_simulated_event_total(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: The study inputs and a counter wrapped around the simulator.
    inputs = _inputs()
    total = 0
    real = sim.simulate

    def counting(model: object, balances: object, kinds: object, seed: int) -> tuple:
        nonlocal total
        events = real(model, balances, kinds, seed)
        total += len(events)
        return events

    monkeypatch.setattr(sim, "simulate", counting)
    # WHEN: Training.
    result = train_logit(inputs, paths=LOGIT_TRAIN.paths, tag="tag_a", pairs=PAIRS, rhos=RHOS,
                         adopters=ADOPTERS)
    # THEN: Every simulated event became exactly one conditional-logit group.
    assert total > 0
    assert result["n_groups"] == total


def test_train_logit_beta_width_matches_the_feature_names() -> None:
    # GIVEN: A config with two logit tail units.
    inputs = _inputs(logit_units=(10, 100))
    # WHEN: Training on two paths.
    result = train_logit(inputs, paths=2, tag="tag_a", pairs=PAIRS, rhos=RHOS, adopters=ADOPTERS)
    # THEN: The coefficients line up column for column with the feature names.
    assert result["feature_names"] == list(feature_names((10, 100)))
    assert len(result["beta"]) == len(result["feature_names"]) == 6


# ==== outer layer ====


def test_train_writes_a_beta_file_the_runner_accepts(tmp_path: Path) -> None:
    # GIVEN: A synthetic snapshot and a config whose only observer is the trained-logit observer.
    config_path, data, digest, config = _setup(tmp_path)
    out = tmp_path / "beta"
    # WHEN: Training the observer.
    summary = train(config_path, data=data, out=str(out))
    # THEN: `logit_beta.json` carries the fit, the commit and the snapshot hash.
    beta_file = out / "logit_beta.json"
    payload = json.loads(beta_file.read_text(encoding="utf-8"))
    assert payload["snapshot_sha256"] == digest
    assert payload["feature_names"] == list(feature_names(LOGIT_UNITS))
    assert len(payload["beta"]) == len(feature_names(LOGIT_UNITS))
    assert payload["commit"]
    assert payload["n_paths"] == LOGIT_TRAIN.paths == summary["n_paths"]
    assert payload["n_groups"] == summary["n_groups"] > 0
    # WHEN: The runner reads that file as its logit_beta input.
    run_out = tmp_path / "run"
    run(config_path, data=data, out=str(run_out), workers=1, logit_beta_path=str(beta_file))
    # THEN: The trained-logit observer scored every simulated path.
    for spec in grid(config):
        rows = read_rows(run_out / spec.key / "rows.csv.gz")
        assert {row["observer"] for row in rows} == {"trained-logit"}
        assert len(rows) > 0
