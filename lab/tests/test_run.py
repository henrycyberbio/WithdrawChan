"""End-to-end tests for the two-layer runner, on a synthetic snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from fractions import Fraction
from typing import TYPE_CHECKING

import pytest

from lab.observers.logit import feature_names
from lab.run import PathSpec, grid, run, run_path
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
from lab.score import read_rows
from lab.sim import DRAIN, KEEP, NON
from lab.study import inputs_from

if TYPE_CHECKING:
    from pathlib import Path

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
DEPOSIT_VALUES = (300, 450, 600, 350, 500, 700, 250, 800, 400, 550, 650, 300)
WITHDRAWAL_BLOCKS = (15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155, 165, 175, 185,
                     195, 205, 215, 225, 235, 245, 255, 265, 275, 285, 295, 305)
WITHDRAWAL_GROSS = (40, 60, 50, 80, 30, 70, 90, 45, 55, 65, 35, 75, 85, 25, 95, 50, 60, 40, 70,
                    110, 120, 80, 100, 90, 130, 60, 70, 140, 100, 80)
LOGIT_BETA = (0.1, -0.2, 0.3, 0.0, 0.05)


def _snapshot() -> Snapshot:
    deposits = tuple(Deposit(10 + 10 * i, 0, "0xaa", "0xdd", str(i + 1), "0xx", value, "0xp")
                     for i, value in enumerate(DEPOSIT_VALUES))
    withdrawals = tuple(Withdrawal(block, 1, "0xbb", gross, 0, "0xrr", None, str(i), "0xnn")
                        for i, (block, gross) in enumerate(zip(WITHDRAWAL_BLOCKS, WITHDRAWAL_GROSS,
                                                               strict=True)))
    return Snapshot(deposits=deposits, withdrawals=withdrawals, ragequits=(), roots=())


def _manifest(digest: str, d: int = D) -> Manifest:
    return Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, 0, d, 1, 1000,
                    "0x" + "00" * 32, digest)


def _config(snapshot_sha256: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "manifest": "manifest.json", "snapshot_sha256": snapshot_sha256, "split_block": 200,
        "theta": THETA, "replicates": 2, "rhos": ("0", "1/2"), "rules": ("need",),
        "m_models": ("double",), "adopters": ("keep", "drain"), "observers": ("historical", "combinatorial"),
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


# ==== grid ====


def test_grid_collapses_rho_zero_to_one_spec_per_replicate() -> None:
    # GIVEN: A config whose axis lists include the zero adoption ratio.
    config = _config("0" * 64)
    # WHEN: Building the grid.
    specs = grid(config)
    # THEN: `rho = 0` gives exactly one fixed spec per replicate, and every other axis is swept.
    assert [s for s in specs if s.rho == 0] == [PathSpec(0, Fraction(0), "need", "double", "keep"),
                                                PathSpec(1, Fraction(0), "need", "double", "keep")]
    assert len(specs) == 2 * (1 + 1 * 1 * 2)


def test_grid_emits_the_need_rule_only_with_the_first_m_model() -> None:
    # GIVEN: A config whose rules and maximum models would cross-product, with the need rule among them.
    config = _config("0" * 64, replicates=1, rhos=("1/2",), rules=("need", "decstep"),
                     m_models=("double", "balance"), adopters=("keep",))
    # WHEN: Building the grid.
    specs = grid(config)
    # THEN: Need runs once, on the first maximum model, and the other rule sweeps both maximum models, in order.
    assert [(s.rule, s.m_model) for s in specs] == [("need", "double"), ("decstep", "double"),
                                                    ("decstep", "balance")]


def test_path_key_names_every_axis() -> None:
    # GIVEN: A spec on a non-trivial corner of the grid.
    spec = PathSpec(7, Fraction(1, 10), "decstep", "balance", "drain")
    # WHEN: Reading its key.
    # THEN: The key carries the replicate, the ratio, the rule, the maximum model and the kind.
    assert spec.key == "r007_rho1-10_decstep_balance_drain"


# ==== outer layer ====


def test_run_writes_every_path_rows_and_run_json(tmp_path: Path) -> None:
    # GIVEN: A synthetic snapshot and a config that includes the trained-logit observer.
    config_path, data, digest, config = _setup(tmp_path, observers=("historical", "trained-logit", "combinatorial"))
    beta_path = tmp_path / "beta.json"
    beta_path.write_text(json.dumps({"beta": LOGIT_BETA, "whole_unit": D, "logit_units": list(LOGIT_UNITS)}),
                         encoding="utf-8")
    out = tmp_path / "out"
    # WHEN: Running the grid.
    summary = run(config_path, data=data, out=str(out), workers=1, logit_beta_path=str(beta_path))
    # THEN: Each path directory holds its rows and a summary whose event count matches the rows.
    for spec in grid(config):
        folder = out / spec.key
        assert (folder / "rows.csv.gz").is_file()
        content = json.loads((folder / "path.json").read_text(encoding="utf-8"))
        rows = read_rows(folder / "rows.csv.gz")
        assert len(rows) == content["n_events"] * len(config.observers)
        assert {row["observer"] for row in rows} == set(config.observers)
    # THEN: `run.json` records the participant count, the pinned hash and the path keys.
    report = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert report["N"] == len(DEPOSIT_VALUES)
    assert report["snapshot_sha256"] == digest
    assert report["paths"] == [spec.key for spec in grid(config)]
    assert summary["paths"] == len(grid(config))


def test_run_is_independent_of_worker_count(tmp_path: Path) -> None:
    # GIVEN: The same config run with one worker and with two.
    config_path, data, _digest, config = _setup(tmp_path)
    one, two = tmp_path / "one", tmp_path / "two"
    run(config_path, data=data, out=str(one), workers=1, logit_beta_path=None)
    run(config_path, data=data, out=str(two), workers=2, logit_beta_path=None)
    # THEN: Every path's gzipped rows are byte-identical.
    for spec in grid(config):
        left = (one / spec.key / "rows.csv.gz").read_bytes()
        right = (two / spec.key / "rows.csv.gz").read_bytes()
        assert left == right


def test_run_skips_finished_paths_on_resume(tmp_path: Path) -> None:
    # GIVEN: A finished run whose path files are stamped to a known old time.
    config_path, data, _digest, config = _setup(tmp_path)
    out = tmp_path / "out"
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    folder = out / grid(config)[0].key
    before = (folder / "rows.csv.gz").read_bytes()
    os.utime(folder / "rows.csv.gz", (1_000_000, 1_000_000))
    os.utime(folder / "path.json", (1_000_000, 1_000_000))
    # WHEN: Running again into the same directory.
    summary = run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    # THEN: The finished path was skipped, not rewritten.
    assert summary["written"] == 0
    assert (folder / "rows.csv.gz").stat().st_mtime_ns == 1_000_000 * 10**9
    assert (folder / "path.json").stat().st_mtime_ns == 1_000_000 * 10**9
    assert (folder / "rows.csv.gz").read_bytes() == before


def test_run_records_the_manifest_sha256(tmp_path: Path) -> None:
    # GIVEN: A manifest file, its snapshot and a config.
    config_path, data, _digest, _config_obj = _setup(tmp_path)
    out = tmp_path / "out"
    # WHEN: Running the grid.
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    # THEN: `run.json` records the manifest file's own digest, next to the snapshot's.
    report = json.loads((out / "run.json").read_text(encoding="utf-8"))
    manifest_bytes = (tmp_path / "manifest.json").read_bytes()
    assert report["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()


def test_run_refuses_to_resume_into_a_run_of_another_config(tmp_path: Path) -> None:
    # GIVEN: A finished run and a config that scores a different observer list.
    config_path, data, digest, _config_obj = _setup(tmp_path)
    out = tmp_path / "out"
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    before = {path.name for path in out.iterdir()}
    other = tmp_path / "other.json"
    other.write_text(json.dumps(_config(digest, observers=("historical",)).to_json()), encoding="utf-8")
    # WHEN: Rerunning that other config into the same output directory.
    # THEN: It fails before touching a path, naming the config key that differs.
    with pytest.raises(ValueError, match="config.observers"):
        run(str(other), data=data, out=str(out), workers=1, logit_beta_path=None)
    assert {path.name for path in out.iterdir()} == before


def test_run_refuses_to_resume_when_the_manifest_changed(tmp_path: Path) -> None:
    # GIVEN: A finished run whose manifest is then rewritten with another rounding unit.
    config_path, data, digest, _config_obj = _setup(tmp_path)
    out = tmp_path / "out"
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(digest, d=D * 10).to_json()), encoding="utf-8")
    # WHEN: Rerunning into the same output directory.
    # THEN: The recorded manifest digest no longer matches, so it fails before any path.
    with pytest.raises(ValueError, match="manifest_sha256"):
        run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)


def test_run_refuses_to_resume_when_the_recorded_snapshot_hash_differs(tmp_path: Path) -> None:
    # GIVEN: A finished run whose `run.json` records another pinned snapshot hash.
    config_path, data, _digest, _config_obj = _setup(tmp_path)
    out = tmp_path / "out"
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    (out / "run.json").write_text(json.dumps({**record, "snapshot_sha256": "0" * 64}), encoding="utf-8")
    # WHEN: Rerunning into the same output directory.
    # THEN: The recorded snapshot hash is compared too, and the mismatch is named.
    with pytest.raises(ValueError, match="snapshot_sha256"):
        run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)


def test_run_resumes_an_interrupted_run_without_a_run_json(tmp_path: Path) -> None:
    # GIVEN: A run whose `run.json` was never written, as an interrupted run leaves it.
    config_path, data, _digest, _config_obj = _setup(tmp_path)
    out = tmp_path / "out"
    run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    (out / "run.json").unlink()
    # WHEN: Rerunning it.
    summary = run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    # THEN: Nothing is unwritten and nothing is refused; the record is written again.
    assert summary["written"] == 0
    assert (out / "run.json").is_file()


def test_run_rejects_the_trained_logit_observer_without_an_artifact(tmp_path: Path) -> None:
    # GIVEN: A config that lists the trained-logit observer, and no coefficients file.
    config_path, data, _digest, _config_obj = _setup(tmp_path, observers=("historical", "trained-logit"))
    out = tmp_path / "out"
    # WHEN: Running the grid.
    # THEN: It fails immediately with a readable message, before writing any path.
    with pytest.raises(ValueError, match="trained-logit"):
        run(config_path, data=data, out=str(out), workers=1, logit_beta_path=None)
    assert not out.exists() or not any(out.iterdir())


def test_run_rejects_a_config_with_a_wrong_snapshot_hash(tmp_path: Path) -> None:
    # GIVEN: A config whose snapshot_sha256 does not match the snapshot.
    config_path, data, _digest, _config_obj = _setup(tmp_path, snapshot_sha256="0" * 64)
    # WHEN: Running it.
    # THEN: The pin check raises before any path is simulated.
    with pytest.raises(PinError):
        run(config_path, data=data, out=str(tmp_path / "out"), workers=1, logit_beta_path=None)


def test_run_accepts_a_beta_file_in_object_form(tmp_path: Path) -> None:
    # GIVEN: A config with the trained-logit observer and the same coefficients as a minimal object
    # and as an object that also carries the feature names, commit and snapshot hash.
    # (Regression: a bare coefficient array is now rejected, so the artifact must pin whole_unit and logit_units.)
    config_path, data, _digest, config = _setup(tmp_path, observers=("trained-logit",))
    minimal = {"beta": LOGIT_BETA, "whole_unit": D, "logit_units": list(LOGIT_UNITS)}
    extra = {**minimal, "feature_names": list(feature_names(LOGIT_UNITS)), "commit": "deadbeef",
             "snapshot_sha256": _digest}
    minimal_path = tmp_path / "minimal.json"
    extra_path = tmp_path / "extra.json"
    minimal_path.write_text(json.dumps(minimal), encoding="utf-8")
    extra_path.write_text(json.dumps(extra), encoding="utf-8")
    # WHEN: Running once with each form.
    minimal_out, extra_out = tmp_path / "minimal", tmp_path / "extra"
    run(config_path, data=data, out=str(minimal_out), workers=1, logit_beta_path=str(minimal_path))
    run(config_path, data=data, out=str(extra_out), workers=1, logit_beta_path=str(extra_path))
    # THEN: The extra provenance fields do not change the scores.
    for spec in grid(config):
        assert (extra_out / spec.key / "rows.csv.gz").read_bytes() == (
            minimal_out / spec.key / "rows.csv.gz").read_bytes()


# ==== inner layer ====


def test_run_path_reproduces_and_the_seed_root_moves_the_events() -> None:
    # GIVEN: The study inputs and a spec.
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    inputs = _inputs(seed_root="root_a")
    # WHEN: The same path is run twice.
    first, rows_a = run_path(spec, inputs, None)
    second, rows_b = run_path(spec, inputs, None)
    # THEN: It reproduces exactly.
    assert rows_a == rows_b
    assert (first["n_events"], first["K"], first["N"]) == (second["n_events"], second["K"], second["N"])
    # WHEN: The seed root changes.
    _moved, rows_c = run_path(spec, _inputs(seed_root="root_b"), None)
    # THEN: The simulated events differ.
    assert rows_c != rows_a


def test_adopter_kind_flag_shows_up_in_the_rows() -> None:
    # GIVEN: Two paths that differ only in the adopter kind.
    inputs = _inputs()
    keep_spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    drain_spec = PathSpec(0, Fraction(1, 2), "need", "double", "drain")
    _keep_content, keep_rows = run_path(keep_spec, inputs, None)
    _drain_content, drain_rows = run_path(drain_spec, inputs, None)
    # THEN: Keep paths only ever use KEEP and NON, while drain paths do use DRAIN.
    assert {row["kind"] for row in keep_rows} <= {NON, KEEP}
    assert KEEP in {row["kind"] for row in keep_rows}
    assert DRAIN in {row["kind"] for row in drain_rows}


def test_p_one_breaks_the_whole_unit_of_adopter_outputs() -> None:
    # GIVEN: The same decimal-rule path run with the tool's random decimal off (p = 0) and on (p = 1).
    spec = PathSpec(0, Fraction(1, 2), "decstep", "balance", "keep")

    def keep_amounts(p: str) -> list[object]:
        _content, rows = run_path(spec, _inputs(p=p), None)
        return [row["y"] for row in rows if row["kind"] == KEEP]

    # WHEN: Simulating both.
    off, on = keep_amounts("0"), keep_amounts("1")
    # THEN: Without the random decimal every KEEP amount is a whole unit, and with it some amount is not.
    assert off
    assert all(y % D == 0 for y in off)
    assert any(y % D != 0 for y in on)
