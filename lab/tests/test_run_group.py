"""The runner's optional `group` observer: what it is given, and that it leaves the old observers alone.

These tests stand in their own `lab.observers.group` module, so they check the runner's wiring (seed,
candidate counts, adopter kind, lazy import, public inputs, unchanged old rows) without depending on
the observer's own implementation.
"""

from __future__ import annotations

import json
import random
import sys
import types
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lab import sim, study
from lab.run import GROUP_STATES, PathSpec, run, run_path
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
from lab.sim import DRAIN, KEEP, NON

if TYPE_CHECKING:
    from pathlib import Path

    from lab.study import StudyInputs

GROUP_MODULE = "lab.observers.group"
BASE_OBSERVERS = ("historical", "trained-logit", "combinatorial")

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


def _manifest(digest: str) -> Manifest:
    return Manifest(1, "0x" + "22" * 20, "0x" + "11" * 20, "0x" + "33" * 20, 0, D, 1, 1000,
                    "0x" + "00" * 32, digest)


def _config(snapshot_sha256: str, **overrides: object) -> ExperimentConfig:
    fields: dict[str, object] = {
        "manifest": "manifest.json", "snapshot_sha256": snapshot_sha256, "split_block": 200,
        "theta": THETA, "replicates": 1, "rhos": ("0", "1/2"), "rules": ("need",),
        "m_models": ("double",), "adopters": ("keep", "drain"), "observers": BASE_OBSERVERS,
        "combinatorial_tolerance": "0", "threshold": THRESHOLD, "quantum": QUANTUM, "step": STEP,
        "historical": "plain",
        "logit_units": LOGIT_UNITS, "logit_train": LOGIT_TRAIN, "seed_root": "synthetic_s0",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _inputs(**overrides: object) -> StudyInputs:
    snapshot = _snapshot()
    return study.inputs_from(_config("0" * 64, **overrides), _manifest(sha256(snapshot)), snapshot,
                             sha256(snapshot))


def _setup(tmp_path: Path, **overrides: object) -> str:
    """Write the snapshot, manifest and config; return the config path."""
    snapshot = _snapshot()
    digest = sha256(snapshot)
    (tmp_path / "manifest.json").write_text(json.dumps(_manifest(digest).to_json()), encoding="utf-8")
    folder = tmp_path / "data" / "manifest"
    folder.mkdir(parents=True)
    (folder / "snapshot.json").write_bytes(canonical(snapshot))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config(digest, **overrides).to_json()), encoding="utf-8")
    return str(config_path)


def _rows_of(rows: list[dict], observer: str) -> list[dict]:
    return [row for row in rows if row["observer"] == observer]


# ==== the stand-in observer ====


class SpyGroupObserver:
    """Stand-in for `lab.observers.group.GroupObserver`: records its inputs, scores uniformly."""

    made: list[SpyGroupObserver] = []

    def __init__(self, historical: object, adopter: object, balances: object, ks: object,
                 max_states: object, rng: object) -> None:
        self.historical = historical
        self.adopter = adopter
        self.balances = tuple(balances)  # type: ignore[arg-type]
        self.ks = ks
        self.max_states = max_states
        self.rng = rng
        self.fallback_count = 0
        self.calls: list[tuple[str, int]] = []
        SpyGroupObserver.made.append(self)

    def posterior(self, v: int) -> np.ndarray:
        self.calls.append(("posterior", v))
        return np.full(len(self.balances), 1.0 / len(self.balances))

    def update(self, y: int) -> None:
        self.calls.append(("update", y))


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> type[SpyGroupObserver]:
    module = types.ModuleType(GROUP_MODULE)
    module.GroupObserver = SpyGroupObserver  # type: ignore[attr-defined]
    SpyGroupObserver.made = []
    monkeypatch.setitem(sys.modules, GROUP_MODULE, module)
    return SpyGroupObserver


# ==== lazy, optional wiring ====


def test_a_config_without_group_never_builds_the_group_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: A `lab.observers.group` module whose observer raises if it is built at all.
    module = types.ModuleType(GROUP_MODULE)

    def explode(*args: object, **kwargs: object) -> object:
        raise AssertionError("group")

    module.GroupObserver = explode  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, GROUP_MODULE, module)
    inputs = _inputs()
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    # WHEN: Scoring a path with only the original three observers.
    content, rows = run_path(spec, inputs, LOGIT_BETA)
    # THEN: Nothing is loaded or built, and the path carries no group metadata.
    assert "group" not in content
    assert {row["observer"] for row in rows} == set(BASE_OBSERVERS)


def test_enabling_group_leaves_the_original_observers_identical(spy: type[SpyGroupObserver]) -> None:
    # GIVEN: The same path scored with and without the group observer, everything else equal.
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    before_content, before_rows = run_path(spec, _inputs(), LOGIT_BETA)
    after_content, after_rows = run_path(spec, _inputs(observers=(*BASE_OBSERVERS, "group")), LOGIT_BETA)
    # THEN: Every original observer's rows are the same rows, in the same order, with the same values.
    assert after_rows[:len(before_rows)] == before_rows
    for name in BASE_OBSERVERS:
        assert _rows_of(after_rows, name) == _rows_of(before_rows, name)
        assert after_content["observers"][name]["fallback_count"] == before_content["observers"][name]["fallback_count"]
    # THEN: The path summary of the original observers and the simulated path are unchanged.
    assert after_content["K"] == before_content["K"]
    assert after_content["N"] == before_content["N"]
    assert after_content["n_events"] == before_content["n_events"]
    assert after_content["spec"] == before_content["spec"]
    assert set(after_content["observers"]) == {*BASE_OBSERVERS, "group"}
    # THEN: The group rows come last and cover the same public events in order.
    group_rows = after_rows[len(before_rows):]
    assert [row["observer"] for row in group_rows] == ["group"] * len(group_rows)
    assert [row["y"] for row in group_rows] == [row["y"] for row in _rows_of(before_rows, "historical")]
    assert len(spy.made) == 1


# ==== what the group observer is given ====


def test_the_candidate_counts_map_each_rho_entry_and_keep_duplicates(spy: type[SpyGroupObserver]) -> None:
    # GIVEN: A config whose `rhos` repeat a ratio and include one that rounds up at the half.
    rhos = ("0", "5/24", "1/2", "1/2", "3/7")
    inputs = _inputs(observers=(*BASE_OBSERVERS, "group"), rhos=rhos)
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    # WHEN: Scoring one path.
    content, _rows = run_path(spec, inputs, LOGIT_BETA)
    # THEN: One `K = floor(rho*N + 1/2)` per entry, in order, the duplicate kept.
    assert spy.made[0].ks == (0, 3, 6, 6, 5)
    assert isinstance(spy.made[0].ks, tuple)
    assert content["group"] == {"K_candidates": [0, 3, 6, 6, 5], "states": GROUP_STATES}
    assert spy.made[0].max_states == GROUP_STATES


@pytest.mark.parametrize(("adopters", "kind"), [("keep", KEEP), ("drain", DRAIN)])
def test_the_adopter_behaviour_is_the_paths_own_kind(spy: type[SpyGroupObserver], adopters: str,
                                                     kind: int) -> None:
    # GIVEN: A keep path and a drain path of the same rule and config.
    spec = PathSpec(0, Fraction(1, 2), "need", "double", adopters)
    inputs = _inputs(observers=(*BASE_OBSERVERS, "group"))
    # WHEN: Scoring the path.
    run_path(spec, inputs, LOGIT_BETA)
    built = spy.made[0]
    behaviour = study.behaviour(inputs, study.theta_of(inputs.config.theta), spec.rule, spec.m_model,
                                p=Fraction(inputs.config.p))
    other = DRAIN if kind == KEEP else KEEP
    probes = [(y, r) for r in (10, 20, 30, 40, 60, 120) for y in (5, 10, 20, 30, 40, 60, 100)]
    # THEN: The historical behaviour is the no-tool one and the adopter behaviour is this path's kind.
    for y, r in probes:
        assert built.historical.mass(y, r) == pytest.approx(behaviour[NON].mass(y, r))
        assert built.adopter.mass(y, r) == pytest.approx(behaviour[kind].mass(y, r))
    # THEN: The two kinds really do differ somewhere, so the check above is not vacuous.
    differing = [(y, r) for y, r in probes
                 if not np.isclose(behaviour[KEEP].mass(y, r), behaviour[DRAIN].mass(y, r))]
    assert differing
    y, r = differing[0]
    assert not np.isclose(built.adopter.mass(y, r), behaviour[other].mass(y, r))


def test_the_rng_is_the_frozen_seed_expression(spy: type[SpyGroupObserver]) -> None:
    # GIVEN: A config and the first path in the grid.
    inputs = _inputs(observers=(*BASE_OBSERVERS, "group"))
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    # WHEN: Scoring it, and a second path with another replicate.
    run_path(spec, inputs, LOGIT_BETA)
    run_path(PathSpec(1, Fraction(1, 2), "need", "double", "keep"), inputs, LOGIT_BETA)
    # THEN: Each observer draws from `random.Random(derive_seed("group", seed_root, N, replicate, rho, rule))`.
    n = len(DEPOSIT_VALUES)
    expected = random.Random(sim.derive_seed("group", inputs.config.seed_root, n, spec.replicate,
                                             str(spec.rho), spec.rule))
    assert spy.made[0].rng.random() == expected.random()
    assert spy.made[1].rng.random() != spy.made[0].rng.random()


def test_the_group_observer_is_scored_from_public_amounts_only(spy: type[SpyGroupObserver]) -> None:
    # GIVEN: A keep path under the group configuration.
    spec = PathSpec(0, Fraction(1, 2), "need", "double", "keep")
    inputs = _inputs(observers=(*BASE_OBSERVERS, "group"))
    # WHEN: Scoring it.
    _content, rows = run_path(spec, inputs, LOGIT_BETA)
    built = spy.made[0]
    # THEN: It sees the public withdrawal amount `y` of each event, in order, and nothing else.
    ys = [row["y"] for row in _rows_of(rows, "historical")]
    assert [value for call, value in built.calls if call == "posterior"] == ys
    assert [value for call, value in built.calls if call == "update"] == ys
    assert all(type(value) is int for _call, value in built.calls)
    # THEN: It knows the public initial balances, not the adopter set.
    assert built.balances == tuple(inputs.balances)
    assert not hasattr(built, "kinds")
    assert not hasattr(built, "source")


# ==== files on disk ====


def test_written_outputs_only_gain_group_metadata_when_group_is_scored(tmp_path: Path,
                                                                       spy: type[SpyGroupObserver]) -> None:
    # GIVEN: A grid run with the original observers, then the same run with group appended.
    plain = tmp_path / "plain"
    plain.mkdir()
    covered = tmp_path / "covered"
    covered.mkdir()
    plain_config = _setup(plain)
    covered_config = _setup(covered, observers=(*BASE_OBSERVERS, "group"))
    # WHEN: Running both.
    run(plain_config, data=str(plain / "data"), out=str(plain / "out"), workers=1, logit_beta_path=_beta(plain))
    run(covered_config, data=str(covered / "data"), out=str(covered / "out"), workers=1,
        logit_beta_path=_beta(covered))
    # THEN: Every path summary keeps exactly the old key set without group, and old values with it.
    plain_paths = sorted((plain / "out").glob("*/path.json"))
    assert len(plain_paths) == 3
    for path in plain_paths:
        content = json.loads(path.read_text(encoding="utf-8"))
        assert set(content) == {"spec", "K", "N", "n_events", "observers"}
        assert set(content["observers"]) == set(BASE_OBSERVERS)
        enabled = json.loads((covered / "out" / path.parent.name / "path.json").read_text(encoding="utf-8"))
        assert set(enabled) == {"spec", "K", "N", "n_events", "observers", "group"}
        assert enabled["group"] == {"K_candidates": [0, 6], "states": GROUP_STATES}
        for key in ("spec", "K", "N", "n_events"):
            assert enabled[key] == content[key]
        # THEN: The original observers' rows are byte-for-byte the same values.
        old_rows = read_rows(path.parent / "rows.csv.gz")
        new_rows = read_rows(covered / "out" / path.parent.name / "rows.csv.gz")
        assert new_rows[:len(old_rows)] == old_rows
        assert {row["observer"] for row in new_rows[len(old_rows):]} == {"group"}
        assert len(new_rows) == len(old_rows) + content["n_events"]


def _beta(folder: Path) -> str:
    path = folder / "beta.json"
    path.write_text(json.dumps({"beta": LOGIT_BETA, "whole_unit": D, "logit_units": list(LOGIT_UNITS)}),
                    encoding="utf-8")
    return str(path)
