"""Regression tests for the exact matrix sharing in :func:`lab.train.train_logit`.

The trainer used to hold one `(N, F)` feature matrix per simulated event: for the ETH training set
(510,902 events, `N` = 3,973, `F` = 9) that is 138 GiB, over the host's 82.92 GiB budget.  The
fix shares one `(matrix, eligible)` pair per distinct public amount `y`.

`logit.features` is a pure function of `(y, public_balances, units, whole_unit)`.  `units` is
`config.logit_units` and `whole_unit` is `inputs.d`, both fixed for the whole call and for the
one ordered balance set a :class:`~lab.study.StudyInputs` carries, so `y` alone selects the row.
The cache is local to one `train_logit` call.

These tests pin the contract:

1. Paths share the array objects of a repeated `y`, and `logit.features` runs once per distinct
   `y` instead of once per event.
2. The cache does not survive between two `train_logit` calls.
3. Every group handed to `logit.fit` is element-wise identical, in the same order, to the frozen
   reference grouping of the same events, and the fitted `beta` is bit-for-bit identical.
4. `logit.fit` only reads the shared matrices: their bytes are unchanged across a fit.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from lab import sim
from lab.observers import logit
from lab.schema import (
    Deposit,
    ExperimentConfig,
    LogitTrainSpec,
    Manifest,
    Snapshot,
    ThetaSpec,
    Withdrawal,
)
from lab.study import inputs_from
from lab.train import train_logit

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pytest

    from lab.sim import Event
    from lab.study import StudyInputs

# ==== synthetic ETH-like data: D is 0.001 ETH = 10**15 wei ====

D = 10**15
DECIMALS = 18
STEP = D // 100
QUANTUM = D // 100
THRESHOLD = D
LOGIT_UNITS = (D // 100, D, 10 * D, 100 * D, 1000 * D)
THETA = ThetaSpec("9/22", "0", "0", "hi", 10 * D, "synthetic")
LOGIT_TRAIN = LogitTrainSpec(3, "0.001", "1/4", "3/4", "1/2", "1", "synthetic")
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
        "logit_units": LOGIT_UNITS, "logit_train": LOGIT_TRAIN, "seed_root": "eth_share_s0",
    }
    fields.update(overrides)
    return ExperimentConfig(**fields)


def _inputs(**overrides: object) -> StudyInputs:
    snapshot = _snapshot()
    return inputs_from(_config("0" * 64, **overrides), _manifest(None), snapshot, "0" * 64)


def _train(inputs: StudyInputs, paths: int, config: ExperimentConfig) -> dict:
    return train_logit(inputs, paths=paths, tag=config.seed_root, pairs=PAIRS, rhos=RHOS,
                       adopters=ADOPTERS)


def _install_spies(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Wrap `sim.simulate`, `logit.features` and `logit.fit` to record what the trainer did."""
    harness: dict = {"records": [], "feature_calls": 0, "fit_calls": 0, "groups": None, "fit_l2": None}
    real_simulate = sim.simulate

    def recording_simulate(model: object, balances: object, kinds: object, seed: int) -> tuple:
        events = real_simulate(model, balances, kinds, seed)
        harness["records"].append((tuple(balances), events))
        return events

    monkeypatch.setattr(sim, "simulate", recording_simulate)
    real_features = logit.features

    def counting_features(y: int, balances: Sequence[int], units: Sequence[int],
                          whole_unit: int) -> tuple:
        harness["feature_calls"] += 1
        return real_features(y, balances, units, whole_unit)

    monkeypatch.setattr(logit, "features", counting_features)
    real_fit = logit.fit

    def capturing_fit(groups: Sequence[logit.Group], *, l2: float = 1e-3, max_iter: int = 200) -> np.ndarray:
        harness["groups"] = list(groups)
        harness["fit_calls"] += 1
        harness["fit_l2"] = l2
        return real_fit(groups, l2=l2, max_iter=max_iter)

    monkeypatch.setattr(logit, "fit", capturing_fit)
    return harness


def _reference_groups(events: Sequence[Event], balances: Sequence[int], units: Sequence[int],
                      whole_unit: int) -> list[logit.Group]:
    """The reference grouping: one `(matrix, eligible, true_source)` per simulated event."""
    groups: list[logit.Group] = []
    for event in events:
        matrix, eligible = logit.features(event.y, balances, units, whole_unit)
        groups.append((matrix, eligible, event.source))
    return groups


def _frozen_groups(records: Sequence[tuple], units: Sequence[int], whole_unit: int) -> list:
    """The frozen grouping: one reference grouping call per path, in the same order."""
    groups: list = []
    for balances, events in records:
        groups.extend(_reference_groups(events, balances, units, whole_unit))
    return groups


def _y_sequence(records: Sequence[tuple]) -> list[int]:
    return [event.y for _balances, events in records for event in events]


# ==== 1. shared balances share the repeated amount's arrays ====


def test_shared_balances_share_one_matrix_per_amount(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: Three paths over one study input.
    inputs = _inputs()
    harness = _install_spies(monkeypatch)
    # WHEN: Training.
    result = _train(inputs, 3, _config("0" * 64))
    # THEN: The feature builder ran once per distinct amount, not once per event.
    ys = _y_sequence(harness["records"])
    assert harness["feature_calls"] == len(set(ys)) < len(ys)
    # AND every group the fit received is one simulated event, in order.
    assert len(harness["groups"]) == len(ys) == result["n_groups"]
    # AND two events with the same public amount point at the same array objects.
    first_seen: dict[int, int] = {}
    for index, y in enumerate(ys):
        if y in first_seen:
            earlier = harness["groups"][first_seen[y]]
            assert harness["groups"][index][0] is earlier[0]
            assert harness["groups"][index][1] is earlier[1]
        else:
            first_seen[y] = index
    # AND the number of distinct matrices is exactly the number of distinct amounts.
    assert len({id(group[0]) for group in harness["groups"]}) == len(set(ys))


# ==== 2. the cache is local to one call ====


def test_the_cache_does_not_survive_between_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: The study inputs and a feature-call counter.
    inputs = _inputs()
    config = _config("0" * 64)
    harness = _install_spies(monkeypatch)
    # WHEN: Training twice.
    _train(inputs, 2, config)
    first = harness["feature_calls"]
    _train(inputs, 2, config)
    second = harness["feature_calls"] - first
    # THEN: The second call rebuilt the same number of matrices, so nothing was kept across calls.
    assert first > 0
    assert second == first


# ==== 3. the groups and the fitted beta equal the frozen grouping bit for bit ====


def test_groups_and_beta_match_the_frozen_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: Three training paths.
    inputs = _inputs()
    harness = _install_spies(monkeypatch)
    # WHEN: Training.
    result = _train(inputs, 3, _config("0" * 64))
    # THEN: Every group is element-wise identical, in the same order, to the frozen grouping.
    frozen = _frozen_groups(harness["records"], LOGIT_UNITS, D)
    got = harness["groups"]
    assert len(got) == len(frozen) == result["n_groups"] == sum(len(events) for _b, events in harness["records"])
    for index, (new_group, old_group) in enumerate(zip(got, frozen, strict=True)):
        assert np.array_equal(new_group[0], old_group[0]), index
        assert new_group[0].tobytes() == old_group[0].tobytes(), index
        assert np.array_equal(new_group[1], old_group[1]), index
        assert new_group[1].tobytes() == old_group[1].tobytes(), index
        assert new_group[2] == old_group[2], index
    # AND the fit was handed the frozen L2 penalty.
    assert harness["fit_l2"] == float(Fraction(LOGIT_TRAIN.l2))
    # AND the fitted beta is bit-for-bit the beta fitted on the frozen grouping.
    frozen_beta = logit.fit(frozen, l2=float(Fraction(LOGIT_TRAIN.l2)))
    assert np.array(result["beta"]).tobytes() == frozen_beta.tobytes()


# ==== 4. the fit only reads the shared matrices ====


def test_fit_does_not_write_to_the_shared_matrices(monkeypatch: pytest.MonkeyPatch) -> None:
    # GIVEN: A fit wrapper that snapshots the shared arrays' bytes around the real fit.
    inputs = _inputs()
    before: list[bytes] = []
    after: list[bytes] = []
    real_fit = logit.fit

    def byte_snapshot_fit(groups: Sequence[logit.Group], *, l2: float = 1e-3,
                          max_iter: int = 200) -> np.ndarray:
        before.extend(group[0].tobytes() + group[1].tobytes() for group in groups)
        beta = real_fit(groups, l2=l2, max_iter=max_iter)
        after.extend(group[0].tobytes() + group[1].tobytes() for group in groups)
        return beta

    monkeypatch.setattr(logit, "fit", byte_snapshot_fit)
    # WHEN: Training.
    _train(inputs, 3, _config("0" * 64))
    # THEN: Every shared matrix and mask is byte-identical before and after the fit.
    assert len(before) == len(after) > 0
    assert before == after
