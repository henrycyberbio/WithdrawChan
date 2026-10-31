"""Behaviour of the observers and scoring: historical and trained-logit posteriors, feature eligibility, and the
gzip CSV row round-trip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lab.observers.historical import HistoricalObserver
from lab.observers.logit import TrainedLogitObserver, features, predict
from lab.score import COLUMNS, event_row, read_rows, write_rows

if TYPE_CHECKING:
    from pathlib import Path


class _Fixed:
    """Duck-typed behaviour: constant `mass` on `y <= r`, active while `r >= 1`."""

    def __init__(self, mass: float) -> None:
        self._mass = mass

    def mass(self, y: int, r: int) -> float:
        """`mass` when `y <= r`, else zero."""
        return self._mass if y <= r else 0.0

    def active(self, r: int) -> bool:
        """Active while any balance remains."""
        return r >= 1


@dataclass(frozen=True)
class _Event:
    """Minimal event carrying the fields the observer and scorer read."""

    public_index: int
    source: int
    kind: int
    x: int
    y: int


# ==== historical ====


def test_static_observer_normalises_the_historical_masses() -> None:
    # GIVEN: Three deposits where the amount has positive mass for two and none for the third.
    observer = HistoricalObserver(_Fixed(1.0), (10, 10, 5))
    # WHEN: Scoring the next public amount that only the first two can source.
    q = observer.posterior(6)
    # THEN: The posterior is the mass vector normalised over the positive entries.
    assert q.tolist() == [0.5, 0.5, 0.0]


def test_static_observer_falls_back_to_uniform_over_eligible_deposits() -> None:
    # GIVEN: A behaviour that gives zero mass everywhere and two eligible deposits.
    observer = HistoricalObserver(_Fixed(0.0), (10, 10, 5))
    # WHEN: Scoring an amount the model cannot explain.
    q = observer.posterior(6)
    # THEN: Every eligible deposit gets equal weight, so the row stays a distribution.
    assert q.tolist() == [0.5, 0.5, 0.0]


# ==== trained logit ====


def test_trained_logit_observer_softmax_is_uniform_over_eligible_deposits() -> None:
    # GIVEN: A zero beta, so every eligible candidate has the same logit.
    observer = TrainedLogitObserver(np.zeros(5), (100, 200, 50), (10**4,), 10**6)
    # WHEN: Scoring an amount only the first two deposits can source.
    q = observer.posterior(100)
    # THEN: The posterior is uniform over them and zero on the ineligible deposit.
    assert q.tolist() == [0.5, 0.5, 0.0]


def test_features_zero_every_ineligible_row() -> None:
    # GIVEN: Balances above and below the public amount.
    matrix, eligible = features(100, (100, 40), (10**4,), 10**6)
    # WHEN: Reading the feature matrix.
    # THEN: The ineligible row is all zeros and only the eligible row is marked.
    assert (eligible.tolist(), matrix[1].tolist()) == ([True, False], [0.0] * 5)


# ==== scoring ====


def test_event_row_reports_tie_aware_rank_and_top1() -> None:
    # GIVEN: A posterior whose maximum sits on the true source.
    event = _Event(3, 1, 2, 10, 100)
    row = event_row(np.array([0.2, 0.5, 0.3]), event, "p", "trained-logit")
    # WHEN: Scoring the event.
    # THEN: The row's p_true and max_posterior are the true source's mass and the top1 hit is full.
    assert (row["p_true"], row["max_posterior"], row["top1"], row["true_rank"]) == (0.5, 0.5, 1.0, 1.0)
    assert row["public_index"] == 3
    assert row["y"] == 100


def test_rows_round_trip_through_gzip_csv(tmp_path: Path) -> None:
    # GIVEN: A scored row.
    row = event_row(np.array([0.5, 0.5]), _Event(0, 0, 1, 7, 8), "p", "historical")
    # WHEN: Writing and reading it back.
    target = tmp_path / "rows.csv.gz"
    write_rows(target, [row])
    back = read_rows(target)
    # THEN: The typed values survive unchanged, and the header matches the schema.
    assert back[0] == row
    assert tuple(row) == COLUMNS
