"""Observers of the public amounts: each guesses which deposit made each withdrawal.

An observer exposes `posterior(v)` for one event's public amount `v` and `update(y)` to
advance to the next event.  The `historical` and `trained-logit` observers score each event
in the online order, reading only the event at hand; the `combinatorial` observer instead
links the whole public stream once when it is built, and its `posterior(v)` reads that
final link state.  Observers see only the public amounts and the initial balances `b`; the
behaviours they are built from expose `mass(y, r)`, the `P(y | r)` of one kind.  The notation is
that of :mod:`lab`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Behaviour", "HistoricalObserver"]


class Behaviour(Protocol):
    """A person kind's next-withdrawal law: `mass(y, r)`, the `P(y | r)` of one kind."""

    def mass(self, y: int, r: int) -> float:
        """`P(next withdrawal is y | remaining r)`."""
        pass


# ==== historical ====


class HistoricalObserver:
    """The `historical` observer: `P(source = i | v)` proportional to the historical `P(v | b_i)`.

    It ignores earlier events.  When the historical model gives `v` zero mass at every balance,
    the posterior is uniform over the deposits with `b_i >= v`.

    Args:
        historical: The historical behaviour; it supplies `mass(v, b)` for a balance `b`.
        balances: Initial public balances `b_i`, one per deposit, in deposit order.
    """

    def __init__(self, historical: Behaviour, balances: Sequence[int]) -> None:
        self.historical = historical
        self.balances = tuple(balances)
        self.fallback_count = 0

    def posterior(self, v: int) -> np.ndarray:
        """Historical-model posterior over the `n` deposits for the next public amount `v`.

        Args:
            v: The next event's public amount.

        Returns:
            A length-`n` distribution over the deposits proportional to `mass(v, b_i)`; when the
            model gives `v` zero mass at every balance it is uniform over `b_i >= v` and
            `fallback_count` is incremented. If no initial balance covers `v`, the result has NaN entries.
        """
        joint = np.array([self.historical.mass(v, b) for b in self.balances])
        if joint.sum() == 0.0:
            self.fallback_count += 1
            eligible = np.array([1.0 if b >= v else 0.0 for b in self.balances])
            return eligible / eligible.sum()
        return joint / joint.sum()

    def update(self, y: int) -> None:
        """No-op: the historical observer keeps no history."""
        pass
