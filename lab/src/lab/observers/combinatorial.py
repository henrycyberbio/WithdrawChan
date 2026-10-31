"""The `combinatorial` observer: it links each withdrawal to a deposit from the public amounts alone.

Candidates of an event are the open deposits whose remaining capacity covers it, narrowed to the
near matches, else the tail matches, else all of them.  Passes link and close deposits until one
links nothing.

Notation

- `j`: an event's position in the public stream; `link[j]` is the deposit it is linked to, or -1.
- `linked[i]`: the amount deposit `i` has already paid out; `closed[i]`: whether it is spent.
- `rem[i] = b_i - linked[i]`: its remaining capacity.
- `slack[i] = floor(b_i * tolerance)`: what may stay unlinked when it closes.
- `tolerance`: the fraction of a deposit that may stay unlinked.
- `tail_unit`: the grid on which a deposit and a withdrawal share a nonzero remainder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from fractions import Fraction

__all__ = ["CombinatorialObserver"]

_INT64_MAX = int(np.iinfo(np.int64).max)


def _int64_safe(b: list[int], amounts: list[int], numerator: int, tail_unit: int) -> bool:
    """Whether the observer's whole run provably stays inside `int64`.

    Every quantity it forms is an add, subtract, remainder or `numerator`-multiple of a value
    bounded by `max(b)` or `max(amounts)`: `linked[i]` never exceeds `b_i` (a link requires
    `rem >= y`), so `rem`, `rem - y` and the remainders stay in range; only the slack product
    `max(b) * numerator` can grow, and `tail_unit` is a modulus.  So the run is safe iff the
    values and `tail_unit` are nonnegative and fit, `numerator` is nonnegative, and that product fits.
    """
    low_b, high_b = min(b, default=0), max(b, default=0)
    low_y, high_y = min(amounts, default=0), max(amounts, default=0)
    return (low_b >= 0 and high_b <= _INT64_MAX and low_y >= 0 and high_y <= _INT64_MAX
            and numerator >= 0 and 0 <= tail_unit <= _INT64_MAX
            and high_b * numerator <= _INT64_MAX)


class CombinatorialObserver:
    """The `combinatorial` observer: `P(source = i | v)` from the final link state of the stream.

    Args:
        balances: Initial public balances `b_i`, one per deposit, in deposit order.
        amounts: The full public withdrawal stream, in event order; it is linked once here.
        tolerance: The fraction of a deposit that may stay unlinked when it closes.
        tail_unit: The grid on which a deposit and a withdrawal share a nonzero remainder.
    """

    def __init__(self, balances: Sequence[int], amounts: Sequence[int], tolerance: Fraction,
                 tail_unit: int) -> None:
        b = [int(x) for x in balances]
        am = [int(y) for y in amounts]
        numerator, denominator = tolerance.numerator, tolerance.denominator
        tail_unit = int(tail_unit)
        # Amounts are asset base units and may exceed int64 (wei-scale ETH does); use exact Python
        # ints unless every value and intermediate provably fits int64.
        dtype = np.int64 if _int64_safe(b, am, numerator, tail_unit) else object
        self.b = np.asarray(b, dtype=dtype)
        self.amounts = np.asarray(am, dtype=dtype)
        self.tail_unit = tail_unit
        self.slack = (self.b * numerator) // denominator
        self.linked = np.zeros(len(self.b), dtype=dtype)
        self.closed = np.zeros(len(self.b), dtype=bool)
        self.link = np.full(len(self.amounts), -1, dtype=np.int64)
        self.fallback_count = 0
        self._index = 0
        self._run()
        self._sets = [np.array([self.link[j]]) if self.link[j] >= 0 else self.candidates(j)
                      for j in range(len(self.amounts))]

    def candidates(self, j: int) -> np.ndarray:
        """Deposit indices that can source withdrawal `j` under the current state.

        Args:
            j: The event's position in the public stream.

        Returns:
            The candidate deposit indices for `j`, ascending: the near matches, else the tail
            matches, else every feasible deposit.
        """
        rem = self.b - self.linked
        y = self.amounts[j]
        feasible = ~self.closed & (rem >= y)
        near = feasible & (rem - y <= self.slack)
        tail = y % self.tail_unit
        tails = feasible & (tail != 0) & (rem % self.tail_unit == tail)
        return np.flatnonzero(near if near.any() else tails if tails.any() else feasible)

    def _run(self) -> None:
        """Link single-candidate withdrawals in passes until one links nothing."""
        while True:
            made = False
            for j in np.flatnonzero(self.link < 0):
                idx = self.candidates(j)
                if len(idx) != 1:
                    continue
                i = int(idx[0])
                self.link[j] = i
                self.linked[i] += self.amounts[j]
                if self.b[i] - self.linked[i] <= self.slack[i]:
                    self.closed[i] = True
                made = True
            if not made:
                return

    def posterior(self, v: int) -> np.ndarray:
        """Uniform posterior over the current event's final set, or over `b_i >= v` when it is empty.

        Args:
            v: The current event's public amount.

        Returns:
            A length-`n` distribution, uniform over the event's final candidate set; when that
            set is empty it is uniform over `b_i >= v` and `fallback_count` is incremented.
            If no initial balance covers `v`, division by zero yields NaN entries.
        """
        candidates = self._sets[self._index]
        if len(candidates) == 0:
            self.fallback_count += 1
            eligible = self.b >= v
            return eligible.astype(np.float64) / eligible.sum()
        posterior = np.zeros(len(self.b), dtype=np.float64)
        posterior[candidates] = 1.0 / len(candidates)
        return posterior

    def update(self, y: int) -> None:
        """Advance to the next event."""
        self._index += 1
