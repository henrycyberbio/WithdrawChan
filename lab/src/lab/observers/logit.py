"""The `trained-logit` observer: a conditional logit over the deposits, on amount features.

For a public amount `y`, each deposit with `b >= y` gets a feature row and the probability of
being the source is the softmax of `row @ beta` over those deposits.  `beta` is fitted by a
deterministic L2-penalised Newton solver.  The notation is that of :mod:`lab`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["TrainedLogitObserver", "feature_names", "features", "fit", "predict"]

#: One training group: feature matrix, eligible mask and the true source index.
Group = tuple[np.ndarray, np.ndarray, int]

# ==== features ====


def feature_names(units: Sequence[int]) -> tuple[str, ...]:
    """Column names of :func:`features`, in order, for the tail `units`.

    Args:
        units: The tail-feature units `u`, one `tail_{u}` column each, in this order.

    Returns:
        The `F = 4 + len(units)` column names, in column order.
    """
    return ("log_ratio", "is_full", *(f"tail_{u}" for u in units), "b_whole", "b_whole_and_y_whole")


def features(
    y: int, public_balances: Sequence[int], units: Sequence[int], whole_unit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-deposit feature matrix `(N, F)` for the public amount `y`, and the eligible mask `b >= y`.

    `whole_unit` is the rounding unit `d` (`manifest.d`), the divisor of the two divisibility
    features.  Features of a deposit with initial balance `b`: `log_ratio = log(b) - log(y)`;
    `is_full`, `b == y`; for each unit `u`, `tail_u`: `y % u` is nonzero and equals `b % u`;
    `b_whole`: `b` is a multiple of `d`; `b_whole_and_y_whole`: both `b` and `y` are.
    An ineligible row is all zeros.  Only `log_ratio` is computed in floats.

    Args:
        y: The event's public amount.
        public_balances: Initial public balances `b`, one per deposit, in deposit order.
        units: The tail-feature units `u`; the tail columns follow this order.
        whole_unit: The rounding unit `d` (`manifest.d`), dividing the two divisibility
            features.

    Returns:
        `(matrix, eligible)`: the `(N, F)` float64 matrix, `N = len(public_balances)` rows in
        deposit order and `F = 4 + len(units)` columns as in :func:`feature_names`, and the
        length-`N` boolean mask `b >= y`.  The inputs are not modified.
    """
    n = len(public_balances)
    width = 4 + len(units)
    matrix = np.zeros((n, width), dtype=np.float64)
    eligible = np.zeros(n, dtype=bool)
    log_y = math.log(y)
    for i, b in enumerate(public_balances):
        if b < y:
            continue
        eligible[i] = True
        row = matrix[i]
        row[0] = math.log(b) - log_y
        row[1] = 1.0 if b == y else 0.0
        for k, u in enumerate(units):
            residue = y % u
            row[2 + k] = 1.0 if residue != 0 and residue == b % u else 0.0
        whole = b % whole_unit == 0
        row[2 + len(units)] = 1.0 if whole else 0.0
        row[3 + len(units)] = 1.0 if whole and y % whole_unit == 0 else 0.0
    return matrix, eligible


# ==== model ====


def predict(beta: np.ndarray, matrix: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Softmax of `matrix @ beta` over the eligible rows; zero on ineligible rows.

    Args:
        beta: One coefficient per feature column of `matrix`.
        matrix: The `(N, F)` feature rows, one per deposit, in deposit order.
        eligible: The boolean mask of the deposits that may be the source; ineligible rows
            are excluded from the softmax.

    Returns:
        With at least one eligible row, a length-`N` distribution summing to one and zero on
        ineligible rows. An all-false eligibility mask yields NaN entries.
    """
    logits = np.where(eligible, matrix @ beta, -np.inf)
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def _nll(beta: np.ndarray, groups: Sequence[Group], l2: float) -> float:
    """L2-penalised negative log-likelihood, `0.5*l2*||beta||^2` plus the group terms."""
    total = 0.5 * l2 * float(beta @ beta)
    for matrix, eligible, true_index in groups:
        logits = np.where(eligible, matrix @ beta, -np.inf)
        top = logits.max()
        total += top + math.log(float(np.exp(logits - top).sum())) - logits[true_index]
    return total


def fit(
    groups: Sequence[Group], l2: float = 1e-3, max_iter: int = 200
) -> np.ndarray:
    """Fit the conditional logit on `(matrix, eligible, true_index)` groups; one `beta` per feature.

    The choice probability is `softmax` over the eligible rows of `matrix @ beta`.  Newton steps
    with a fixed backtracking line search make the result deterministic.

    Args:
        groups: The training groups `(matrix, eligible, true_index)`, consumed in this order;
            every group shares the feature width.  The groups are not modified.
        l2: The L2 penalty.
        max_iter: The maximum number of Newton steps.

    Returns:
        `beta`, one coefficient per feature column.
    """
    width = groups[0][0].shape[1]
    beta = np.zeros(width)
    eye = np.eye(width)
    for _ in range(max_iter):
        grad = l2 * beta
        hessian = l2 * eye
        for matrix, eligible, true_index in groups:
            logits = np.where(eligible, matrix @ beta, -np.inf)
            weights = np.exp(logits - logits.max())
            probs = weights / weights.sum()
            xbar = probs @ matrix
            eligible_rows = matrix[eligible]
            hessian += ((eligible_rows * probs[eligible][:, None]).T @ eligible_rows
                        - np.outer(xbar, xbar))
            grad = grad + xbar - matrix[true_index]
        step = np.linalg.solve(hessian, grad)
        cur = _nll(beta, groups, l2)
        scale = 1.0
        while scale > 1e-12 and _nll(beta - scale * step, groups, l2) > cur:
            scale *= 0.5
        beta = beta - scale * step
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    return beta


# ==== observer ====


class TrainedLogitObserver:
    """The `trained-logit` observer: `P(source = i | v)` from the fitted `beta`; it ignores earlier events."""

    def __init__(
        self, beta: np.ndarray, balances: Sequence[int], units: Sequence[int], whole_unit: int
    ) -> None:
        self.beta = beta
        self.balances = tuple(balances)
        self.units = tuple(units)
        self.whole_unit = whole_unit
        self.fallback_count = 0

    def posterior(self, v: int) -> np.ndarray:
        """Conditional-logit posterior over the `n` deposits for the next public amount `v`."""
        return predict(self.beta, *features(v, self.balances, self.units, self.whole_unit))

    def update(self, y: int) -> None:
        """No-op: the feature model is frozen, so the state never changes."""
        pass
