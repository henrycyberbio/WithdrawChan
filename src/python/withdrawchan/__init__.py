"""WithdrawChan: Mixer withdrawal amount recommendation.

Given the need `x`, the algorithm first rounds it up to the nearest multiple of the rounding unit
`d`, then walks up a ladder of candidate amounts with probability `q` at each step. Finally, with
probability `p`, it adds the random decimal `u` uniform in `[0, d)`, provided the result stays
within the maximum `m`.

All amounts are integers in the asset's smallest unit, and `p` and `q` are exact fractions. The
TypeScript implementation follows the same procedure and is validated against identical test
vectors.
"""

from __future__ import annotations

import json
import secrets
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["P", "Q", "UNITS", "AmountError", "decimal_rungs", "default_m", "distribution", "recommend", "unit",
           "walk", "walk_distribution"]

# ==== defaults ====

#: Probability of climbing one rung.
Q = Fraction(1, 2)
#: Probability of adding the random decimal below one rounding unit.
P = Fraction(0)


# ==== units ====

#: Path of the shared units file, the only copy of the per-asset rounding units.
_UNITS_PATH = Path(__file__).resolve().parents[2] / "units.json"

#: Rounding unit per asset class, in whole tokens, read from :data:`_UNITS_PATH` at import.
UNITS: dict[str, Fraction] = {
    code: Fraction(str(value)) for code, value in json.loads(_UNITS_PATH.read_text(encoding="utf-8")).items()
}


def unit(code: str, decimals: int) -> int:
    """Return the rounding unit of an asset class in the asset's smallest unit.

    Args:
        code: The asset class, a key of :data:`UNITS`.
        decimals: The number of decimals in the asset's smallest unit.

    Returns:
        `UNITS[code]` scaled to the smallest unit, as an integer.

    Raises:
        KeyError: If `code` is not in :data:`UNITS`.
        ValueError: If `decimals` is too small for the result to be a whole number.
    """
    scaled = UNITS[code] * 10**decimals
    if scaled.denominator != 1:
        msg = f"unit({code!r}, {decimals}) is not a whole number of smallest units"
        raise ValueError(msg)
    return scaled.numerator


# ==== ladder ====


def _ceil_to(v: int, unit: int) -> int:
    return -(-v // unit) * unit


def default_m(x: int, d: int) -> int:
    """Return the default maximum amount.

    The default maximum is `2 * x` rounded up to the rounding unit `d`.

    Args:
        x: The need, the least amount to withdraw.
        d: The rounding unit.

    Returns:
        The smallest multiple of `d` not below `2 * x`.
    """
    return _ceil_to(2 * x, d)


def decimal_rungs(x: int, d: int, m: int) -> list[int]:
    """Build the decimal rung ladder for the walk.

    Rungs start at `x` rounded up to `d`, then that amount rounded up to one significant digit,
    then each higher power of ten, keeping those not above `m`.

    Args:
        x: The need, the least amount to withdraw.
        d: The rounding unit.
        m: The maximum, the largest amount to withdraw.

    Returns:
        Increasing candidate amounts not above `m`, starting at `x` rounded up to `d`.
    """
    c0 = _ceil_to(x, d)
    e = len(str(c0)) - 1
    rungs = [c0]
    for c in (_ceil_to(c0, 10**e), *(10**k for k in range(e + 1, len(str(m)) + 1))):
        if rungs[-1] < c <= m:
            rungs.append(c)
    return rungs


# ==== walk ====


def _coin(prob: Fraction, rng: Callable[[int], int]) -> bool:
    """Return whether a Bernoulli trial succeeds.

    Certain outcomes (`prob` 0 or 1) draw nothing from `rng`.

    Args:
        prob: The success probability.
        rng: A callable returning a uniform integer in `[0, n)`.

    Returns:
        True with probability `prob`; otherwise False.
    """
    if prob in (0, 1):
        return prob == 1
    return rng(prob.denominator) < prob.numerator


def _check_probs(q: Fraction, p: Fraction) -> None:
    """Raise :class:`AmountError` unless `q` and `p` are probabilities."""
    if not 0 <= q <= 1:
        raise AmountError(f"q={q} must satisfy 0 <= q <= 1")
    if not 0 <= p <= 1:
        raise AmountError(f"p={p} must satisfy 0 <= p <= 1")


def walk(rungs: Sequence[int], m: int, *, d: int, q: Fraction = Q, p: Fraction = P,
         rng: Callable[[int], int] = secrets.randbelow) -> int:
    """Draw one recommended amount from the rung ladder.

    The walk climbs with probability `q` at each rung, then adds the random decimal with
    probability `p` if the result stays within `m`. `rng` is called once per uncertain coin and
    once for the random decimal, in walk order.

    Args:
        rungs: Increasing candidate amounts not above `m`.
        m: The maximum, the largest amount to withdraw.
        d: The rounding unit.
        q: The probability of climbing one rung.
        p: The probability of adding the random decimal.
        rng: A callable returning a uniform integer in `[0, n)`.

    Returns:
        The recommended amount.

    Raises:
        AmountError: If `q` or `p` is not a probability, or `rungs` is empty.
    """
    _check_probs(q, p)
    if not rungs:
        raise AmountError("rungs must not be empty")
    j = 0
    while j + 1 < len(rungs) and _coin(q, rng):
        j += 1
    y = rungs[j]
    if _coin(p, rng):
        u = rng(d)
        if y + u <= m:
            y += u
    return y


def walk_distribution(rungs: Sequence[int], m: int, *, d: int, q: Fraction = Q, p: Fraction = P) -> dict[int, Fraction]:
    """Return the exact output distribution of :func:`walk`.

    Args:
        rungs: Increasing candidate amounts not above `m`.
        m: The maximum, the largest amount to withdraw.
        d: The rounding unit.
        q: The probability of climbing one rung.
        p: The probability of adding the random decimal.

    Returns:
        A mapping from output amount to its exact probability. Zero-probability outputs are
        omitted.

    Raises:
        AmountError: If `q` or `p` is not a probability.
    """
    _check_probs(q, p)
    out: dict[int, Fraction] = {}
    reach = Fraction(1)
    for j, v in enumerate(rungs):
        stop = reach if j == len(rungs) - 1 else reach * (1 - q)
        reach *= q
        if not stop:
            continue
        top = min(v + d - 1, m) if p else v
        out[v] = out.get(v, Fraction(0)) + stop * ((1 - p) + p * (d - (top - v)) / d)
        for y in range(v + 1, top + 1):
            out[y] = out.get(y, Fraction(0)) + stop * p / d
    return out


# ==== recommendation ====


class AmountError(ValueError):
    """Raised when the need, the maximum, the rounding unit, or a probability is invalid."""

    def __init__(self, message: str) -> None:
        """Initialize the AmountError with a description of the invalid input.

        Args:
            message: Which input is invalid and what it must satisfy.
        """
        super().__init__(message)


def _pow10(d: int) -> bool:
    """Return whether `d` is a positive power of ten."""
    return d > 0 and 10 ** (len(str(d)) - 1) == d


def _limit(x: int, m: int | None, d: int) -> int:
    if x <= 0 or not _pow10(d) or (m is not None and (m % d or m < x)):
        raise AmountError(f"x={x} must be positive, d={d} a power of ten, and m={m} a multiple of d at least x")
    return default_m(x, d) if m is None else m


def recommend(x: int, m: int | None = None, *, d: int, q: Fraction = Q, p: Fraction = P,
              rng: Callable[[int], int] = secrets.randbelow) -> int:
    """Recommend a withdrawal amount.

    Args:
        x: The need, the least amount to withdraw.
        m: The maximum, the largest amount to withdraw. If omitted, :func:`default_m` is used.
        d: The rounding unit.
        q: The probability of climbing one rung.
        p: The probability of adding the random decimal.
        rng: A callable returning a uniform integer in `[0, n)`.

    Returns:
        One recommended amount.

    Raises:
        AmountError: If `x <= 0`, `d` is not a positive power of ten, `m` is not a multiple of
            `d` at least `x`, or `q` or `p` is not a probability.
    """
    maximum = _limit(x, m, d)
    return walk(decimal_rungs(x, d, maximum), maximum, d=d, q=q, p=p, rng=rng)


def distribution(x: int, m: int | None = None, *, d: int, q: Fraction = Q, p: Fraction = P) -> dict[int, Fraction]:
    """Return the exact output distribution of :func:`recommend`.

    Args:
        x: The need, the least amount to withdraw.
        m: The maximum, the largest amount to withdraw. If omitted, :func:`default_m` is used.
        d: The rounding unit.
        q: The probability of climbing one rung.
        p: The probability of adding the random decimal.

    Returns:
        A mapping from output amount to its exact probability.

    Raises:
        AmountError: If `x <= 0`, `d` is not a positive power of ten, `m` is not a multiple of
            `d` at least `x`, or `q` or `p` is not a probability.
    """
    maximum = _limit(x, m, d)
    return walk_distribution(decimal_rungs(x, d, maximum), maximum, d=d, q=q, p=p)
