"""Shared test vectors for the Python and TypeScript implementations.

The vectors are built by exhaustive enumeration of every random-call tree used by
:func:`withdrawchan.walk`.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import fire

import withdrawchan as wc

if TYPE_CHECKING:
    from collections.abc import Callable

VECTORS_PATH = Path(__file__).resolve().parents[2] / "vectors.json"
EXAMPLES_PATH = VECTORS_PATH.parent / "examples.json"
USDT = 10**6

# ==== enumeration ====


def _scripted(prefix: list[int], calls: list[tuple[int, int]]) -> Callable[[int], int]:
    """Create a scripted random generator.

    The generator replays the provided prefix, then answers 0 for any remaining draw. Each
    draw logs the pair (`n`, `value`).

    Args:
        prefix: The scripted random values to replay.
        calls: The log of all draws, recorded as (`n`, `value`) pairs.

    Returns:
        A callable with the `rng` signature that replays `prefix`.
    """

    def rng(n: int) -> int:
        value = prefix[len(calls)] if len(calls) < len(prefix) else 0
        calls.append((n, value))
        return value

    return rng


def enumerate_walk(x: int, m: int | None, d: int, q: Fraction, p: Fraction) -> dict[int, Fraction]:
    """Enumerate the output distribution of :func:`withdrawchan.recommend`.

    Args:
        x: The need, the least amount to withdraw.
        m: The maximum, or None if the default is used.
        d: The rounding unit.
        q: The probability of climbing one rung.
        p: The probability of adding the random decimal.

    Returns:
        A mapping from output amount to its exact probability.
    """
    out: dict[int, Fraction] = {}
    stack: list[list[int]] = [[]]
    while stack:
        prefix = stack.pop()
        calls: list[tuple[int, int]] = []
        y = wc.recommend(x, m, d=d, q=q, p=p, rng=_scripted(prefix, calls))
        weight = Fraction(1)
        for n, _ in calls:
            weight /= n
        out[y] = out.get(y, Fraction(0)) + weight
        for i in range(len(prefix), len(calls)):
            base = [v for _, v in calls[:i]]
            stack.extend(base + [v] for v in range(1, calls[i][0]))
    return out


# ==== vectors ====

#: The worked examples shared by both test suites, with their asset scale.
EXAMPLES = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
EXAMPLE_SCALE = 10 ** EXAMPLES["decimals"]


def example_amount(amount: str) -> int:
    """Convert a worked-example amount to the asset's smallest unit.

    Args:
        amount: The decimal amount string from the example file.

    Returns:
        The integer amount in the asset's smallest unit.
    """
    return int(Fraction(amount) * EXAMPLE_SCALE)


def _cases() -> list[tuple[int, int | None, int, Fraction, Fraction]]:
    cases = [(example_amount(e["x"]), None if e["m"] is None else example_amount(e["m"]), USDT, wc.Q, wc.P)
             for e in EXAMPLES["examples"]]
    rng = random.Random(20261009)
    for _ in range(400):
        d = rng.choice((10, 100, USDT))
        q = rng.choice((Fraction(1), Fraction(1, 2)))
        p = Fraction(0) if d == USDT else rng.choice((Fraction(0), Fraction(1, 10), Fraction(1, 2), Fraction(1)))
        x = rng.randrange(1, 10**6) * (10**4 if d == USDT else 1)
        m = None if rng.random() < 0.5 else -(-x // d) * d + d * rng.randrange(0, max(1, 50 * x // d))
        cases.append((x, m, d, q, p))
    return cases


def _fraction(f: Fraction) -> str:
    """Render a Fraction as a reduced string of the form numerator/denominator."""
    return f"{f.numerator}/{f.denominator}"


def build() -> list[dict]:
    """Build all recorded test vectors.

    Each vector includes the inputs, the enumerated distribution, and one recorded random script
    together with the sampled output.

    Returns:
        A list of JSON-serializable dictionaries describing the vectors.
    """
    vectors = []
    rng = random.Random(1)
    for x, m, d, q, p in _cases():
        calls: list[tuple[int, int]] = []

        def record(n: int, calls: list[tuple[int, int]] = calls) -> int:
            value = rng.randrange(n)
            calls.append((n, value))
            return value

        y = wc.recommend(x, m, d=d, q=q, p=p, rng=record)
        dist = enumerate_walk(x, m, d, q, p)
        vectors.append({
            "x": str(x), "m": None if m is None else str(m), "d": str(d), "q": _fraction(q), "p": _fraction(p),
            "distribution": {str(k): _fraction(v) for k, v in sorted(dist.items())},
            "rng": [[str(n), str(v)] for n, v in calls], "y": str(y),
        })
    return vectors


def write(path: str = str(VECTORS_PATH)) -> str:
    """Rebuild the vectors and write them to the target path.

    Args:
        path: The destination file path.

    Returns:
        The path that was written.
    """
    Path(path).write_text(json.dumps({"vectors": build()}, indent=1) + "\n", encoding="utf-8", newline="\n")
    return path


if __name__ == "__main__":
    fire.Fire({"write": write})
