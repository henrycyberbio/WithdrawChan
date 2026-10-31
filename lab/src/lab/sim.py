"""One pool path until every person is inactive.

Each attempt draws a deposit uniformly; if it is active, its kind's behaviour draws `(x, y)`
from an rng keyed only by `(seed, attempt_id)`, otherwise the attempt leaves no event.  The
notation is that of :mod:`lab`.  Seeds derive from `sha256` of their string parts, so results
do not depend on the number of worker processes.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from fractions import Fraction

__all__ = ["DRAIN", "Event", "KEEP", "NON", "adoption_membership", "derive_seed", "simulate"]

#: Person kinds (see :mod:`lab`): they index the behaviour tuple of :func:`lab.rules.model` and are
#: the `kind` values persisted in a path's `rows.csv.gz`; they must not be renumbered.
NON, KEEP, DRAIN = 0, 1, 2


@dataclass(frozen=True)
class Event:
    """One public withdrawal: `public_index` counts events, `attempt_id` counts attempts
    (events included); `source` and `kind` are the deposit and its kind; `x` the need, `y` the
    public withdrawal amount, `before`/`after` the deposit's balance around it."""

    public_index: int
    attempt_id: int
    source: int
    kind: int
    x: int
    y: int
    before: int
    after: int


def derive_seed(*parts: object) -> int:
    """First 8 bytes (big-endian) of `sha256("p4v2|" + "|".join(str(p) for p in parts))`.

    Args:
        *parts: The seed's parts; the `str` of each is joined with `|` in order, so their order and
            textual form matter.

    Returns:
        The integer seed.
    """
    text = "|".join(str(p) for p in ("p4v2", *parts))
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def adoption_membership(n: int, rho: Fraction, seed: int) -> tuple[bool, ...]:
    """Fixed hidden adoption: the first `K = floor(rho*n + 1/2)` of a seeded permutation.

    One permutation per `(n, seed)` makes the adopter set nested across `rho`.

    Args:
        n: The participant count.
        rho: The adoption ratio.
        seed: The permutation seed, fixed per run.

    Returns:
        A tuple of `n` flags, `True` for each adopter.
    """
    k = (2 * rho.numerator * n + rho.denominator) // (2 * rho.denominator)
    order = list(range(n))
    random.Random(seed).shuffle(order)
    adopted = set(order[:k])
    return tuple(i in adopted for i in range(n))


def simulate(model: Sequence, balances: Sequence[int], kinds: Sequence[int], seed: int) -> tuple[Event, ...]:
    """Run one path until no person is active.

    `model` is the behaviour tuple indexed by kind and `kinds` gives each deposit's kind.

    Args:
        model: The behaviour tuple of :func:`lab.rules.model`, indexed by kind; each entry exposes
            `active(r)` and `sample(r, rng) -> (x, y)`.
        balances: The initial public balances, one per participant.
        kinds: The kind of each participant, in the same order as `balances`.
        seed: The path seed; each attempt draws from an rng keyed by `(seed, attempt_id)`.

    Returns:
        The events in `public_index` order; attempts that leave no event are absent.
    """
    n = len(balances)
    remaining = list(balances)
    active = sum(1 for r, kind in zip(remaining, kinds, strict=True) if model[kind].active(r))
    events: list[Event] = []
    public_index = 0
    attempt_id = 0
    while active:
        source = random.Random(f"{seed}:{attempt_id}:source").randrange(n)
        kind = kinds[source]
        before = remaining[source]
        if not model[kind].active(before):
            attempt_id += 1
            continue
        x, y = model[kind].sample(before, random.Random(f"{seed}:{attempt_id}:demand"))
        after = before - y
        remaining[source] = after
        events.append(Event(public_index, attempt_id, source, kind, x, y, before, after))
        public_index += 1
        if not model[kind].active(after):
            active -= 1
        attempt_id += 1
    return tuple(events)
