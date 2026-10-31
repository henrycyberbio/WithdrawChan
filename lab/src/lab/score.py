"""Per-event scoring: one row per (path, observer, public event).

Besides the event's `public_index`, `source`, `kind`, `x` and `y`, a row holds
`p_true`, the posterior of the true source; `true_rank`, its rank (1 is the top guess, ties
share their mean rank); `top1`, `1/t` when the true source is among `t` tied top guesses,
else 0; and `max_posterior`, the `V` in the per-observation min-entropy `-log2 V`.  The
notation is that of :mod:`lab`.
"""

from __future__ import annotations

import csv
import gzip
import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from lab.sim import Event

__all__ = ["COLUMNS", "event_row", "read_rows", "write_rows"]

COLUMNS = ("path", "observer", "public_index", "source", "kind", "x", "y", "p_true", "true_rank",
           "top1", "max_posterior")

_INT_COLUMNS = frozenset({"public_index", "source", "kind", "x", "y"})
_FLOAT_COLUMNS = frozenset({"p_true", "true_rank", "top1", "max_posterior"})

# ==== rows ====


def event_row(posterior: np.ndarray, event: Event, path: str, observer: str) -> dict[str, str | int | float]:
    """Score one public event for one observer; `event` supplies the amounts and the true source.

    Args:
        posterior: The observer's posterior over the deposits; read as float, not modified.
        event: The scored public event; it supplies `public_index`, `kind`, `x`, `y` and the
            true `source`.
        path: The path label written to the row.
        observer: The observer name written to the row.

    Returns:
        A dict keyed by exactly `COLUMNS`.  `y` is the event's public withdrawal amount;
        `p_true`, `true_rank` and `top1` use the simulated truth `event.source`.
        `max_posterior` is the largest posterior, independent of the true source.
    """
    posterior = np.asarray(posterior, dtype=float)
    top = float(np.max(posterior))
    true_p = float(posterior[event.source])
    above = int(np.count_nonzero(posterior > true_p))
    equal = int(np.count_nonzero(posterior == true_p))
    return {
        "path": path, "observer": observer, "public_index": event.public_index,
        "source": event.source, "kind": event.kind, "x": event.x, "y": event.y,
        "p_true": true_p, "true_rank": above + (equal + 1) / 2.0,
        "top1": (1.0 / int(np.count_nonzero(posterior == top))) if true_p == top else 0.0,
        "max_posterior": top,
    }


def _cell(value: str | int | float) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def write_rows(path: Path, rows: Sequence[dict[str, str | int | float]]) -> None:
    """Write `rows` as gzip CSV with the fixed header; floats use `repr` (bit-exact).

    The gzip timestamp is fixed at zero, but the header records the destination basename.
    Identical rows and header metadata produce identical bytes under the same gzip implementation.

    Args:
        path: The destination `.csv.gz` path.
        rows: The rows to write; their keys are `COLUMNS`, written in that column order as text.
    """
    with gzip.GzipFile(path, "wb", mtime=0) as gz, io.TextIOWrapper(gz, encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in COLUMNS])


def _typed(raw: dict[str, str]) -> dict[str, str | int | float]:
    out: dict[str, str | int | float] = {}
    for c in COLUMNS:
        text = raw[c]
        if c in _INT_COLUMNS:
            out[c] = int(text)
        elif c in _FLOAT_COLUMNS:
            out[c] = float(text)
        else:
            out[c] = text
    return out


def read_rows(path: Path) -> list[dict[str, str | int | float]]:
    """Inverse of :func:`write_rows`; values come back int / float / str.

    Args:
        path: A `.csv.gz` file written by :func:`write_rows`.

    Returns:
        One dict per data row, keyed by `COLUMNS`: the integer columns are `int`, the float
        columns are `float` and the rest are `str`.
    """
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        return [_typed(raw) for raw in csv.DictReader(fh)]
