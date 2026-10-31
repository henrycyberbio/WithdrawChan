"""Comparison of two experiment outputs sharing a grid and a seed root: per-cell entropy bounds.

Notation local to this module.

- `base`, `alt`: the two outputs; `alt` is the varied run, one config value set elsewhere.
- `cell`: a `(rho, rule, m_model, adopters, observer, group)` row of the shared grid.
- `best`: the row family whose observer is the strongest public observer of a cell, chosen
  independently per side; the kind-aware `group` observer never competes for it.
- `H`: a cell's min-entropy upper bound, `-log2` of its pooled hit rate.

The two outputs must share their grid, seed root, replicate count, `N` and snapshot, so their cells
pair replicate by replicate; only the swept config values differ.  `H` pools every event of a cell
before the log, as :mod:`lab.report.summary` does, and each interval resamples replicate indices jointly, one
draw for both sides.
"""

from __future__ import annotations

import csv
import json
import math
import random
from fractions import Fraction
from pathlib import Path

from lab.report.summary import BOOTSTRAP, KIND_AWARE, SEED, bits, percentiles, scan

__all__ = ["compare"]

Row = dict[str, str | int | float]
# rho, rule, m_model, adopters, observer, group
Cell = tuple[str, str, str, str, str, str]
# rho, rule, m_model, adopters, group
Family = tuple[str, str, str, str, str]
# Per-replicate event statistics of one observer's cell: summed `top1` and event count.
Stats = dict[int, tuple[float, int]]
# One side's arm: per-replicate statistics aligned to a cell's shared replicates.
Arm = list[tuple[float, int]]

COMPARE_COLUMNS = ("rho", "rule", "m_model", "adopters", "observer", "group", "n_events_base", "n_events_alt",
                   "bits_base", "bits_alt", "diff_bits", "diff_bits_lo", "diff_bits_hi", "relative_bits",
                   "relative_bits_lo", "relative_bits_hi")

#: The `observer` value of the strongest-observer row family.
BEST = "best"


class CompareError(ValueError):
    """Two experiment outputs differ in a value their cells must share, so they cannot be paired."""

    def __init__(self, what: str, base: object, alt: object) -> None:
        super().__init__(f"{what} mismatch: base {base!r}, alt {alt!r}")


# ==== reading ====


def _run(out: Path) -> dict:
    """The `run.json` of an experiment output."""
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def _check(base: dict, alt: dict) -> None:
    """Raise unless both runs share their seed root, replicate count, `N`, snapshot and grid.

    The outputs must differ only in the swept config values, so a cell of one run pairs with the same
    cell of the other replicate by replicate.
    """
    fields = (("seed_root", base["config"]["seed_root"], alt["config"]["seed_root"]),
              ("replicates", base["config"]["replicates"], alt["config"]["replicates"]),
              ("N", base["N"], alt["N"]),
              ("snapshot_sha256", base["snapshot_sha256"], alt["snapshot_sha256"]))
    for name, left, right in fields:
        if left != right:
            raise CompareError(name, left, right)
    if set(base["paths"]) != set(alt["paths"]):
        raise CompareError("paths", len(base["paths"]), len(alt["paths"]))


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    """Every leaf of a config as `dotted.path -> value`; nested objects recurse, the rest are leaves."""
    if not isinstance(value, dict):
        return {prefix: value}
    leaves: dict[str, object] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        leaves.update(_leaves(item, path))
    return leaves


def _changed(base: dict, alt: dict) -> dict[str, dict[str, object]]:
    """The config keys whose value differs between the two runs, as `dotted.path -> {base, alt}`.

    `to_json` omits a field equal to its default, so a key may be present on one side only; the
    absent side reads `None`.
    """
    left = _leaves(base["config"])
    right = _leaves(alt["config"])
    return {key: {"base": left.get(key), "alt": right.get(key)} for key in sorted(set(left) | set(right))
            if left.get(key) != right.get(key)}


# ==== bounds ====


def _pooled(arm: Arm) -> float:
    """One arm's hit rate: its summed `top1` over its summed event counts."""
    return math.fsum(sum_top1 for sum_top1, _n in arm) / math.fsum(n for _s, n in arm)


def _diff(base_bits: float, alt_bits: float) -> float:
    """`H_alt - H_base`; equal bounds leave zero even when the bound is infinite."""
    return 0.0 if base_bits == alt_bits else alt_bits - base_bits


def _relative(base_bits: float, alt_bits: float) -> float:
    """`(H_alt - H_base) / H_base`; equal bounds leave zero, a zero or infinite baseline NaN."""
    if base_bits == alt_bits:
        return 0.0
    if base_bits == 0 or not (math.isfinite(base_bits) and math.isfinite(alt_bits)):
        return math.nan
    return (alt_bits - base_bits) / base_bits


def _bounds(arms: list[Arm]) -> tuple[float, int]:
    """The strongest arm's `(bits, n)`: the highest pooled hit rate, ties to the first arm."""
    pooled = [_pooled(arm) for arm in arms]
    index = min(range(len(arms)), key=lambda i: (-pooled[i], i))
    return bits(pooled[index]), sum(n for _s, n in arms[index])


def _draws(base: list[Arm], alt: list[Arm], rng: random.Random) -> tuple[list[float], list[float]]:
    """Joint replicate resample of both sides: one index draw per replicate, strongest arm kept.

    `base` and `alt` hold, per side, one arm per observer aligned to the shared replicates.  Every
    draw pools each arm on the drawn indices and keeps the highest hit rate, so the interval includes
    the choice of the strongest observer.  Returns the per-draw bit differences and relative changes.
    """
    count = len(base[0])
    diffs: list[float] = []
    relatives: list[float] = []
    for _ in range(BOOTSTRAP):
        picked = rng.choices(range(count), k=count)
        base_bits = bits(max(_pooled([arm[i] for i in picked]) for arm in base))
        alt_bits = bits(max(_pooled([arm[i] for i in picked]) for arm in alt))
        diffs.append(_diff(base_bits, alt_bits))
        relatives.append(_relative(base_bits, alt_bits))
    return diffs, relatives


# ==== cells ====


def _cell_key(cell: Cell) -> tuple[Fraction, str, str, str, str, str]:
    """Sort key of a compare row: the ratio as a number, the rest as text."""
    return (Fraction(cell[0]), cell[1], cell[2], cell[3], cell[4], cell[5])


def _family_key(family: Family) -> tuple[Fraction, str, str, str, str]:
    """Sort key of a family: the ratio as a number, the rest as text."""
    return (Fraction(family[0]), family[1], family[2], family[3], family[4])


def _row_key(row: Row) -> tuple[Fraction, str, str, str, str, str]:
    """Sort key of a written row: the ratio as a number, the rest as text."""
    return (Fraction(str(row["rho"])), str(row["rule"]), str(row["m_model"]), str(row["adopters"]),
            str(row["observer"]), str(row["group"]))


def _paired(base: Stats, alt: Stats) -> tuple[Arm, Arm]:
    """Both sides' statistics over the replicates they share, one arm each."""
    replicates = sorted(set(base) & set(alt))
    return [base[r] for r in replicates], [alt[r] for r in replicates]


def _row(cell: Cell, base: list[Arm], alt: list[Arm], rng: random.Random) -> Row:
    """One compare row: each side's strongest arm's bounds, their difference and joint intervals."""
    base_bits, base_n = _bounds(base)
    alt_bits, alt_n = _bounds(alt)
    diffs, relatives = _draws(base, alt, rng)
    diff_lo, diff_hi = percentiles(diffs)
    relative_lo, relative_hi = percentiles(relatives)
    return {"rho": cell[0], "rule": cell[1], "m_model": cell[2], "adopters": cell[3], "observer": cell[4],
            "group": cell[5], "n_events_base": base_n, "n_events_alt": alt_n, "bits_base": base_bits,
            "bits_alt": alt_bits, "diff_bits": _diff(base_bits, alt_bits), "diff_bits_lo": diff_lo,
            "diff_bits_hi": diff_hi, "relative_bits": _relative(base_bits, alt_bits),
            "relative_bits_lo": relative_lo, "relative_bits_hi": relative_hi}


def _single_rows(base: dict[Cell, Stats], alt: dict[Cell, Stats], rng: random.Random) -> list[Row]:
    """One row per cell present in both runs, both sides pooled over their shared replicates."""
    rows: list[Row] = []
    for cell in sorted(set(base) & set(alt), key=_cell_key):
        base_arm, alt_arm = _paired(base[cell], alt[cell])
        if base_arm:
            rows.append(_row(cell, [base_arm], [alt_arm], rng))
    return rows


def _families(cells: dict[Cell, Stats]) -> dict[Family, dict[str, Stats]]:
    """Group the public cells by their observer-free family, keeping each observer's statistics.

    The kind-aware observer (:data:`~lab.report.summary.KIND_AWARE`) has model knowledge the
    release observers lack; it is reported in its own rows but never competes for a `best` row, as
    in `summary`'s strongest-public-observer envelope.
    """
    grouped: dict[Family, dict[str, Stats]] = {}
    for (rho, rule, m_model, adopters, observer, group), stats in cells.items():
        if observer == KIND_AWARE:
            continue
        grouped.setdefault((rho, rule, m_model, adopters, group), {})[observer] = stats
    return grouped


def _shared_replicates(base: dict[str, Stats], alt: dict[str, Stats]) -> list[int]:
    """The replicates every observer of both sides of a family covers."""
    return sorted(set.intersection(*(set(stats) for stats in base.values()))
                  & set.intersection(*(set(stats) for stats in alt.values())))


def _arms(observers: dict[str, Stats], replicates: list[int]) -> list[Arm]:
    """Each observer's statistics over `replicates`, in observer-name order."""
    return [[stats[r] for r in replicates] for _name, stats in sorted(observers.items())]


def _best_rows(base: dict[Cell, Stats], alt: dict[Cell, Stats], rng: random.Random) -> list[Row]:
    """One row per family present in both runs, each side's strongest observer re-chosen per draw."""
    base_families, alt_families = _families(base), _families(alt)
    rows: list[Row] = []
    for family in sorted(set(base_families) & set(alt_families), key=_family_key):
        replicates = _shared_replicates(base_families[family], alt_families[family])
        if replicates:
            cell = (family[0], family[1], family[2], family[3], BEST, family[4])
            rows.append(_row(cell, _arms(base_families[family], replicates),
                             _arms(alt_families[family], replicates), rng))
    return rows


# ==== writing ====


def _text(value: str | int | float) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _label(out: Path) -> str:
    """The shareable name of an output directory, `parent/name`, carrying no absolute path."""
    return f"{out.parent.name}/{out.name}"


def _write(path: Path, rows: list[Row]) -> None:
    """Write `rows` as CSV with the fixed header; floats use `repr`."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COMPARE_COLUMNS)
        for row in rows:
            writer.writerow([_text(row[c]) for c in COMPARE_COLUMNS])


def compare(base: str, alt: str, *, dest: str) -> dict:
    """Compare two experiment outputs on a shared grid and seed root into `dest`.

    Writes `compare.csv`, one row per shared `(rho, rule, m_model, adopters, observer, group)` cell
    plus a `best` row family whose observer is each side's strongest public observer, giving both
    sides' min-entropy bounds and their joint-bootstrap difference and relative change; and
    `compare.json`, naming each run by its last two directory names (shareable, no absolute path)
    and recording the config keys that differ.  The outputs must come from the same grid, seed root,
    replicate count, `N` and snapshot.

    Args:
        base: The base experiment output directory.
        alt: The varied experiment output directory; only swept config values may differ from `base`.
        dest: The directory to write `compare.csv` and `compare.json` into.

    Returns:
        A mapping with `out`, the destination; `rows`, the number of written rows; and `changed`,
        the differing config keys in sorted order.

    Raises:
        :class:`CompareError`: If the runs differ in grid, seed root, replicate count, `N` or snapshot.
    """
    base_out, alt_out = Path(base), Path(alt)
    base_run, alt_run = _run(base_out), _run(alt_out)
    _check(base_run, alt_run)
    base_cells = scan(base_out)[0]
    alt_cells = scan(alt_out)[0]
    rng = random.Random(SEED)
    rows = _single_rows(base_cells, alt_cells, rng)
    rows.extend(_best_rows(base_cells, alt_cells, rng))
    rows.sort(key=_row_key)
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "compare.csv", rows)
    changed = _changed(base_run, alt_run)
    report = {"base": _label(base_out), "alt": _label(alt_out), "seed_root": base_run["config"]["seed_root"],
              "replicates": base_run["config"]["replicates"], "N": base_run["N"],
              "snapshot_sha256": base_run["snapshot_sha256"], "changed": changed, "rows": len(rows)}
    (folder / "compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"out": str(folder), "rows": len(rows), "changed": sorted(changed)}
