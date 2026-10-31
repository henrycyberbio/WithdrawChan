"""Summary tables over an experiment output: per-observer hit rates and their entropy bounds.

Notation local to this module.

- `group`: an event's kind, `non` / `keep` / `drain`; a paired contrast runs on the adopter
  group of a path, the events of its `adopters` kind.
- kind-aware observer: the `group` observer knows the rule and the path's KEEP/DRAIN stopping
  policy, so it is not a public observer.  `groups` and `paired` report it, the default `best`
  never considers it, and it draws from its own bootstrap streams so that adding it leaves every
  public row's interval untouched.  An explicit `best_observers` set may name it: that table is a
  combined envelope over the named observers, not a public-only one.
- `hit`: the mean `top1` pooled over a cell's events.
- `minuend`, `subtrahend`: the two values of the contrasted factor, the first minus the second;
  `fixed` is the value of the factor held fixed.
- `cost`: `(y - x) / x` of one adopter event, the over-withdrawal of the rounding relative to the need.
"""

from __future__ import annotations

import csv
import json
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

from lab.score import read_rows
from lab.sim import DRAIN, KEEP, NON

__all__ = ["BOOTSTRAP", "KIND_AWARE", "SEED", "bits", "percentiles", "scan", "summarize"]

Row = dict[str, str | int | float]
# rho, rule, m_model, adopters, observer, group
GroupsKey = tuple[str, str, str, str, str, str]
# replicate, rho, rule, m_model, adopters, observer
PathKey = tuple[int, str, str, str, str, str]
# contrast, minuend, subtrahend, rho, fixed, adopters, observer
PairKey = tuple[str, str, str, str, str, str, str]
# rho, rule, m_model, adopters
CostKey = tuple[str, str, str, str]

GROUPS_COLUMNS = ("rho", "rule", "m_model", "adopters", "observer", "group", "n_events", "hit_rate",
                  "bits", "hit_lo", "hit_hi", "bits_lo", "bits_hi", "point_bits_mean", "point_bits_q10")
PAIRED_COLUMNS = ("contrast", "minuend", "subtrahend", "rho", "fixed", "adopters", "observer", "n_pairs", "diff_hit",
                  "diff_lo", "diff_hi", "relative_hit", "relative_lo", "relative_hi",
                  "relative_bits", "relative_bits_lo", "relative_bits_hi")
COST_COLUMNS = ("rho", "rule", "m_model", "adopters", "n_events", "n_final", "same_share", "cost_mean",
                "cost_median", "cost_p90", "withdrawals_per_adopter")

#: Resamples behind every interval.
BOOTSTRAP = 2000
#: Seed of the bootstrap generator; the pair fixes the output across runs.
SEED = 20261010

_GROUP = {NON: "non", KEEP: "keep", DRAIN: "drain"}
#: The one observer with non-public model knowledge: it is told the rule and the path-wide
#: KEEP/DRAIN stopping policy.  It is a kind-aware attack comparison, not a candidate for the
#: strongest-public-observer envelope of `best.csv`.
KIND_AWARE = "group"
#: Marker of the combined statistics in `paired.csv`: a statistic over the chosen candidates, not
#: an observer of the experiment.
_BEST = "best"
#: Person kind of the adopter events of a spec's `adopters` field.
_ADOPTER_KIND = {"keep": KEEP, "drain": DRAIN}
_M_MODEL_PAIRS = (("balance", "double"),)
_RULE_PAIRS = (("popstep", "decstep"), ("popsmooth", "decsmooth"))


# ==== entropy ====


def bits(hit: float) -> float:
    """The min-entropy upper bound of a hit rate, `-log2(hit)`.

    Args:
        hit: The pooled hit rate of a cell, in `[0, 1]`.

    Returns:
        `-log2(hit)`, or `inf` when `hit` is 0.
    """
    return math.inf if hit == 0.0 else -math.log2(hit)


# ==== bootstrap ====


def percentiles(draws: list[float]) -> tuple[float, float]:
    """The 2.5 and 97.5 percentiles of a bootstrap's draws.

    Args:
        draws: One value per resample.

    Returns:
        The lower then the upper percentile.
    """
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _pooled_interval(stats: list[tuple[float, int]], rng: random.Random) -> tuple[float, float]:
    """The 95% interval of a cell's hit rate: resample the replicate indices and pool their events."""
    count = len(stats)
    draws: list[float] = []
    for _ in range(BOOTSTRAP):
        picked = rng.choices(range(count), k=count)
        draws.append(math.fsum(stats[i][0] for i in picked) / math.fsum(stats[i][1] for i in picked))
    return percentiles(draws)


def _diff_interval(diffs: list[float], rng: random.Random) -> tuple[float, float]:
    """The 95% interval of a mean difference: resample the per-replicate differences."""
    count = len(diffs)
    draws = [math.fsum(rng.choices(diffs, k=count)) / count for _ in range(BOOTSTRAP)]
    return percentiles(draws)


# ==== experiment output ====


def _folders(out: Path) -> list[Path]:
    """The path directories of an experiment output: those holding `path.json`."""
    return sorted(folder for folder in out.iterdir() if (folder / "path.json").is_file())


#: How many differing directory names an error message spells out.
_NAMED = 3


def _few(names: list[str]) -> str:
    """The first few names of a sorted set, then the count of the rest."""
    return ", ".join(names[:_NAMED]) + (f", and {len(names) - _NAMED} more" if len(names) > _NAMED else "")


def _check_paths(out: Path) -> None:
    """Raise unless the path directories of `out` are exactly the ones its `run.json` records.

    `run.json` is written last, so a present one is a complete manifest: an output missing some of
    its directories would otherwise summarize into a full-looking table over fewer replicates.

    Raises:
        ValueError: Naming a few of the missing and of the extra directories, with both counts.
    """
    manifest = out / "run.json"
    if not manifest.is_file():
        return
    recorded = set(json.loads(manifest.read_text(encoding="utf-8"))["paths"])
    found = {folder.name for folder in _folders(out)}
    missing, extra = sorted(recorded - found), sorted(found - recorded)
    if not (missing or extra):
        return
    parts = [f"{len(names)} {label} ({_few(names)})"
             for names, label in ((missing, "missing"), (extra, "extra")) if names]
    msg = f"{out}: path directories differ from run.json: {'; '.join(parts)}"
    raise ValueError(msg)


def scan(out: Path) -> tuple[dict[GroupsKey, dict[int, tuple[float, int]]], dict[GroupsKey, list[float]],
                              dict[PathKey, tuple[float, int]]]:
    """One pass over every path of `out`.

    Args:
        out: An experiment output; its path directories are those holding `path.json`.

    Returns:
        Per group cell, the per-replicate `(sum top1, n)` and the pooled `max_posterior` values; and,
        per path, the `(sum top1, n)` over its adopter events.
    """
    cells: dict[GroupsKey, dict[int, tuple[float, int]]] = {}
    maxes: dict[GroupsKey, list[float]] = {}
    hits: dict[PathKey, tuple[float, int]] = {}
    for folder in _folders(out):
        spec = json.loads((folder / "path.json").read_text(encoding="utf-8"))["spec"]
        replicate, rho, rule = spec["replicate"], spec["rho"], spec["rule"]
        m_model, adopters = spec["m_model"], spec["adopters"]
        for row in read_rows(folder / "rows.csv.gz"):
            observer, group, top1 = row["observer"], _GROUP[row["kind"]], float(row["top1"])
            key = (rho, rule, m_model, adopters, observer, group)
            sum_top1, n = cells.setdefault(key, {}).get(replicate, (0.0, 0))
            cells[key][replicate] = (sum_top1 + top1, n + 1)
            maxes.setdefault(key, []).append(float(row["max_posterior"]))
            if group == adopters:
                path_key = (replicate, rho, rule, m_model, adopters, observer)
                hit_sum, hit_n = hits.get(path_key, (0.0, 0))
                hits[path_key] = (hit_sum + top1, hit_n + 1)
    return cells, maxes, hits


# ==== groups ====


def _groups_key(key: GroupsKey) -> tuple[Fraction, str, str, str, str, str]:
    """Sort key of a group cell: the ratio as a number, the rest as text."""
    return (Fraction(key[0]), key[1], key[2], key[3], key[4], key[5])


def _group_row_key(row: Row) -> tuple[Fraction, str, str, str, str, str]:
    """Sort key of a `groups` row: the ratio as a number, the rest as text."""
    return _groups_key((str(row["rho"]), str(row["rule"]), str(row["m_model"]), str(row["adopters"]),
                        str(row["observer"]), str(row["group"])))


def _groups(cells: dict[GroupsKey, dict[int, tuple[float, int]]], maxes: dict[GroupsKey, list[float]],
            rng: random.Random) -> list[Row]:
    """One row per (rho, rule, m_model, adopters, observer, group), sorted by its key columns."""
    rows: list[Row] = []
    for key in sorted(cells, key=_groups_key):
        stats = [cells[key][replicate] for replicate in sorted(cells[key])]
        n = sum(count for _s, count in stats)
        hit = math.fsum(s for s, _count in stats) / n
        hit_lo, hit_hi = _pooled_interval(stats, rng)
        point = [bits(max_p) for max_p in maxes[key]]
        rows.append({
            "rho": key[0], "rule": key[1], "m_model": key[2], "adopters": key[3], "observer": key[4],
            "group": key[5], "n_events": n, "hit_rate": hit, "bits": bits(hit), "hit_lo": hit_lo,
            "hit_hi": hit_hi, "bits_lo": bits(hit_hi), "bits_hi": bits(hit_lo),
            "point_bits_mean": math.fsum(point) / len(point),
            "point_bits_q10": float(np.quantile(point, 0.10)),
        })
    return rows


def _best_key(key: tuple[str, str, str, str, str]) -> tuple[Fraction, str, str, str, str]:
    return (Fraction(key[0]), key[1], key[2], key[3], key[4])


def _max_interval(stats: list[list[tuple[float, int]]], rng: random.Random) -> tuple[float, float]:
    """The 95% interval of the highest hit rate over observers.

    `stats` holds, per observer, the per-replicate `(sum top1, n)` in one shared replicate order.
    Each draw resamples the replicate indices once, pools every observer's events on them and keeps
    the highest pooled hit rate.
    """
    count = len(stats[0])
    draws: list[float] = []
    for _ in range(BOOTSTRAP):
        picked = rng.choices(range(count), k=count)
        draws.append(max(math.fsum(per[i][0] for i in picked) / math.fsum(per[i][1] for i in picked)
                         for per in stats))
    return percentiles(draws)


def _best(groups: list[Row], cells: dict[GroupsKey, dict[int, tuple[float, int]]],
          rng: random.Random) -> list[Row]:
    """Per (rho, rule, m_model, adopters, group), the strongest observer: the highest `hit_rate`.

    `observer`, `n_events`, `hit_rate`, `bits` and the point-bits columns are the winning
    observer's row of `groups`; ties go to the first observer name in sorted order.  `hit_lo` and
    `hit_hi` (and the bits bounds) come from :func:`_max_interval` over every observer of the cell,
    so the interval includes the choice of the winner.

    `cells` holds public observers only: the kind-aware `group` observer is never a candidate, so a
    cell it alone observes has no `best` row at all.  Pass `groups` filtered the same way.
    """
    best: dict[tuple[str, str, str, str, str], Row] = {}
    for row in groups:
        key = (row["rho"], row["rule"], row["m_model"], row["adopters"], row["group"])
        current = best.get(key)
        if current is None or row["hit_rate"] > current["hit_rate"] or (
                row["hit_rate"] == current["hit_rate"] and row["observer"] < current["observer"]):
            best[key] = row
    observers: dict[tuple[str, str, str, str, str], list[list[tuple[float, int]]]] = {}
    for key in sorted(cells, key=_groups_key):
        per_replicate = cells[key]
        cell = (key[0], key[1], key[2], key[3], key[5])
        observers.setdefault(cell, []).append([per_replicate[replicate] for replicate in sorted(per_replicate)])
    rows: list[Row] = []
    for key in sorted(best, key=_best_key):
        hit_lo, hit_hi = _max_interval(observers[key], rng)
        rows.append({**best[key], "hit_lo": hit_lo, "hit_hi": hit_hi, "bits_lo": bits(hit_hi),
                     "bits_hi": bits(hit_lo)})
    return rows


def _select_best_observers(cells: dict[GroupsKey, dict[int, tuple[float, int]]],
                           best_observers: object) -> tuple[str, ...]:
    """Validate an explicit `best_observers` set against the scanned groups cells.

    The set must be a non-empty list or tuple of distinct names, all observed in the input, and
    every cell must hold every candidate with the same replicate IDs and the same per-replicate
    event count.  A bare string is not a list, and an unknown name is not silently dropped.

    Args:
        cells: The scanned per-cell sufficient statistics of every observer.
        best_observers: The caller's candidate collection.

    Returns:
        The candidates in the caller's order.

    Raises:
        ValueError: If the set is malformed, names an observer of no cell, or is incomplete in a
            cell or inconsistent in a cell's replicate IDs or per-replicate `n`.
    """
    if not isinstance(best_observers, (list, tuple)):
        msg = ("best_observers must be a list or tuple of observer names, "
               f"not {type(best_observers).__name__}")
        raise ValueError(msg)  # noqa: TRY004 - the published contract rejects a bare string as such
    names = list(best_observers)
    if not names:
        msg = "best_observers names no observer"
        raise ValueError(msg)
    if any(not isinstance(name, str) or not name for name in names):
        msg = f"best_observers holds a non-name: {names!r}"
        raise ValueError(msg)
    if len(set(names)) != len(names):
        msg = f"best_observers repeats a name: {names!r}"
        raise ValueError(msg)
    known = {key[4] for key in cells}
    unknown = [name for name in names if name not in known]
    if unknown:
        msg = f"best_observers names observers absent from the input: {unknown}"
        raise ValueError(msg)
    present: dict[tuple[str, str, str, str, str], dict[str, dict[int, int]]] = {}
    # Register every observed cell first, not only the cells a candidate touches: a cell holding
    # none of the candidates is an incomplete cell below, never a silently dropped one.
    for key in cells:
        present.setdefault((key[0], key[1], key[2], key[3], key[5]), {})
    for key, replicates in cells.items():
        if key[4] in names:
            present[(key[0], key[1], key[2], key[3], key[5])][key[4]] = {
                replicate: count for replicate, (_sum_top1, count) in replicates.items()}
    for cell, candidates in present.items():
        missing = [name for name in names if name not in candidates]
        if missing:
            msg = f"best_observers candidates missing from cell {cell}: {missing}"
            raise ValueError(msg)
        reference = candidates[names[0]]
        for name in names[1:]:
            if candidates[name] != reference:
                msg = f"cell {cell}: {name} differs in replicate IDs or per-replicate event counts"
                raise ValueError(msg)
    return tuple(names)


def _candidate_best(cells: dict[GroupsKey, dict[int, tuple[float, int]]],
                    maxes: dict[GroupsKey, list[float]], candidates: tuple[str, ...],
                    rng: random.Random) -> list[Row]:
    """Per (rho, rule, m_model, adopters, group), the best of the named candidates.

    Each candidate's repeated experiments are merged first: its pooled hit rate is the total `top1`
    over the total event count of the cell, not a mean of per-replicate rates and not a
    per-replicate maximum.  The winner is the highest pooled rate, ties going to the first name in
    sorted order; its own row supplies the point estimate.  `hit_lo` and `hit_hi` come from
    :func:`_max_interval` over the candidates, whose replicate IDs and per-replicate `n` the
    validation made equal, so each draw resamples one shared index set and re-chooses the winner.

    Args:
        cells: The scanned per-cell sufficient statistics of every observer.
        maxes: The scanned per-cell `max_posterior` values.
        candidates: A validated, non-empty candidate set.
        rng: The generator of the `best` bootstrap, seeded with :data:`SEED`.

    Returns:
        One row per cell, sorted by the `best` key columns.
    """
    by_cell: dict[tuple[str, str, str, str, str], dict[str, dict[int, tuple[float, int]]]] = {}
    for key, replicates in cells.items():
        if key[4] in candidates:
            cell = (key[0], key[1], key[2], key[3], key[5])
            by_cell.setdefault(cell, {})[key[4]] = replicates
    rows: list[Row] = []
    for cell in sorted(by_cell, key=_best_key):
        per_observer = by_cell[cell]
        rates: dict[str, float] = {}
        for observer, replicates in per_observer.items():
            stats = [replicates[replicate] for replicate in sorted(replicates)]
            rates[observer] = math.fsum(sum_top1 for sum_top1, _n in stats) / sum(n for _s, n in stats)
        winner = min(rates, key=lambda name: (-rates[name], name))
        key = (cell[0], cell[1], cell[2], cell[3], winner, cell[4])
        stats = [cells[key][replicate] for replicate in sorted(cells[key])]
        n = sum(count for _sum_top1, count in stats)
        hit = math.fsum(sum_top1 for sum_top1, _count in stats) / n
        hit_lo, hit_hi = _max_interval(
            [[per_observer[name][replicate] for replicate in sorted(per_observer[name])]
             for name in sorted(per_observer)], rng)
        point = [bits(max_posterior) for max_posterior in maxes[key]]
        rows.append({
            "rho": cell[0], "rule": cell[1], "m_model": cell[2], "adopters": cell[3],
            "observer": winner, "group": cell[4], "n_events": n, "hit_rate": hit, "bits": bits(hit),
            "hit_lo": hit_lo, "hit_hi": hit_hi, "bits_lo": bits(hit_hi), "bits_hi": bits(hit_lo),
            "point_bits_mean": math.fsum(point) / len(point),
            "point_bits_q10": float(np.quantile(point, 0.10)),
        })
    return rows


# ==== paired contrasts ====


def _context_key(context: tuple[int, str, str, str, str]) -> tuple[int, Fraction, str, str, str]:
    return (context[0], Fraction(context[1]), context[2], context[3], context[4])


def _pairs_key(key: PairKey) -> tuple[str, str, str, Fraction, str, str, str]:
    """Sort key of a paired row: the ratio as a number, the rest as text."""
    return (key[0], key[1], key[2], Fraction(key[3]), key[4], key[5], key[6])


def _paired(hits: dict[PathKey, tuple[float, int]], rng: random.Random, label: str = "") -> list[Row]:
    """The two contrast families, one row per (contrast, minuend, subtrahend, rho, fixed, adopters, observer).

    `label` names the bootstrap streams of a separate table share: the empty label keeps the
    module's original seeds, a label keeps the kind-aware observer's draws off the public stream.
    """
    models: dict[tuple[int, str, str, str, str], dict[str, float]] = {}
    rules: dict[tuple[int, str, str, str, str], dict[str, float]] = {}
    for (replicate, rho, rule, m_model, adopters, observer), (sum_top1, n) in hits.items():
        hit = sum_top1 / n
        models.setdefault((replicate, rho, rule, adopters, observer), {})[m_model] = hit
        rules.setdefault((replicate, rho, m_model, adopters, observer), {})[rule] = hit
    diffs: dict[PairKey, list[float]] = {}
    pairs: dict[PairKey, list[tuple[float, float]]] = {}
    for context in sorted(models, key=_context_key):
        factors = models[context]
        for minuend, subtrahend in _M_MODEL_PAIRS:
            if minuend in factors and subtrahend in factors:
                key = ("m_model", minuend, subtrahend, context[1], context[2], context[3], context[4])
                diffs.setdefault(key, []).append(factors[minuend] - factors[subtrahend])
                pairs.setdefault(key, []).append((factors[minuend], factors[subtrahend]))
    for context in sorted(rules, key=_context_key):
        factors = rules[context]
        for minuend, subtrahend in _RULE_PAIRS:
            if minuend in factors and subtrahend in factors:
                key = ("rule", minuend, subtrahend, context[1], context[2], context[3], context[4])
                diffs.setdefault(key, []).append(factors[minuend] - factors[subtrahend])
                pairs.setdefault(key, []).append((factors[minuend], factors[subtrahend]))
    rows: list[Row] = []
    relative_rng = random.Random(SEED if not label else f"{SEED}|{label}|relative")
    entropy_rng = random.Random(SEED if not label else f"{SEED}|{label}|entropy")
    for key in sorted(diffs, key=_pairs_key):
        values = diffs[key]
        diff_lo, diff_hi = _diff_interval(values, rng)
        count = len(pairs[key])
        relative_lo, relative_hi = percentiles([
            _relative_change(relative_rng.choices(pairs[key], k=count)) for _ in range(BOOTSTRAP)])
        entropy_pairs = _entropy_pairs(hits, key)
        entropy_lo, entropy_hi = percentiles([
            _relative_entropy(entropy_rng.choices(entropy_pairs, k=count)) for _ in range(BOOTSTRAP)])
        rows.append({
            "contrast": key[0], "minuend": key[1], "subtrahend": key[2], "rho": key[3], "fixed": key[4],
            "adopters": key[5], "observer": key[6], "n_pairs": len(values), "diff_hit": math.fsum(values) / len(values),
            "diff_lo": diff_lo, "diff_hi": diff_hi,
            "relative_hit": _relative_change(pairs[key]), "relative_lo": relative_lo,
            "relative_hi": relative_hi,
            "relative_bits": _relative_entropy(entropy_pairs), "relative_bits_lo": entropy_lo,
            "relative_bits_hi": entropy_hi,
        })
    return rows


def _need_context_key(context: tuple[int, str, str, str]) -> tuple[int, Fraction, str, str]:
    """Sort key of a need-family context: the ratio as a number, the rest as text."""
    return (context[0], Fraction(context[1]), context[2], context[3])


def _need_rows(hits: dict[PathKey, tuple[float, int]], label: str = "") -> list[Row]:
    """The need contrast family: every rounding rule at every maximum model minus the need baseline.

    The baseline is the `need` path at `double`, the only maximum model that rule has; a pair
    matches on replicate, rho and adopters, as in the other families.  Its generators seed from
    :data:`SEED` and their own labels, so the family cannot shift the other bootstrap streams;
    `label` prefixes those labels for a table share of its own.
    """
    contexts: dict[tuple[int, str, str, str], dict[tuple[str, str], tuple[float, int]]] = {}
    for (replicate, rho, rule, m_model, adopters, observer), stats in hits.items():
        contexts.setdefault((replicate, rho, adopters, observer), {})[(rule, m_model)] = stats
    diffs: dict[PairKey, list[float]] = {}
    pairs: dict[PairKey, list[tuple[float, float]]] = {}
    matched: dict[PairKey, list[tuple[tuple[float, int], tuple[float, int]]]] = {}
    for context in sorted(contexts, key=_need_context_key):
        factors = contexts[context]
        baseline = factors.get(("need", "double"))
        if baseline is None:
            continue
        for rule, m_model in sorted(factors):
            if rule == "need":
                continue
            comparison = factors[rule, m_model]
            key = ("need", rule, "need", context[1], m_model, context[2], context[3])
            diffs.setdefault(key, []).append(comparison[0] / comparison[1] - baseline[0] / baseline[1])
            pairs.setdefault(key, []).append((comparison[0] / comparison[1], baseline[0] / baseline[1]))
            matched.setdefault(key, []).append((comparison, baseline))
    rows: list[Row] = []
    seed = f"{SEED}|{label}" if label else str(SEED)
    diff_rng = random.Random(f"{seed}|need|diff")
    relative_rng = random.Random(f"{seed}|need|relative")
    entropy_rng = random.Random(f"{seed}|need|entropy")
    for key in sorted(diffs, key=_pairs_key):
        values = diffs[key]
        diff_lo, diff_hi = _diff_interval(values, diff_rng)
        count = len(pairs[key])
        relative_lo, relative_hi = percentiles([
            _relative_change(relative_rng.choices(pairs[key], k=count)) for _ in range(BOOTSTRAP)])
        entropy_lo, entropy_hi = percentiles([
            _relative_entropy(entropy_rng.choices(matched[key], k=count)) for _ in range(BOOTSTRAP)])
        rows.append({
            "contrast": key[0], "minuend": key[1], "subtrahend": key[2], "rho": key[3], "fixed": key[4],
            "adopters": key[5], "observer": key[6], "n_pairs": len(values),
            "diff_hit": math.fsum(values) / len(values), "diff_lo": diff_lo, "diff_hi": diff_hi,
            "relative_hit": _relative_change(pairs[key]), "relative_lo": relative_lo, "relative_hi": relative_hi,
            "relative_bits": _relative_entropy(matched[key]), "relative_bits_lo": entropy_lo,
            "relative_bits_hi": entropy_hi,
        })
    return rows


def _entropy_pairs(hits: dict[PathKey, tuple[float, int]], key: PairKey
                   ) -> list[tuple[tuple[float, int], tuple[float, int]]]:
    """Matched replicate sufficient statistics, comparison then baseline."""
    contrast, minuend, subtrahend, rho, fixed, adopters, observer = key
    replicates = sorted({k[0] for k in hits if (k[1], k[4], k[5]) == (rho, adopters, observer)})
    pairs = []
    for replicate in replicates:
        if contrast == "m_model":
            comparison = (replicate, rho, fixed, minuend, adopters, observer)
            baseline = (replicate, rho, fixed, subtrahend, adopters, observer)
        else:
            comparison = (replicate, rho, minuend, fixed, adopters, observer)
            baseline = (replicate, rho, subtrahend, fixed, adopters, observer)
        if comparison in hits and baseline in hits:
            pairs.append((hits[comparison], hits[baseline]))
    return pairs


def _relative_entropy(pairs: list[tuple[tuple[float, int], tuple[float, int]]]) -> float:
    """Relative entropy-bound change after pooling each arm's events, then taking logs.

    Zero-hit arms or a zero-bit baseline make this ratio undefined, retained as NaN.
    """
    comparison = bits(math.fsum(a[0] for a, _b in pairs) / sum(a[1] for a, _b in pairs))
    baseline = bits(math.fsum(b[0] for _a, b in pairs) / sum(b[1] for _a, b in pairs))
    if baseline == 0 or not (math.isfinite(comparison) and math.isfinite(baseline)):
        return math.nan
    return (comparison - baseline) / baseline


def _relative_change(pairs: list[tuple[float, float]]) -> float:
    """Ratio of paired mean hit rates minus one; undefined for a zero baseline.

    The baseline is the subtrahend. Both arms retain equal replicate weights, as in `diff_hit`;
    this is not the mean of per-replicate ratios. Undefined bootstrap draws make the interval NaN.
    """
    baseline = math.fsum(base for _value, base in pairs)
    return math.fsum(value - base for value, base in pairs) / baseline if baseline else math.nan


def _arm_rate(replicates: dict[int, tuple[tuple[float, int], tuple[float, int]]], picked: list[int],
              arm: int) -> float:
    """One candidate's pooled hit rate of one arm over the picked replicates."""
    return (math.fsum(replicates[replicate][arm][0] for replicate in picked)
            / sum(replicates[replicate][arm][1] for replicate in picked))


def _best_entropy_change(comparison: float, baseline: float) -> float:
    """Relative change of the entropy bounds of two pooled hit rates.

    A zero-hit arm has an infinite bound and a baseline at 1 has a zero-bit bound; either leaves
    the ratio undefined as NaN.
    """
    top, base = bits(comparison), bits(baseline)
    if not (math.isfinite(top) and math.isfinite(base)) or base == 0:
        return math.nan
    return (top - base) / base


def _best_paired(hits: dict[PathKey, tuple[float, int]], candidates: tuple[str, ...]) -> list[Row]:
    """The combined `observer="best"` rows of the three paired families.

    For every point and every draw each arm is pooled over the candidate's replicates first, then
    the highest arm rate over the candidates is taken: `A` and `B` may come from different
    candidates, and each arm keeps its own event denominator.  `n_pairs` is the number of
    replicates both arms observe.  One generator, seeded with :data:`SEED` and labelled `best`,
    serves all rows in `_pairs_key` order; each draw picks the replicate indices once and re-uses
    them for both arms and every candidate.  A zero baseline leaves the relative columns NaN
    without dropping the draw.

    Args:
        hits: The scanned per-path sufficient statistics of every observer.
        candidates: A validated, non-empty candidate set.

    Returns:
        One row per contrast in the `m_model`, `rule` and `need` families, sorted by `_pairs_key`.
    """
    arms: dict[PairKey, dict[str, dict[int, tuple[tuple[float, int], tuple[float, int]]]]] = {}
    models: dict[tuple[int, str, str, str, str], dict[str, tuple[float, int]]] = {}
    rules: dict[tuple[int, str, str, str, str], dict[str, tuple[float, int]]] = {}
    contexts: dict[tuple[int, str, str, str], dict[tuple[str, str], tuple[float, int]]] = {}
    for (replicate, rho, rule, m_model, adopters, observer), stats in hits.items():
        if observer not in candidates:
            continue
        models.setdefault((replicate, rho, rule, adopters, observer), {})[m_model] = stats
        rules.setdefault((replicate, rho, m_model, adopters, observer), {})[rule] = stats
        contexts.setdefault((replicate, rho, adopters, observer), {})[(rule, m_model)] = stats
    for (replicate, rho, rule, adopters, observer), factors in models.items():
        for minuend, subtrahend in _M_MODEL_PAIRS:
            if minuend in factors and subtrahend in factors:
                arms.setdefault(("m_model", minuend, subtrahend, rho, rule, adopters, _BEST),
                                {}).setdefault(observer, {})[replicate] = (factors[minuend],
                                                                           factors[subtrahend])
    for (replicate, rho, m_model, adopters, observer), factors in rules.items():
        for minuend, subtrahend in _RULE_PAIRS:
            if minuend in factors and subtrahend in factors:
                arms.setdefault(("rule", minuend, subtrahend, rho, m_model, adopters, _BEST),
                                {}).setdefault(observer, {})[replicate] = (factors[minuend],
                                                                           factors[subtrahend])
    for (replicate, rho, adopters, observer), factors in contexts.items():
        baseline = factors.get(("need", "double"))
        if baseline is None:
            continue
        for rule, m_model in sorted(factors):
            if rule == "need":
                continue
            arms.setdefault(("need", rule, "need", rho, m_model, adopters, _BEST),
                            {}).setdefault(observer, {})[replicate] = (factors[rule, m_model], baseline)
    rng = random.Random(f"{SEED}|{_BEST}|paired")
    rows: list[Row] = []
    for key in sorted(arms, key=_pairs_key):
        per_candidate = arms[key]
        replicates = sorted(set.intersection(*(set(candidate) for candidate in per_candidate.values())))
        count = len(replicates)
        if not count:
            continue
        comparison = max(_arm_rate(candidate, replicates, 0) for candidate in per_candidate.values())
        baseline = max(_arm_rate(candidate, replicates, 1) for candidate in per_candidate.values())
        diffs: list[float] = []
        relatives: list[float] = []
        entropies: list[float] = []
        for _ in range(BOOTSTRAP):
            picked = [replicates[index] for index in rng.choices(range(count), k=count)]
            draw_comparison = max(_arm_rate(candidate, picked, 0)
                                  for candidate in per_candidate.values())
            draw_baseline = max(_arm_rate(candidate, picked, 1)
                                for candidate in per_candidate.values())
            diff = draw_comparison - draw_baseline
            diffs.append(diff)
            relatives.append(diff / draw_baseline if draw_baseline else math.nan)
            entropies.append(_best_entropy_change(draw_comparison, draw_baseline))
        diff = comparison - baseline
        diff_lo, diff_hi = percentiles(diffs)
        relative_lo, relative_hi = percentiles(relatives)
        entropy_lo, entropy_hi = percentiles(entropies)
        rows.append({
            "contrast": key[0], "minuend": key[1], "subtrahend": key[2], "rho": key[3],
            "fixed": key[4], "adopters": key[5], "observer": key[6], "n_pairs": count,
            "diff_hit": diff, "diff_lo": diff_lo, "diff_hi": diff_hi,
            "relative_hit": diff / baseline if baseline else math.nan,
            "relative_lo": relative_lo, "relative_hi": relative_hi,
            "relative_bits": _best_entropy_change(comparison, baseline),
            "relative_bits_lo": entropy_lo, "relative_bits_hi": entropy_hi,
        })
    return rows


# ==== cost ====


def _events(path: Path) -> list[tuple[int, int, int, int]]:
    """One `(source, kind, x, y)` per public event of a path, in `public_index` order.

    `source`, `kind`, `x` and `y` belong to the event, not to the observer, so every
    observer's row repeats them; a path whose observers disagree on an event raises `ValueError`.
    """
    events: dict[int, tuple[int, int, int, int]] = {}
    for row in read_rows(path):
        index = int(row["public_index"])
        event = (int(row["source"]), int(row["kind"]), int(row["x"]), int(row["y"]))
        if index in events and events[index] != event:
            raise ValueError(path, index)
        events[index] = event
    return list(events.values())


def _scan_cost(out: Path) -> tuple[dict[CostKey, list[float]], dict[CostKey, list[bool]], dict[CostKey, int],
                                   dict[CostKey, int]]:
    """One pass over every path of `out` for the cost table.

    Returns, per (rho, rule, m_model, adopters), each adopter event's `(y - x) / x`, whether it
    withdrew exactly the need, the cell's total adopter count `K` (zero at `rho = 0`), and the
    count of drain final withdrawals.  A drain adopter stays active down to a balance of one, so its
    last event, the last of its `source` in `public_index` order, empties the balance (`y = r`); it
    takes the whole remaining balance exactly when `y < threshold`, the run's tool-use threshold, and
    that take-all is no output of the rounding rule and is kept out of the cost statistics.
    """
    costs: dict[CostKey, list[float]] = {}
    same: dict[CostKey, list[bool]] = {}
    counts: dict[CostKey, int] = {}
    finals: dict[CostKey, int] = {}
    threshold: int | None = None
    for folder in _folders(out):
        content = json.loads((folder / "path.json").read_text(encoding="utf-8"))
        spec = content["spec"]
        key = (spec["rho"], spec["rule"], spec["m_model"], spec["adopters"])
        counts[key] = counts.get(key, 0) + content["K"]
        kind = _ADOPTER_KIND[spec["adopters"]]
        adopter = [(source, x, y) for source, event_kind, x, y in _events(folder / "rows.csv.gz")
                   if event_kind == kind]
        final: set[int] = set()
        if kind == DRAIN:
            if threshold is None:
                run = json.loads((out / "run.json").read_text(encoding="utf-8"))
                threshold = int(run["config"]["threshold"])
            last_of_source: dict[int, int] = {}
            for position, (source, _x, _y) in enumerate(adopter):
                last_of_source[source] = position
            final = {position for position in last_of_source.values() if adopter[position][2] < threshold}
        finals[key] = finals.get(key, 0) + len(final)
        for position, (_source, x, y) in enumerate(adopter):
            if position in final:
                continue
            costs.setdefault(key, []).append((y - x) / x)
            same.setdefault(key, []).append(y == x)
    return costs, same, counts, finals


def _cost_key(key: CostKey) -> tuple[Fraction, str, str, str]:
    """Sort key of a cost row: the ratio as a number, the rest as text."""
    return (Fraction(key[0]), key[1], key[2], key[3])


def _cost(costs: dict[CostKey, list[float]], same: dict[CostKey, list[bool]], counts: dict[CostKey, int],
          finals: dict[CostKey, int]) -> list[Row]:
    """One row per (rho, rule, m_model, adopters), sorted by its key columns.

    `n_events`, `same_share` and the `cost_*` columns leave out the drain take-alls (`n_final` of
    them), whose `x = y` is the withdrawal of a balance below the threshold, not a rounding output;
    `withdrawals_per_adopter` counts every adopter event, the take-alls included.  An empty cell,
    the `need` cell of `rho = 0`, has no adopter and leaves its statistics NaN.
    """
    rows: list[Row] = []
    for key in sorted(counts, key=_cost_key):
        values = costs.get(key, [])
        n = len(values)
        final = finals.get(key, 0)
        rows.append({
            "rho": key[0], "rule": key[1], "m_model": key[2], "adopters": key[3], "n_events": n,
            "n_final": final,
            "same_share": math.fsum(same[key]) / n if n else math.nan,
            "cost_mean": math.fsum(values) / n if n else math.nan,
            "cost_median": float(np.quantile(values, 0.5)) if n else math.nan,
            "cost_p90": float(np.quantile(values, 0.9)) if n else math.nan,
            "withdrawals_per_adopter": (n + final) / counts[key] if counts[key] else math.nan,
        })
    return rows


# ==== writing ====


def _cell(value: str | int | float) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _write(path: Path, columns: tuple[str, ...], rows: list[Row]) -> None:
    """Write `rows` as CSV with `columns` as the header; floats use `repr`."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row[c]) for c in columns])


def summarize(out: str, *, best_observers: list[str] | tuple[str, ...] | None = None
              ) -> dict[str, str | int]:
    """Write `groups.csv`, `best.csv`, `paired.csv` and `cost.csv` under `<out>/summary`.

    `groups` and `paired` draw their bootstrap from one generator in that order; `best` draws
    from its own, also seeded with :data:`SEED`, and the need contrast family from generators seeded
    with :data:`SEED` and its own labels.  The cost table counts adopter events, one observer's row
    per event: `x` is the need, a multiple of `quantum` in `[quantum, r]`, except on a drain
    adopter's take-all, the last withdrawal of a `source` whose `y` falls below the run's tool-use
    threshold: that event takes the whole remaining balance (`x = y = r`), is left out of `n_events`
    and the cost statistics, and is counted apart as `n_final`.  The threshold is read from
    `<out>/run.json`.  The tool-unusable fallback withdraws the need (`y = x`) and stays in the cost
    statistics; the scored rows carry neither the threshold nor the remaining balance that would
    tell it apart.

    The `group` observer is a kind-aware attack comparison: it is told the rule and the path-wide
    KEEP/DRAIN stopping policy, so it has model knowledge the release observers lack.  It is
    therefore reported in `groups` and `paired` but never enters the default `best`, whose envelope
    stays the strongest public observer's; an output holding only `group` rows writes `best.csv`
    with its header and no row.  Its tables draw from generators of their own, labelled with its
    name, so mixing it into an output leaves every public row, its bootstrap interval included,
    unchanged; in `groups` its rows take their sorted place, in `paired` they follow the public
    family blocks.  `cost` reads the events, not the observers, so the observer adds neither a row
    nor an event there.

    `best_observers` switches `best.csv` to an explicit combined envelope over the named
    observers: each candidate's repeated experiments are merged to one pooled hit rate per cell,
    the highest rate wins (ties to the first name), and the interval re-draws one shared replicate
    set and re-chooses the winner.  The names must be a non-empty list or tuple of distinct
    observers of the input, complete in every cell with the same replicate IDs and per-replicate
    event count; a violation raises `ValueError` before anything is written.  Such a run may name
    the kind-aware observer, and it appends `observer="best"` rows to `paired.csv` that compare
    each arm's highest pooled rate over the candidates, with one shared bootstrap stream.  The
    per-observer tables keep their rows and random streams in either mode; `None`, the default,
    keeps the public-only envelope and the previous output bytes.

    Args:
        out: An experiment output; its path directories are those holding `path.json`, and when a
            `run.json` is present they must be exactly the paths it records.
        best_observers: The candidates of the combined `best` table, or `None` for the default
            strongest-public-observer envelope.

    Returns:
        A mapping with `out`, the `<out>/summary` directory it wrote, and the row counts `groups`,
        `best` and `paired`.

    Raises:
        ValueError: If `<out>/run.json` records paths that the path directories do not match, or
            if `best_observers` is malformed or incomplete.
    """
    source = Path(out)
    folder = source / "summary"
    _check_paths(source)
    cells, maxes, hits = scan(source)
    candidates = None if best_observers is None else _select_best_observers(cells, best_observers)
    rng = random.Random(SEED)
    # The public tables are built first, in their original order and from the original generator,
    # so their rows and intervals do not depend on whether a kind-aware observer is scored at all.
    public_cells = {key: cell for key, cell in cells.items() if key[4] != KIND_AWARE}
    public_maxes = {key: value for key, value in maxes.items() if key[4] != KIND_AWARE}
    public_hits = {key: value for key, value in hits.items() if key[5] != KIND_AWARE}
    groups = _groups(public_cells, public_maxes, rng)
    best = (_best(groups, public_cells, random.Random(SEED)) if candidates is None
            else _candidate_best(cells, maxes, candidates, random.Random(SEED)))
    paired = _paired(public_hits, rng) + _need_rows(public_hits)
    aware_cells = {key: cell for key, cell in cells.items() if key[4] == KIND_AWARE}
    if aware_cells:
        aware_maxes = {key: value for key, value in maxes.items() if key[4] == KIND_AWARE}
        aware_hits = {key: value for key, value in hits.items() if key[5] == KIND_AWARE}
        groups = sorted([*groups,
                         *_groups(aware_cells, aware_maxes, random.Random(f"{SEED}|{KIND_AWARE}|groups"))],
                        key=_group_row_key)
        # `paired` keeps its family blocks: the public rows stay the table's prefix, the kind-aware
        # share follows them with the same block order.
        paired = [*paired,
                  *_paired(aware_hits, random.Random(f"{SEED}|{KIND_AWARE}|paired"), KIND_AWARE),
                  *_need_rows(aware_hits, KIND_AWARE)]
    if candidates is not None:
        # The combined share is a statistic over the candidates, appended after every observer
        # block; it owns its generator and leaves the per-observer rows untouched.
        paired = [*paired, *_best_paired(hits, candidates)]
    costs = _cost(*_scan_cost(source))
    folder.mkdir(parents=True, exist_ok=True)
    _write(folder / "groups.csv", GROUPS_COLUMNS, groups)
    _write(folder / "best.csv", GROUPS_COLUMNS, best)
    _write(folder / "paired.csv", PAIRED_COLUMNS, paired)
    _write(folder / "cost.csv", COST_COLUMNS, costs)
    return {"out": str(folder), "groups": len(groups), "best": len(best), "paired": len(paired)}
