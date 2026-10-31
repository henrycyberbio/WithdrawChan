"""Tests for the explicit-observer `best` statistic of :func:`lab.report.summary.summarize`.

The four-observer envelope merges each candidate's repeated experiments first and then keeps the
highest pooled hit rate, so these tests pin the merge-before-choose order, the shared bootstrap
indices with a re-chosen winner, the two-arm paired statistic with unequal denominators, and the
explicit rejection of incomplete candidate sets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from typing import TYPE_CHECKING

import pytest

from lab.report.summary import summarize
from lab.score import write_rows
from lab.sim import KEEP, NON

if TYPE_CHECKING:
    from pathlib import Path

HIST, LOGIT, COMB, GROUP = "historical", "trained-logit", "combinatorial", "group"
FOUR = (HIST, LOGIT, COMB, GROUP)


def _row(observer: str, kind: int, top1: float, max_posterior: float, index: int, *,
         x: int = 100, y: int = 100, source: int = 0) -> dict:
    return {"path": "p", "observer": observer, "public_index": index, "source": source, "kind": kind,
            "x": x, "y": y, "p_true": max_posterior, "true_rank": 1.0, "top1": top1,
            "max_posterior": max_posterior}


def _path(out: Path, replicate: int, rule: str, m_model: str, adopters: str, rows: list[dict],
          k: int = 0) -> None:
    """Write one path folder's `path.json` and gzipped rows under `out`; rho is always 1/2."""
    key = f"r{replicate:03d}_rho1-2_{rule}_{m_model}_{adopters}"
    folder = out / key
    folder.mkdir(parents=True)
    spec = {"replicate": replicate, "rho": "1/2", "rule": rule, "m_model": m_model, "adopters": adopters}
    (folder / "path.json").write_text(json.dumps({"spec": spec, "K": k}), encoding="utf-8")
    write_rows(folder / "rows.csv.gz", rows)


def _rows(hits: dict[str, list[float]], kind: int = KEEP) -> list[dict]:
    """One row per (observer, value); observers with equal list lengths share a denominator."""
    return [_row(observer, kind, value, 0.5, index)
            for observer, values in hits.items() for index, value in enumerate(values)]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _four_out(tmp_path: Path) -> Path:
    """Two replicates of one keep cell with all four candidates and equal denominators."""
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", _rows({name: [0.5, 0.5] for name in FOUR}), k=1)
    _path(out, 1, "need", "double", "keep", _rows({name: [0.5, 1.0] for name in FOUR}), k=1)
    return out


def test_best_merges_repeated_experiments_before_choosing(tmp_path: Path) -> None:
    # GIVEN: Unequal per-replicate denominators, and candidates whose per-replicate best differs
    # from their pooled hit rate: historical wins replicate 0 alone, trained-logit replicate 1.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep",
          _rows({HIST: [1.0], LOGIT: [0.0], COMB: [0.0], GROUP: [0.0]}), k=1)
    _path(out, 1, "need", "double", "keep",
          _rows({HIST: [0.0, 0.0, 0.0], LOGIT: [1.0, 1.0, 0.0],
                 COMB: [0.0, 0.0, 0.0], GROUP: [0.0, 0.0, 0.0]}), k=1)
    # WHEN: Summarizing with the four explicit candidates.
    summarize(str(out), best_observers=FOUR)
    best = _read(out / "summary" / "best.csv")
    # THEN: The merged rates are 1/4 and 1/2, so trained-logit wins with 1/2; choosing per
    # replicate (1.0) or averaging per-replicate rates (a tie with historical) would differ.
    assert len(best) == 1
    row = best[0]
    assert (row["observer"], row["n_events"], float(row["hit_rate"])) == (LOGIT, "4", 0.5)
    assert float(row["bits"]) == 1.0


def test_best_ignores_a_stronger_candidate_outside_the_explicit_set(tmp_path: Path) -> None:
    # GIVEN: A perfect fifth observer that the explicit set does not name.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep",
          _rows({HIST: [1.0], LOGIT: [0.0], COMB: [0.0], GROUP: [0.0], "extra": [1.0]}), k=1)
    _path(out, 1, "need", "double", "keep",
          _rows({HIST: [0.0], LOGIT: [1.0], COMB: [0.0], GROUP: [0.0], "extra": [1.0]}), k=1)
    # WHEN: Summarizing with the four named candidates only.
    summarize(str(out), best_observers=FOUR)
    # THEN: Both historical and trained-logit pool 1/2; the tie goes to the first name, and the
    # unselected observer's perfect rate never enters `best`.
    row = _read(out / "summary" / "best.csv")[0]
    assert (row["observer"], float(row["hit_rate"])) == (HIST, 0.5)


def test_best_interval_resamples_shared_indices_and_rechooses_the_winner(tmp_path: Path) -> None:
    # GIVEN: Two public candidates that pool 1/4 each but each hit only in its own replicate, plus
    # two zero candidates.  The denominator is 1 in replicate 0 and 3 in replicate 1.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep",
          _rows({HIST: [1.0], LOGIT: [0.0], COMB: [0.0], GROUP: [0.0]}), k=1)
    _path(out, 1, "need", "double", "keep",
          _rows({HIST: [0.0, 0.0, 0.0], LOGIT: [1.0, 0.0, 0.0],
                 COMB: [0.0, 0.0, 0.0], GROUP: [0.0, 0.0, 0.0]}), k=1)
    # WHEN: Summarizing.
    summarize(out=str(out), best_observers=FOUR)
    row = _read(out / "summary" / "best.csv")[0]
    # THEN: The point tie goes to historical at 1/4, but every shared resample lets the other
    # candidate win when its replicate is drawn, so the interval is [1/4, 1] rather than the
    # single-winner interval [0, 1] a fixed winner would give.
    assert (row["observer"], float(row["hit_rate"])) == (HIST, 0.25)
    assert (float(row["hit_lo"]), float(row["hit_hi"])) == (0.25, 1.0)


def test_best_pools_each_arm_and_keeps_unequal_arm_denominators(tmp_path: Path) -> None:
    # GIVEN: balance arms with two events per replicate and double arms with one.
    out = tmp_path / "out"
    for replicate, balance, double in (
            (0, {HIST: [1.0, 1.0], LOGIT: [0.0, 0.0], COMB: [0.0, 0.0], GROUP: [0.0, 0.0]},
             {HIST: [0.0], LOGIT: [1.0], COMB: [0.0], GROUP: [0.0]}),
            (1, {HIST: [1.0, 0.0], LOGIT: [1.0, 1.0], COMB: [0.0, 0.0], GROUP: [0.0, 0.0]},
             {HIST: [0.0], LOGIT: [0.0], COMB: [0.0], GROUP: [0.0]})):
        _path(out, replicate, "decstep", "balance", "keep", _rows(balance), k=1)
        _path(out, replicate, "decstep", "double", "keep", _rows(double), k=1)
    # WHEN: Summarizing the paired best statistic.
    summarize(str(out), best_observers=FOUR)
    best = [row for row in _read(out / "summary" / "paired.csv") if row["observer"] == "best"]
    # THEN: One row per family member.  A is historical's pooled 3/4, B is trained-logit's pooled
    # 1/2, each arm chosen on its own; the per-replicate difference means would give 1/2 instead.
    assert len(best) == 1
    row = best[0]
    assert (row["contrast"], row["minuend"], row["subtrahend"], row["fixed"], row["adopters"]) == (
        "m_model", "balance", "double", "decstep", "keep")
    assert row["n_pairs"] == "2"
    assert float(row["diff_hit"]) == 0.25
    assert float(row["relative_hit"]) == 0.5
    assert math.isclose(float(row["relative_bits"]), (-math.log2(0.75) - 1.0) / 1.0)


def test_paired_best_rechooses_the_winning_arm_inside_each_draw(tmp_path: Path) -> None:
    # GIVEN: The two candidates trade the balance arm per replicate; every double arm hits nothing.
    out = tmp_path / "out"
    for replicate, balance in ((0, {HIST: [1.0], LOGIT: [0.0]}),
                               (1, {HIST: [0.0], LOGIT: [1.0]})):
        _path(out, replicate, "decstep", "balance", "keep", _rows(balance), k=1)
        _path(out, replicate, "decstep", "double", "keep",
              _rows({HIST: [0.0], LOGIT: [0.0]}), k=1)
    # WHEN: Summarizing.
    summarize(str(out), best_observers=(HIST, LOGIT))
    row = next(row for row in _read(out / "summary" / "paired.csv") if row["observer"] == "best")
    # THEN: A zero baseline makes both relative columns NaN, but the difference keeps the maximum
    # of the two arms in every draw, never a fixed winner's 0 lower bound.
    assert float(row["diff_hit"]) == 0.5
    assert (float(row["diff_lo"]), float(row["diff_hi"])) == (0.5, 1.0)
    assert math.isnan(float(row["relative_hit"]))
    assert math.isnan(float(row["relative_bits"]))


def test_paired_best_relative_bits_is_nan_when_the_baseline_hits_everything(tmp_path: Path) -> None:
    # GIVEN: A balance arm at 1/2 and a double arm at 1, whose entropy bound is zero bits.
    out = tmp_path / "out"
    for replicate in (0, 1):
        _path(out, replicate, "decstep", "balance", "keep",
              _rows({HIST: [1.0, 0.0]}), k=1)
        _path(out, replicate, "decstep", "double", "keep", _rows({HIST: [1.0]}), k=1)
    # WHEN: Summarizing.
    summarize(str(out), best_observers=(HIST,))
    row = next(row for row in _read(out / "summary" / "paired.csv") if row["observer"] == "best")
    # THEN: 0-hit ratios stay undefined (NaN, not dropped), while the hit difference is exact.
    assert float(row["relative_hit"]) == -0.5
    assert math.isnan(float(row["relative_bits"]))
    assert math.isnan(float(row["relative_bits_lo"]))


def test_explicit_best_considers_group_while_the_default_still_excludes_it(tmp_path: Path) -> None:
    # GIVEN: The kind-aware observer hits everything and the public ones at most half the time.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep",
          _rows({HIST: [1.0, 0.0], LOGIT: [0.0, 1.0], COMB: [0.0, 0.0],
                 GROUP: [1.0, 1.0]}), k=1)
    # WHEN: Summarizing once with the default candidates and once with the four.
    summarize(str(out))
    default = _read(out / "summary" / "best.csv")[0]
    summarize(str(out), best_observers=FOUR)
    explicit = _read(out / "summary" / "best.csv")[0]
    # THEN: The default envelope stays the strongest public observer, the explicit one may name the
    # kind-aware candidate it was asked for.
    assert (default["observer"], float(default["hit_rate"])) == (HIST, 0.5)
    assert (explicit["observer"], float(explicit["hit_rate"])) == (GROUP, 1.0)


@pytest.mark.parametrize("bad", [[], (), ["historical", "historical"], ["historical", "nope"],
                                 "historical", ["historical", 3]])
def test_rejects_bad_candidate_sets_before_writing(tmp_path: Path, bad: object) -> None:
    # GIVEN: A complete four-observer output.
    out = _four_out(tmp_path)
    # WHEN: The candidate argument is empty, repeated, unknown or not a list of names.
    # THEN: `summarize` refuses and leaves no summary directory behind.
    with pytest.raises(ValueError, match="best_observers"):
        summarize(str(out), best_observers=bad)  # type: ignore[arg-type]
    assert not (out / "summary").exists()


def test_rejects_a_candidate_missing_from_a_cell(tmp_path: Path) -> None:
    # GIVEN: `group` is absent from one replicate, so the cell is incomplete.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", _rows({name: [0.5] for name in FOUR}), k=1)
    _path(out, 1, "need", "double", "keep", _rows({name: [0.5] for name in (HIST, LOGIT, COMB)}), k=1)
    with pytest.raises(ValueError, match="group"):
        summarize(str(out), best_observers=FOUR)
    assert not (out / "summary").exists()


def test_rejects_a_cell_holding_none_of_the_candidates(tmp_path: Path) -> None:
    # GIVEN: One complete four-observer cell and another cell (a different rule) that holds only
    # an unselected observer, so every candidate is known from the input but this cell has none.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", _rows({name: [0.5, 0.5] for name in FOUR}), k=1)
    _path(out, 1, "decstep", "double", "keep", _rows({"extra": [0.5]}), k=1)
    # WHEN: Summarizing with the four named candidates.
    # THEN: The empty cell is an incomplete cell, not a silently dropped one; nothing is written.
    with pytest.raises(ValueError, match="missing from cell"):
        summarize(str(out), best_observers=FOUR)
    assert not (out / "summary").exists()


def test_rejects_mismatched_replicates_and_denominators(tmp_path: Path) -> None:
    # GIVEN: `group` has a different replicate ID set, then a different per-replicate denominator.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", _rows({name: [0.5] for name in FOUR}), k=1)
    _path(out, 1, "need", "double", "keep", _rows({name: [0.5] for name in (HIST, LOGIT, COMB)})
          + _rows({GROUP: [0.5]}), k=1)
    _path(out, 2, "need", "double", "keep", _rows({GROUP: [0.5]}), k=1)
    with pytest.raises(ValueError, match="differs"):
        summarize(str(out), best_observers=FOUR)
    assert not (out / "summary").exists()

    uneven = tmp_path / "uneven"
    _path(uneven, 0, "need", "double", "keep", _rows({name: [0.5] for name in (HIST, LOGIT, COMB)})
          + _rows({GROUP: [0.5, 0.5]}), k=1)
    _path(uneven, 1, "need", "double", "keep", _rows({name: [0.5] for name in FOUR}), k=1)
    with pytest.raises(ValueError, match="differs"):
        summarize(str(uneven), best_observers=FOUR)
    assert not (uneven / "summary").exists()


def _compat_case(out: Path) -> None:
    """Three rules at two maximum models over two replicates, all four observers, adopter costs."""
    for replicate in (0, 1):
        for rule in ("need", "decstep", "popstep"):
            for m_model in ("double", "balance"):
                if rule == "need" and m_model == "balance":
                    continue
                folded = [0.25, 0.75] if replicate else [1.0, 0.0]
                hits = {
                    HIST: list(folded),
                    LOGIT: list(reversed(folded)),
                    COMB: [0.0, 0.5],
                    GROUP: [1.0 if replicate else 0.0, 0.0],
                }
                rows = []
                for observer, values in hits.items():
                    rows += [_row(observer, KEEP, value, 0.5, index, x=100, y=110)
                             for index, value in enumerate(values)]
                    rows += [_row(observer, NON, 0.0, 0.25, 2, x=100, y=100)]
                _path(out, replicate, rule, m_model, "keep", rows, k=2)


#: sha256 of groups, best, paired and cost, concatenated in name order, captured from the frozen
#: release before `best_observers` existed: the default call must keep these bytes.
_DEFAULT_TABLES_SHA256 = "cf1c82dde0fad2b33fe6f7f9305a70d505d0418a24f2d6e004a046499e2bd84f"


def _digest(folder: Path) -> str:
    return hashlib.sha256(b"".join((folder / name).read_bytes()
                                   for name in ("groups.csv", "best.csv", "paired.csv", "cost.csv"))).hexdigest()


def test_default_output_keeps_its_bytes_and_explicit_mode_only_appends(tmp_path: Path) -> None:
    # GIVEN: A complete output summarized with the historical default.
    out = tmp_path / "out"
    _compat_case(out)
    summarize(str(out))
    folder = out / "summary"
    default = {name: (folder / name).read_bytes() for name in ("groups.csv", "best.csv", "paired.csv",
                                                              "cost.csv")}
    default_paired = _read(folder / "paired.csv")
    # WHEN: Summarizing again with the keyword omitted and set to None, then with the four names.
    summarize(str(out), best_observers=None)
    assert {name: (folder / name).read_bytes() for name in default} == default
    assert _digest(folder) == _DEFAULT_TABLES_SHA256
    summarize(str(out), best_observers=FOUR)
    # THEN: Only `best.csv` and an appended `best` share of `paired.csv` change: the per-observer
    # group, pair and cost tables keep their bytes and random streams.
    assert (folder / "groups.csv").read_bytes() == default["groups.csv"]
    assert (folder / "cost.csv").read_bytes() == default["cost.csv"]
    explicit_paired = _read(folder / "paired.csv")
    assert explicit_paired[:len(default_paired)] == default_paired
    best_block = explicit_paired[len(default_paired):]
    assert {row["observer"] for row in best_block} == {"best"}
    assert {row["contrast"] for row in best_block} == {"m_model", "need", "rule"}
    assert [row["contrast"] for row in best_block] == sorted(row["contrast"] for row in best_block)
    explicit_best = _read(folder / "best.csv")
    cells = {(row["rho"], row["rule"], row["m_model"], row["adopters"], row["group"])
             for row in explicit_best}
    assert len(explicit_best) == len(cells)
    assert len(cells) > 0
    assert {row["observer"] for row in explicit_best} <= set(FOUR)
