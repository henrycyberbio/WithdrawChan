"""Tests for the kind-aware `group` observer's place in the summary tables.

The fixture is synthetic: `group` is told the rule and the path-wide stopping policy, so it must be
reported but never enter the public strongest-observer envelope, and it must not move any public
interval.
"""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

from lab.report.summary import GROUPS_COLUMNS, summarize
from lab.score import write_rows
from lab.sim import KEEP

if TYPE_CHECKING:
    from pathlib import Path

HIST, LOGIT, COMB = "historical", "trained-logit", "combinatorial"
PUBLIC = (HIST, LOGIT, COMB)
GROUP = "group"

#: `top1` per observer, replicate then event; the kind-aware observer hits every event, so it wins
#: every cell if it were allowed into the comparison.
_TOP1 = {
    HIST: ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    LOGIT: ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    COMB: ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    GROUP: ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
}

#: Columns identifying one row of each table.
_GROUPS_KEY = GROUPS_COLUMNS[:6]
_PAIR_KEY = ("contrast", "minuend", "subtrahend", "rho", "fixed", "adopters", "observer")

#: The path's public events, `(kind, x, y)` in `public_index` order; every observer repeats them.
_EVENTS = ((KEEP, 100, 110), (KEEP, 200, 190), (KEEP, 50, 60))


def _row(observer: str, kind: int, top1: float, index: int, x: int, y: int) -> dict:
    return {"path": "p", "observer": observer, "public_index": index, "source": 0, "kind": kind,
            "x": x, "y": y, "p_true": 0.5, "true_rank": 1.0, "top1": top1, "max_posterior": 0.5}


def _path(out: Path, replicate: int, rule: str, m_model: str, observers: tuple[str, ...]) -> None:
    """Write one path whose `rho` is 1/2 and whose adopters are keepers, scored by `observers`."""
    folder = out / f"r{replicate:03d}_rho1-2_{rule}_{m_model}_keep"
    folder.mkdir(parents=True)
    spec = {"replicate": replicate, "rho": "1/2", "rule": rule, "m_model": m_model, "adopters": "keep"}
    (folder / "path.json").write_text(json.dumps({"spec": spec, "K": 1}), encoding="utf-8")
    rows = [_row(observer, kind, _TOP1[observer][replicate][index], index, x, y)
            for observer in observers for index, (kind, x, y) in enumerate(_EVENTS)]
    write_rows(folder / "rows.csv.gz", rows)


def _grid(out: Path, observers: tuple[str, ...]) -> None:
    """Two replicates of both rules at both maximum models, scored by `observers`."""
    for replicate in (0, 1):
        for rule in ("decstep", "popstep"):
            for m_model in ("double", "balance"):
                _path(out, replicate, rule, m_model, observers)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _keyed(rows: list[dict[str, str]], columns: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[column] for column in columns): row for row in rows}


def _rows_of(rows: list[dict[str, str]], observer: str) -> list[dict[str, str]]:
    return [row for row in rows if row["observer"] == observer]


def test_group_appears_in_the_group_and_paired_tables(tmp_path: Path) -> None:
    # GIVEN: A mixed grid with the three public observers and the kind-aware one.
    out = tmp_path / "out"
    _grid(out, (*PUBLIC, GROUP))
    # WHEN: Summarizing.
    summarize(str(out))
    groups, paired = _read(out / "summary" / "groups.csv"), _read(out / "summary" / "paired.csv")
    # THEN: Both tables report the kind-aware observer, with both contrast families.
    assert {row["observer"] for row in groups} == set(PUBLIC) | {GROUP}
    assert len(_rows_of(groups, GROUP)) == 4
    assert {(row["contrast"], row["observer"]) for row in paired} == {
        (contrast, observer) for contrast in ("m_model", "rule") for observer in (*PUBLIC, GROUP)}


def test_adding_group_leaves_the_public_rows_and_the_other_tables_unchanged(tmp_path: Path) -> None:
    # GIVEN: The same grid summarized without the kind-aware observer and with it.
    base, mixed = tmp_path / "base", tmp_path / "mixed"
    _grid(base, PUBLIC)
    _grid(mixed, (*PUBLIC, GROUP))
    # WHEN: Summarizing both.
    summarize(str(base))
    summarize(str(mixed))
    # THEN: Every public row keeps all its numbers, its bootstrap interval included, and the
    # group-free files are byte-identical; cost counts events, not observers.
    for name, columns in (("groups.csv", _GROUPS_KEY), ("paired.csv", _PAIR_KEY)):
        position = columns.index("observer")
        before = _keyed(_read(base / "summary" / name), columns)
        after = _keyed(_read(mixed / "summary" / name), columns)
        assert after.keys() >= before.keys()
        assert {key: value for key, value in after.items() if key[position] != GROUP} == before
        assert _rows_of(_read(mixed / "summary" / name), GROUP)
        # `groups` is one sorted table, so the public rows keep their order; `paired` keeps its
        # family blocks, so the public rows stay the table's prefix.
        base_rows = _read(base / "summary" / name)
        public = [row for row in _read(mixed / "summary" / name) if row["observer"] != GROUP]
        if name == "groups.csv":
            assert public == base_rows
        else:
            assert _read(mixed / "summary" / name)[:len(base_rows)] == base_rows
    for name in ("best.csv", "cost.csv"):
        assert (mixed / "summary" / name).read_bytes() == (base / "summary" / name).read_bytes()


def test_group_only_output_writes_an_empty_best_table(tmp_path: Path) -> None:
    # GIVEN: An output scored by the kind-aware observer alone.
    out = tmp_path / "out"
    _grid(out, (GROUP,))
    # WHEN: Summarizing.
    result = summarize(str(out))
    # THEN: `groups` and `paired` report it, `best` keeps its header and no row, and cost still
    # measures the events.
    assert result["best"] == 0
    assert _read(out / "summary" / "best.csv") == []
    assert (out / "summary" / "best.csv").read_text(encoding="utf-8").splitlines() == [",".join(GROUPS_COLUMNS)]
    assert {row["observer"] for row in _read(out / "summary" / "groups.csv")} == {GROUP}
    assert {row["observer"] for row in _read(out / "summary" / "paired.csv")} == {GROUP}
    assert len(_read(out / "summary" / "cost.csv")) == 4


def test_group_rows_depend_only_on_the_group_rows(tmp_path: Path) -> None:
    # GIVEN: The kind-aware observer's rows in a mixed output and in an output of their own.
    mixed, only = tmp_path / "mixed", tmp_path / "only"
    _grid(mixed, (*PUBLIC, GROUP))
    _grid(only, (GROUP,))
    # WHEN: Summarizing both.
    summarize(str(mixed))
    summarize(str(only))
    # THEN: Its rows and intervals are the same in both: they draw from their own streams, not from
    # a position in the public ones.
    for name, columns in (("groups.csv", _GROUPS_KEY), ("paired.csv", _PAIR_KEY)):
        assert _keyed(_rows_of(_read(mixed / "summary" / name), GROUP), columns) == \
            _keyed(_rows_of(_read(only / "summary" / name), GROUP), columns)
