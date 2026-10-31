"""Tests for `lab compare`, on synthetic experiment output directories with hand-written rows."""

from __future__ import annotations

import csv
import json
import math
from typing import TYPE_CHECKING

import pytest

from lab.report.compare import BEST, compare
from lab.score import write_rows
from lab.sim import KEEP

if TYPE_CHECKING:
    from pathlib import Path

HIST, LOGIT, GROUP = "historical", "trained-logit", "group"
# One path key per replicate of a small grid, all at rho 1/2, rule need, maximum model double, keep adopters.
PATH_KEYS = [f"r{replicate:03d}_rho1-2_need_double_keep" for replicate in range(2)]


def _row(observer: str, top1: float, index: int) -> dict:
    return {"path": "p", "observer": observer, "public_index": index, "source": 0, "kind": KEEP,
            "x": 1, "y": 1, "p_true": top1, "true_rank": 1.0, "top1": top1, "max_posterior": 0.5}


def _path(out: Path, replicate: int, rows: list[dict]) -> None:
    """Write one path folder's `path.json` and gzipped rows for the fixed grid key."""
    folder = out / PATH_KEYS[replicate]
    folder.mkdir(parents=True, exist_ok=True)
    spec = {"replicate": replicate, "rho": "1/2", "rule": "need", "m_model": "double", "adopters": "keep"}
    (folder / "path.json").write_text(json.dumps({"spec": spec}), encoding="utf-8")
    write_rows(folder / "rows.csv.gz", rows)


def _run(out: Path, config: dict) -> None:
    """Write a `run.json` pinning the fixed grid, `N`, snapshot and seed root of every test run."""
    out.mkdir(parents=True, exist_ok=True)
    report = {"config": config, "N": 8, "snapshot_sha256": "abc", "commit": "c", "paths": PATH_KEYS,
              "seconds": 0.0}
    (out / "run.json").write_text(json.dumps(report), encoding="utf-8")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_compare_pools_events_and_reports_the_change(tmp_path: Path) -> None:
    # GIVEN: A base run hitting 3/4 of its keep events and an alt run hitting 1/4, over two replicates.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0), _row(HIST, 1.0, 1)])
    _path(base, 1, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1)])
    _path(alt, 0, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1)])
    _path(alt, 1, [_row(HIST, 0.0, 0), _row(HIST, 0.0, 1)])
    _run(base, {"seed_root": "s", "replicates": 2, "theta": {"pi": "1/2"}})
    _run(alt, {"seed_root": "s", "replicates": 2, "theta": {"pi": "2/5"}})
    # WHEN: Comparing the two runs.
    result = compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    rows = {(row["observer"], row["group"]): row for row in _read(tmp_path / "dest" / "compare.csv")}
    row = rows[(HIST, "keep")]
    # THEN: Each side's bound is the log of its pooled hit rate, and the change is alt minus base.
    base_bits, alt_bits = -math.log2(0.75), 2.0
    assert result == {"out": str(tmp_path / "dest"), "rows": 2, "changed": ["theta.pi"]}
    assert (row["n_events_base"], row["n_events_alt"]) == ("4", "4")
    assert float(row["bits_base"]) == base_bits
    assert float(row["bits_alt"]) == alt_bits
    assert float(row["diff_bits"]) == alt_bits - base_bits
    assert math.isclose(float(row["relative_bits"]), (alt_bits - base_bits) / base_bits)


def test_compare_records_the_changed_config_keys(tmp_path: Path) -> None:
    # GIVEN: Two runs differing in a nested theta key and a top-level p.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0)])
    _path(base, 1, [_row(HIST, 1.0, 0)])
    _path(alt, 0, [_row(HIST, 1.0, 0)])
    _path(alt, 1, [_row(HIST, 1.0, 0)])
    _run(base, {"seed_root": "s", "replicates": 2, "p": 0, "theta": {"pi": "1/2", "w": 0}})
    _run(alt, {"seed_root": "s", "replicates": 2, "p": "1/4", "theta": {"pi": "1/2", "w": "1/2"}})
    # WHEN: Comparing.
    compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    report = json.loads((tmp_path / "dest" / "compare.json").read_text(encoding="utf-8"))
    # THEN: Only the differing leaves are listed, each with both values.
    assert report["changed"] == {"p": {"base": 0, "alt": "1/4"},
                                 "theta.w": {"base": 0, "alt": "1/2"}}


def test_compare_reports_a_key_the_base_config_omits(tmp_path: Path) -> None:
    # GIVEN: A base run whose config omits `p` (equal to its default, so `to_json` dropped it) and
    # an alt run that sets it.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0)])
    _path(base, 1, [_row(HIST, 1.0, 0)])
    _path(alt, 0, [_row(HIST, 1.0, 0)])
    _path(alt, 1, [_row(HIST, 1.0, 0)])
    _run(base, {"seed_root": "s", "replicates": 2, "theta": {"pi": "1/2"}})
    _run(alt, {"seed_root": "s", "replicates": 2, "p": "1/2", "theta": {"pi": "1/2"}})
    # WHEN: Comparing.
    compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    report = json.loads((tmp_path / "dest" / "compare.json").read_text(encoding="utf-8"))
    # THEN: `p` is listed with a null base, not dropped for being absent from the base's key set.
    # (Regression: a default value on the base side used to hide the key from the comparison.)
    assert report["changed"] == {"p": {"base": None, "alt": "1/2"}}


def test_compare_names_the_runs_by_directory_not_path(tmp_path: Path) -> None:
    # GIVEN: Two runs under the real layout `run/<run name>/grid`, so the full paths are absolute.
    base, alt = tmp_path / "run" / "usdt-p-1-2" / "grid", tmp_path / "run" / "usdt" / "grid"
    for out in (base, alt):
        _path(out, 0, [_row(HIST, 1.0, 0)])
        _path(out, 1, [_row(HIST, 1.0, 0)])
        _run(out, {"seed_root": "s", "replicates": 2})
    # WHEN: Comparing.
    compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    report = json.loads((tmp_path / "dest" / "compare.json").read_text(encoding="utf-8"))
    # THEN: Each run is named by its last two directory names, carrying no absolute path.
    assert (report["base"], report["alt"]) == ("usdt-p-1-2/grid", "usdt/grid")


def test_compare_of_a_run_with_itself_is_all_zero(tmp_path: Path) -> None:
    # GIVEN: One run at two replicates, with hit rates strictly inside (0, 1) so the bits stay finite.
    out = tmp_path / "out"
    _path(out, 0, [_row(HIST, 1.0, 0), _row(HIST, 1.0, 1)])
    _path(out, 1, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1)])
    _run(out, {"seed_root": "s", "replicates": 2, "theta": {"pi": "1/2"}})
    # WHEN: Comparing the run with itself.
    compare(str(out), str(out), dest=str(tmp_path / "dest"))
    # THEN: Every difference and its interval collapse to zero, bootstrap draws included.
    for row in _read(tmp_path / "dest" / "compare.csv"):
        assert float(row["diff_bits"]) == 0.0
        assert (float(row["diff_bits_lo"]), float(row["diff_bits_hi"])) == (0.0, 0.0)
        assert float(row["relative_bits"]) == 0.0
        assert (float(row["relative_bits_lo"]), float(row["relative_bits_hi"])) == (0.0, 0.0)


def test_compare_rejects_a_run_on_another_grid(tmp_path: Path) -> None:
    # GIVEN: Two runs whose path key sets differ.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0)])
    _path(alt, 0, [_row(HIST, 1.0, 0)])
    _run(base, {"seed_root": "s", "replicates": 2})
    _run(alt, {"seed_root": "s", "replicates": 2})
    (alt / "run.json").write_text(json.dumps({"config": {"seed_root": "s", "replicates": 2}, "N": 8,
                                              "snapshot_sha256": "abc", "commit": "c", "paths": ["other"],
                                              "seconds": 0.0}), encoding="utf-8")
    # WHEN: Comparing runs that do not share a grid.
    # THEN: The comparison is refused rather than pairing cells across different paths.
    with pytest.raises(ValueError, match="paths mismatch"):
        compare(str(base), str(alt), dest=str(tmp_path / "dest"))


def test_compare_rejects_another_seed_root(tmp_path: Path) -> None:
    # GIVEN: Two runs on the same grid but with different seed roots.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0)])
    _path(alt, 0, [_row(HIST, 1.0, 0)])
    _run(base, {"seed_root": "s", "replicates": 2})
    _run(alt, {"seed_root": "other", "replicates": 2})
    # WHEN: Comparing runs whose adoption draws no longer align.
    # THEN: The comparison is refused, since the replicates would not pair.
    with pytest.raises(ValueError, match="seed_root mismatch"):
        compare(str(base), str(alt), dest=str(tmp_path / "dest"))


def test_compare_best_row_takes_the_strongest_observer_per_side(tmp_path: Path) -> None:
    # GIVEN: A base run where historical beats trained-logit and an alt run where the order flips.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0), _row(HIST, 1.0, 1), _row(LOGIT, 0.0, 0), _row(LOGIT, 0.0, 1)])
    _path(base, 1, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1), _row(LOGIT, 1.0, 0), _row(LOGIT, 0.0, 1)])
    _path(alt, 0, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1), _row(LOGIT, 1.0, 0), _row(LOGIT, 1.0, 1)])
    _path(alt, 1, [_row(HIST, 0.0, 0), _row(HIST, 0.0, 1), _row(LOGIT, 0.0, 0), _row(LOGIT, 0.0, 1)])
    _run(base, {"seed_root": "s", "replicates": 2})
    _run(alt, {"seed_root": "s", "replicates": 2})
    # WHEN: Comparing.
    compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    rows = {(row["observer"], row["group"]): row for row in _read(tmp_path / "dest" / "compare.csv")}
    # THEN: The best row carries each side's stronger observer: 3/4 on the base, 1/2 on the alt.
    best = rows[("best", "keep")]
    assert float(best["bits_base"]) == -math.log2(0.75)
    assert float(best["bits_alt"]) == -math.log2(0.5)
    assert float(rows[(HIST, "keep")]["bits_base"]) == -math.log2(0.75)
    assert float(rows[(LOGIT, "keep")]["bits_base"]) == 2.0


def test_compare_is_byte_identical_across_runs(tmp_path: Path) -> None:
    # GIVEN: A comparison written once.
    base, alt = tmp_path / "base", tmp_path / "alt"
    _path(base, 0, [_row(HIST, 1.0, 0), _row(HIST, 1.0, 1)])
    _path(base, 1, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1)])
    _path(alt, 0, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1)])
    _path(alt, 1, [_row(HIST, 0.0, 0), _row(HIST, 0.0, 1)])
    _run(base, {"seed_root": "s", "replicates": 2})
    _run(alt, {"seed_root": "s", "replicates": 2})
    compare(str(base), str(alt), dest=str(tmp_path / "first"))
    first = (tmp_path / "first" / "compare.csv").read_bytes()
    # WHEN: Comparing the same pair again.
    compare(str(base), str(alt), dest=str(tmp_path / "second"))
    # THEN: The table is unchanged, bootstrap resamples included.
    assert first == (tmp_path / "second" / "compare.csv").read_bytes()


def test_compare_best_rows_leave_out_the_kind_aware_observer(tmp_path: Path) -> None:
    # GIVEN: Two runs on one grid whose every path is scored by historical and by the kind-aware
    # observer, which hits every event and so is the strongest of the two.
    base, alt = tmp_path / "base", tmp_path / "alt"
    for side in (base, alt):
        for replicate in range(2):
            _path(side, replicate, [_row(HIST, 1.0, 0), _row(HIST, 0.0, 1),
                                    _row(GROUP, 1.0, 0), _row(GROUP, 1.0, 1)])
        _run(side, {"seed_root": "s", "replicates": 2})
    # WHEN: Comparing the two runs.
    compare(str(base), str(alt), dest=str(tmp_path / "dest"))
    rows = {(row["observer"], row["group"]): row for row in _read(tmp_path / "dest" / "compare.csv")}
    # THEN: The kind-aware observer keeps its own row, but the strongest-observer family stays the
    # strongest public one, as `best.csv` does.
    assert float(rows[(GROUP, "keep")]["bits_base"]) == 0.0
    assert float(rows[(BEST, "keep")]["bits_base"]) == float(rows[(HIST, "keep")]["bits_base"]) == 1.0
    assert rows[(BEST, "keep")]["n_events_base"] == rows[(HIST, "keep")]["n_events_base"]
