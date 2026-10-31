"""Tests for the summary tables, on synthetic experiment output directories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from typing import TYPE_CHECKING

import pytest

from lab.report.summary import summarize
from lab.score import write_rows
from lab.sim import DRAIN, KEEP, NON

if TYPE_CHECKING:
    from pathlib import Path

HIST, LOGIT = "historical", "trained-logit"


def _row(observer: str, kind: int, top1: float, max_posterior: float, index: int, *,
         x: int = 1, y: int = 1, source: int = 0) -> dict:
    return {"path": "p", "observer": observer, "public_index": index, "source": source, "kind": kind,
            "x": x, "y": y, "p_true": max_posterior, "true_rank": 1.0, "top1": top1,
            "max_posterior": max_posterior}


def _key(replicate: int, rule: str = "need", m_model: str = "double", adopters: str = "keep") -> str:
    """The directory name of a path folder at rho 1/2."""
    return f"r{replicate:03d}_rho1-2_{rule}_{m_model}_{adopters}"


def _path(out: Path, replicate: int, rule: str, m_model: str, adopters: str, rows: list[dict],
          k: int = 0) -> None:
    """Write one path folder's `path.json` and gzipped rows under `out`; rho is always 1/2."""
    key = _key(replicate, rule, m_model, adopters)
    folder = out / key
    folder.mkdir(parents=True)
    spec = {"replicate": replicate, "rho": "1/2", "rule": rule, "m_model": m_model, "adopters": adopters}
    (folder / "path.json").write_text(json.dumps({"spec": spec, "K": k}), encoding="utf-8")
    write_rows(folder / "rows.csv.gz", rows)


def _manifest(out: Path, paths: list[str], **config: object) -> None:
    """Write a `run.json` recording the path keys and config of an output."""
    (out / "run.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({"config": config, "paths": paths}), encoding="utf-8")


def _link(target: Path, link: Path) -> None:
    """Make `link` a directory link to `target`.

    A host that refuses symlinks without a privilege gets a junction instead, which `iterdir` and
    `is_file` treat the same way.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],  # noqa: S603,S607
                       check=True, capture_output=True)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_groups_pool_the_replicates_and_bound_the_hit_rate(tmp_path: Path) -> None:
    # GIVEN: Two replicates of one cell: keep events with 3 hits of 4, and non events with no hits.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 0.0, 0.25, 1),
                                             _row(HIST, NON, 0.0, 0.25, 2)])
    _path(out, 1, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 1.0, 0.5, 1),
                                             _row(HIST, NON, 0.0, 0.25, 2)])
    # WHEN: Summarizing.
    result = summarize(str(out))
    rows = {(row["observer"], row["group"]): row for row in _read(out / "summary" / "groups.csv")}
    keep, non = rows[("historical", "keep")], rows[("historical", "non")]
    # THEN: The keep cell pools 3 hits over 4 events, and the zero-hit non cell has an infinite bound.
    assert result == {"out": str(out / "summary"), "groups": 2, "best": 2, "paired": 0}
    assert (keep["n_events"], float(keep["hit_rate"])) == ("4", 0.75)
    assert float(keep["bits"]) == -math.log2(0.75)
    assert (float(keep["hit_lo"]), float(keep["hit_hi"])) == (0.5, 1.0)
    assert (float(keep["bits_lo"]), float(keep["bits_hi"])) == (0.0, 1.0)
    assert (float(keep["point_bits_mean"]), float(keep["point_bits_q10"])) == (1.25, 1.0)
    assert non["n_events"] == "2"
    assert float(non["hit_rate"]) == 0.0
    assert float(non["bits"]) == math.inf


def test_best_picks_the_observer_with_the_higher_hit_rate(tmp_path: Path) -> None:
    # GIVEN: One cell observed by two observers with hit rates 3/4 and 1/4.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 0.0, 0.5, 1),
                                             _row(LOGIT, KEEP, 0.0, 0.5, 0), _row(LOGIT, KEEP, 0.0, 0.5, 1)])
    _path(out, 1, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 1.0, 0.5, 1),
                                             _row(LOGIT, KEEP, 1.0, 0.5, 0), _row(LOGIT, KEEP, 0.0, 0.5, 1)])
    # WHEN: Summarizing.
    summarize(str(out))
    rows = _read(out / "summary" / "best.csv")
    # THEN: The single keep row names the stronger observer.
    assert [row["observer"] for row in rows] == ["historical"]
    assert float(rows[0]["hit_rate"]) == 0.75


def test_best_interval_covers_the_choice_of_observer(tmp_path: Path) -> None:
    # GIVEN: Two observers with the same pooled hit rate 1/2 that each hit in a different replicate,
    # so every resample in which one observer drops to 0 lifts the other to 1.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(LOGIT, KEEP, 0.0, 0.5, 0)])
    _path(out, 1, "need", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0), _row(LOGIT, KEEP, 1.0, 0.5, 0)])
    # WHEN: Summarizing.
    summarize(str(out))
    single = {row["observer"]: row for row in _read(out / "summary" / "groups.csv")}
    best = _read(out / "summary" / "best.csv")
    # THEN: Each observer alone spans [0, 1], but the strongest of the two never falls below 1/2;
    # the tie goes to the first name.
    assert (float(single[HIST]["hit_lo"]), float(single[HIST]["hit_hi"])) == (0.0, 1.0)
    assert [row["observer"] for row in best] == [HIST]
    assert float(best[0]["hit_rate"]) == 0.5
    assert (float(best[0]["hit_lo"]), float(best[0]["hit_hi"])) == (0.5, 1.0)
    assert (float(best[0]["bits_lo"]), float(best[0]["bits_hi"])) == (0.0, 1.0)


def test_paired_differences_skip_unpaired_paths(tmp_path: Path) -> None:
    # GIVEN: Both maximum models at two replicates, plus a balance path whose replicate has no double.
    out = tmp_path / "out"
    _path(out, 0, "decstep", "balance", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    _path(out, 0, "decstep", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0)])
    _path(out, 1, "decstep", "balance", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0),
                                                  _row(HIST, KEEP, 0.0, 0.5, 1)])
    _path(out, 1, "decstep", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0)])
    _path(out, 2, "decstep", "balance", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    # WHEN: Summarizing.
    summarize(str(out))
    rows = _read(out / "summary" / "paired.csv")
    # THEN: One row contrasts balance minus double over the two paired replicates only.
    assert len(rows) == 1
    row = rows[0]
    assert (row["contrast"], row["minuend"], row["subtrahend"], row["rho"], row["fixed"], row["adopters"]) == (
        "m_model", "balance", "double", "1/2", "decstep", "keep")
    assert (row["observer"], row["n_pairs"]) == ("historical", "2")
    assert float(row["diff_hit"]) == 0.75
    assert (float(row["diff_lo"]), float(row["diff_hi"])) == (0.5, 1.0)


def test_relative_difference_resamples_both_sides_of_the_pair(tmp_path: Path) -> None:
    # GIVEN: Paired hit rates (1, 1/2) and (1/2, 1/4), with a varying baseline.
    out = tmp_path / "out"
    for replicate, comparison, baseline in ((0, [1.0], [1.0, 0.0]),
                                           (1, [1.0, 0.0], [1.0, 0.0, 0.0, 0.0])):
        for model, values in (("balance", comparison), ("double", baseline)):
            _path(out, replicate, "decstep", model, "keep",
                  [_row(HIST, KEEP, value, 0.5, index) for index, value in enumerate(values)])
    # WHEN: Summarizing the paired relative change.
    summarize(str(out))
    row = _read(out / "summary" / "paired.csv")[0]
    # THEN: The ratio of paired means is 100%, and stays 100% under joint resampling.
    assert float(row["relative_hit"]) == 1.0
    assert float(row["relative_lo"]) == 1.0
    assert float(row["relative_hi"]) == 1.0


def test_relative_entropy_pools_events_before_taking_logs(tmp_path: Path) -> None:
    # GIVEN: Balance hits 3/4 and default hits 1/4 over two paired replicates.
    out = tmp_path / "out"
    for replicate, balance, default in ((0, [1.0, 0.0], [0.0, 0.0]),
                                        (1, [1.0, 1.0], [1.0, 0.0])):
        for model, values in (("balance", balance), ("double", default)):
            _path(out, replicate, "decstep", model, "keep",
                  [_row(HIST, KEEP, value, 0.5, i) for i, value in enumerate(values)])
    # WHEN: Computing the entropy-bound relative change.
    summarize(str(out))
    row = _read(out / "summary" / "paired.csv")[0]
    # THEN: The estimate uses pooled hit rates, not averages of replicate entropies.
    assert math.isclose(float(row["relative_bits"]), (-math.log2(0.75) - 2.0) / 2.0)


def test_summarize_is_byte_identical_across_runs(tmp_path: Path) -> None:
    # GIVEN: An output summarized once.
    out = tmp_path / "out"
    _path(out, 0, "popstep", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, NON, 0.0, 0.25, 1)])
    _path(out, 1, "popstep", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0), _row(HIST, NON, 0.0, 0.25, 1)])
    summarize(str(out))
    first = {name: (out / "summary" / name).read_bytes() for name in ("groups.csv", "best.csv", "paired.csv")}
    # WHEN: Summarizing again.
    summarize(str(out))
    # THEN: The three files are unchanged, bootstrap resamples included.
    again = {name: (out / "summary" / name).read_bytes() for name in ("groups.csv", "best.csv", "paired.csv")}
    assert first == again


def _digest(folder: Path) -> str:
    """sha256 of the three old tables' bytes, concatenated in name order."""
    return hashlib.sha256(b"".join((folder / name).read_bytes()
                                   for name in ("groups.csv", "best.csv", "paired.csv"))).hexdigest()


#: sha256 of groups, best and paired for :func:`test_paired_keeps_the_old_tables_without_need_paths`,
#: captured from the release before the need family existed.
_OLD_TABLES_SHA256 = "cb6823841047f269e5a104036c7b76ddc11904c6a61ef63111e5ad35f0abf478"


def test_need_family_contrasts_each_rule_against_the_need_baseline(tmp_path: Path) -> None:
    # GIVEN: One replicate where the need baseline hits 1/2, decstep at double hits 1 and decstep at
    # balance hits 1/2, all on keep events.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 0.0, 0.5, 1)])
    _path(out, 0, "decstep", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 1.0, 0.5, 1)])
    _path(out, 0, "decstep", "balance", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0), _row(HIST, KEEP, 1.0, 0.5, 1)])
    # WHEN: Summarizing.
    summarize(str(out))
    rows = _read(out / "summary" / "paired.csv")
    by_fixed = {row["fixed"]: row for row in rows if row["contrast"] == "need"}
    # THEN: The maximum model family comes first, then the need family, each rule paired with need at double.
    assert [row["contrast"] for row in rows] == ["m_model", "need", "need"]
    assert [row["fixed"] for row in rows if row["contrast"] == "need"] == ["balance", "double"]
    double, balance = by_fixed["double"], by_fixed["balance"]
    assert (double["minuend"], double["subtrahend"], double["rho"], double["adopters"], double["observer"]) == (
        "decstep", "need", "1/2", "keep", "historical")
    assert (double["n_pairs"], float(double["diff_hit"])) == ("1", 0.5)
    assert (float(double["relative_hit"]), float(double["relative_bits"])) == (1.0, -1.0)
    assert (balance["n_pairs"], float(balance["diff_hit"])) == ("1", 0.0)
    assert (float(balance["relative_hit"]), float(balance["relative_bits"])) == (0.0, 0.0)


def test_cost_table_measures_the_over_withdrawal_of_the_rounding(tmp_path: Path) -> None:
    # GIVEN: A keep path whose three events need 100 and withdraw 100 (fallback), 110 and 220.
    out = tmp_path / "out"
    _path(out, 0, "decstep", "double", "keep",
          [_row(HIST, KEEP, 0.0, 0.5, 0, x=100, y=100), _row(HIST, KEEP, 0.0, 0.5, 1, x=100, y=110),
           _row(HIST, KEEP, 0.0, 0.5, 2, x=200, y=220)], k=2)
    # WHEN: Summarizing.
    summarize(str(out))
    row = _read(out / "summary" / "cost.csv")[0]
    # THEN: The costs (y - x)/x are 0, 0.1 and 0.1 over the path's two adopters, and a keep path has
    # no final withdrawal to leave out.
    assert (row["rule"], row["m_model"], row["adopters"], row["n_events"], row["n_final"]) == (
        "decstep", "double", "keep", "3", "0")
    assert math.isclose(float(row["same_share"]), 1 / 3)
    assert math.isclose(float(row["cost_mean"]), 0.2 / 3)
    assert math.isclose(float(row["cost_median"]), 0.1)
    assert math.isclose(float(row["cost_p90"]), 0.1)
    assert math.isclose(float(row["withdrawals_per_adopter"]), 1.5)


def test_cost_table_counts_one_row_per_event_across_observers(tmp_path: Path) -> None:
    # GIVEN: Two events scored by two observers, which repeat the event's amounts.
    out = tmp_path / "out"
    rows = [_row(observer, KEEP, 0.0, 0.5, index, x=100, y=110)
            for observer in (HIST, LOGIT) for index in (0, 1)]
    _path(out, 0, "decstep", "double", "keep", rows, k=1)
    # WHEN: Summarizing.
    summarize(str(out))
    row = _read(out / "summary" / "cost.csv")[0]
    # THEN: Each event is counted once, not once per observer.
    assert row["n_events"] == "2"
    assert math.isclose(float(row["cost_mean"]), 0.1)


def test_cost_table_of_the_need_rule_is_zero(tmp_path: Path) -> None:
    # GIVEN: A need path whose adopter events withdraw exactly the need.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep",
          [_row(HIST, KEEP, 1.0, 0.5, 0, x=100, y=100), _row(HIST, KEEP, 1.0, 0.5, 1, x=200, y=200)], k=1)
    # WHEN: Summarizing.
    summarize(str(out))
    row = _read(out / "summary" / "cost.csv")[0]
    # THEN: Every event withdrew the need: the rounding cost is zero.
    assert (row["rule"], row["n_events"], float(row["same_share"]), float(row["cost_mean"])) == (
        "need", "2", 1.0, 0.0)


def test_cost_table_leaves_out_the_drain_final_withdrawal(tmp_path: Path) -> None:
    # GIVEN: A drain path with two adopters: source 0 rounds a need of 100 up to 110 and then takes its
    # last 50 whole; source 1 rounds 200 to 210 and then takes its last 20 whole.  The run's threshold
    # is 60, so both take-alls fall below it.
    out = tmp_path / "out"
    _manifest(out, [_key(0, "decstep", "double", "drain")], threshold="60")
    _path(out, 0, "decstep", "double", "drain",
          [_row(HIST, DRAIN, 0.0, 0.5, 0, x=100, y=110, source=0),
           _row(HIST, DRAIN, 0.0, 0.5, 1, x=200, y=210, source=1),
           _row(HIST, DRAIN, 0.0, 0.5, 2, x=50, y=50, source=0),
           _row(HIST, DRAIN, 0.0, 0.5, 3, x=20, y=20, source=1)], k=2)
    # WHEN: Summarizing.
    summarize(str(out))
    row = _read(out / "summary" / "cost.csv")[0]
    # THEN: Each source's take-all is counted apart and kept out of the cost statistics, which hold
    # only the two rounded withdrawals, while the final ones still count as withdrawals per adopter.
    # (Regression: the take-all's x = y used to enter as a zero-cost event, understating the drain.)
    assert (row["adopters"], row["n_events"], row["n_final"]) == ("drain", "2", "2")
    assert math.isclose(float(row["same_share"]), 0.0)
    assert math.isclose(float(row["cost_mean"]), 0.075)
    assert math.isclose(float(row["cost_median"]), 0.075)
    assert math.isclose(float(row["withdrawals_per_adopter"]), 2.0)


def test_cost_table_keeps_a_drain_rule_output_that_empties_the_balance_at_the_threshold(
        tmp_path: Path) -> None:
    # GIVEN: A drain path with threshold 100: source 0 rounds 50 up to 60 and then empties its last 100
    # by a rule output (need 80, y = r >= threshold), and source 1 takes its last 60 whole.
    out = tmp_path / "out"
    _manifest(out, [_key(0, "decstep", "double", "drain")], threshold="100")
    _path(out, 0, "decstep", "double", "drain",
          [_row(HIST, DRAIN, 0.0, 0.5, 0, x=50, y=60, source=0),
           _row(HIST, DRAIN, 0.0, 0.5, 1, x=80, y=100, source=0),
           _row(HIST, DRAIN, 0.0, 0.5, 2, x=60, y=60, source=1)], k=2)
    # WHEN: Summarizing.
    summarize(str(out))
    row = _read(out / "summary" / "cost.csv")[0]
    # THEN: Only the last event below the threshold is a take-all; the rule output that empties the
    # balance at the threshold stays in the cost statistics.
    # (Regression: every drain last event used to be counted as a take-all, dropping that output.)
    assert (row["adopters"], row["n_events"], row["n_final"]) == ("drain", "2", "1")
    assert math.isclose(float(row["same_share"]), 0.0)
    assert math.isclose(float(row["cost_mean"]), 0.225)
    assert math.isclose(float(row["cost_median"]), 0.225)
    assert math.isclose(float(row["withdrawals_per_adopter"]), 1.5)


def test_paired_keeps_the_old_tables_without_need_paths(tmp_path: Path) -> None:
    # GIVEN: Only the maximum model and rule families: two rules at both models over two replicates.
    out = tmp_path / "out"
    for replicate in (0, 1):
        for rule in ("decstep", "popstep"):
            for model in ("double", "balance"):
                _path(out, replicate, rule, model, "keep",
                      [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 0.0, 0.5, 1)])
    # WHEN: Summarizing.
    summarize(str(out))
    # THEN: No need row appears, and the three tables keep the release's bytes; a second run reproduces
    # them.  (Regression: the appended family must add no row and shift no bootstrap stream when the
    # output holds no need path.)
    assert {row["contrast"] for row in _read(out / "summary" / "paired.csv")} == {"m_model", "rule"}
    assert _digest(out / "summary") == _OLD_TABLES_SHA256
    summarize(str(out))
    assert _digest(out / "summary") == _OLD_TABLES_SHA256


def test_run_json_rejects_an_output_missing_a_path_directory(tmp_path: Path) -> None:
    # GIVEN: An output of two paths whose run.json records three: the third replicate never landed.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    _path(out, 1, "need", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0)])
    _manifest(out, [_key(replicate) for replicate in (0, 1, 2)])
    # WHEN: Summarizing the truncated output.
    # THEN: It is refused, counting and naming the missing directory, and no table is written.
    with pytest.raises(ValueError, match=re.escape(f"1 missing ({_key(2)})")):
        summarize(str(out))
    assert not (out / "summary").exists()


def test_run_json_rejects_a_path_directory_it_does_not_record(tmp_path: Path) -> None:
    # GIVEN: An output of two paths whose run.json records the first only.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    _path(out, 1, "need", "double", "keep", [_row(HIST, KEEP, 0.0, 0.5, 0)])
    _manifest(out, [_key(0)])
    # WHEN: Summarizing an output grown past its manifest.
    # THEN: It is refused, counting and naming the unrecorded directory.
    with pytest.raises(ValueError, match=re.escape(f"1 extra ({_key(1)})")):
        summarize(str(out))


def test_run_json_rejects_an_output_whose_extra_directories_are_many(tmp_path: Path) -> None:
    # GIVEN: A recorded output of one path, four path folders it does not record.
    out = tmp_path / "out"
    for replicate in range(5):
        _path(out, replicate, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    _manifest(out, [_key(0)])
    # WHEN: Summarizing.
    # THEN: The refusal states the count and names a few, rather than every directory.
    with pytest.raises(ValueError, match=re.escape("4 extra (r001_rho1-2_need_double_keep, "
                                                  "r002_rho1-2_need_double_keep, "
                                                  "r003_rho1-2_need_double_keep, and 1 more)")):
        summarize(str(out))


def test_an_output_without_run_json_summarizes_the_directories_it_has(tmp_path: Path) -> None:
    # GIVEN: One path folder and no run.json, as a partially hand-built output.
    out = tmp_path / "out"
    _path(out, 0, "need", "double", "keep", [_row(HIST, KEEP, 1.0, 0.5, 0)])
    # WHEN: Summarizing.
    result = summarize(str(out))
    # THEN: The paths it finds are summarized; without a manifest there is nothing to compare to.
    assert result["groups"] == 1


def test_a_scratch_directory_linking_every_path_summarizes_the_same(tmp_path: Path) -> None:
    # GIVEN: A complete output and a scratch directory linking each of its path folders plus its
    # run.json, the way an archived run is re-summarized without copying it.
    out, scratch = tmp_path / "out", tmp_path / "scratch"
    for replicate in (0, 1):
        _path(out, replicate, "need", "double", "keep",
              [_row(HIST, KEEP, 1.0, 0.5, 0), _row(HIST, KEEP, 0.0, 0.25, 1)])
    _manifest(out, [_key(replicate) for replicate in (0, 1)], threshold="60")
    scratch.mkdir()
    for folder in sorted(out.iterdir()):
        if (folder / "path.json").is_file():
            _link(folder, scratch / folder.name)
    (scratch / "run.json").write_text((out / "run.json").read_text(encoding="utf-8"), encoding="utf-8")
    # WHEN: Summarizing both.
    summarize(str(out))
    summarize(str(scratch))
    # THEN: The linked scratch directory passes the manifest check and yields the same four tables.
    for name in ("groups.csv", "best.csv", "paired.csv", "cost.csv"):
        assert (scratch / "summary" / name).read_bytes() == (out / "summary" / name).read_bytes()
