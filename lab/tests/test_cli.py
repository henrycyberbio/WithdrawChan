"""Tests for the `lab` commands: `figures` draws only the three main charts and forwards its
parameters to :func:`lab.report.figures.figures`, and `summarize` forwards the explicit
`best_observers` candidate set to :func:`lab.report.summary.summarize`.

The Fire-level tests run the real console entry point on a synthetic four-observer output, so a
JSON list literal on the command line is proven to reach the report layer as a list.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from typing import TYPE_CHECKING

from lab.cli import Lab
from lab.report.summary import GROUPS_COLUMNS, PAIRED_COLUMNS
from lab.score import write_rows
from lab.sim import KEEP

if TYPE_CHECKING:
    from pathlib import Path

FIGURE_NAMES = ("fig1", "fig2", "fig3")
FOUR = ("historical", "trained-logit", "combinatorial", "group")


def _best(rho: str, rule: str, m_model: str, adopters: str, group: str, bits: float) -> dict[str, str]:
    row = dict.fromkeys(GROUPS_COLUMNS, "0")
    row.update({"rho": rho, "rule": rule, "m_model": m_model, "adopters": adopters, "group": group,
                "observer": "combinatorial", "bits": repr(bits), "bits_lo": repr(bits - 0.5),
                "bits_hi": repr(bits + 0.5)})
    return row


# Two rules with different bounds, so a forwarded `rule` visibly changes figure 1.
BEST = [
    _best("0/1", "need", "double", "keep", "non", 1.0),
    _best("1/2", "decsmooth", "double", "keep", "non", 1.5),
    _best("1/2", "decsmooth", "double", "keep", "keep", 6.0),
    _best("1/10", "decsmooth", "double", "keep", "keep", 5.0),
    _best("1/2", "decsmooth", "double", "drain", "drain", 3.0),
    _best("1/2", "decstep", "double", "keep", "keep", 4.0),
    _best("1/2", "decstep", "double", "drain", "drain", 2.0),
]


def _write_summary(out: Path) -> None:
    """Write the three summary tables the command reads, at two adoption rates."""
    (out / "summary").mkdir(parents=True, exist_ok=True)
    with (out / "summary" / "best.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(BEST)
    with (out / "summary" / "groups.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows([_best(rho, rule, "double", group, group, 5.0 + index / 10)
                          for rho in ("1/2", "1") for group in ("keep", "drain")
                          for index, rule in enumerate(("decstep", "popstep", "decsmooth", "popsmooth"))])
    with (out / "summary" / "paired.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        rates = {"1/2": 0.05, "1": 0.25}
        for rho, rate in rates.items():
            for group in ("keep", "drain"):
                for index, fixed in enumerate(("decstep", "decsmooth")):
                    value = -(rate + index * 0.01)
                    row = dict.fromkeys(PAIRED_COLUMNS, "0")
                    row.update({"observer": "combinatorial", "adopters": group, "rho": rho,
                                "contrast": "m_model", "minuend": "balance", "subtrahend": "double",
                                "fixed": fixed, "relative_hit": "0.1", "relative_lo": "-0.1",
                                "relative_hi": "0.3", "relative_bits": repr(value),
                                "relative_bits_lo": repr(value - 0.01), "relative_bits_hi": repr(value + 0.01)})
                    writer.writerow(row)


def test_figures_command_draws_only_the_three_main_charts(tmp_path: Path) -> None:
    # GIVEN: A summarized experiment output.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: The CLI draws its figures.
    written = Lab().figures(str(out))
    # THEN: Only figures 1-3 are named and written, as PNGs under <out>/figures.
    assert set(written) == set(FIGURE_NAMES)
    assert sorted(path.name for path in (out / "figures").iterdir()) == \
        [f"{name}.png" for name in FIGURE_NAMES]
    for name in FIGURE_NAMES:
        assert (out / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_figures_command_forwards_the_tool_setting(tmp_path: Path) -> None:
    # GIVEN: A summarized output whose best table holds two rules with different bounds.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing figure 1 under each rule.
    Lab().figures(str(out), rule="decsmooth")
    smooth = (out / "figures" / "fig1.png").read_bytes()
    Lab().figures(str(out), rule="decstep")
    step = (out / "figures" / "fig1.png").read_bytes()
    # THEN: The forwarded rule reaches the plot, so the two figures differ.
    assert smooth != step


def test_figures_command_forwards_the_figure1_observer(tmp_path: Path) -> None:
    # GIVEN: A summarized output whose groups table has rows the envelope table does not.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing figure 1 with a named observer selected for it.
    Lab().figures(str(out))
    envelope = (out / "figures" / "fig1.png").read_bytes()
    written = Lab().figures(str(out), figure1_observer="combinatorial")
    selected = (out / "figures" / "fig1.png").read_bytes()
    # THEN: The forwarded observer reaches the plot, and figure 1 is still under <out>/figures.
    assert written["fig1"] == str(out / "figures" / "fig1.png")
    assert selected != envelope


def test_figures_command_forwards_the_comparison_selection(tmp_path: Path) -> None:
    # GIVEN: A summarized output with paired rows at two adoption rates.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing the comparison chart at each adoption rate.
    Lab().figures(str(out), comparison_rho="1/2")
    half = (out / "figures" / "fig2.png").read_bytes()
    Lab().figures(str(out), comparison_rho="1")
    whole = (out / "figures" / "fig2.png").read_bytes()
    # THEN: The forwarded comparison rate reaches the plot, so the two figures differ.
    assert half != whole


def _composite_summary(out: Path) -> None:
    """Write an output whose combined tables win each cell for a different observer."""
    (out / "summary").mkdir(parents=True, exist_ok=True)
    winners = {"keep": ("historical", "combinatorial", "group", "trained-logit"),
               "drain": ("group", "trained-logit", "combinatorial", "historical")}
    best = [_best("0/1", "need", "double", "keep", "non", 1.0),
            _best("1/2", "decsmooth", "double", "keep", "non", 1.5)]
    best[0]["observer"], best[1]["observer"] = "historical", "trained-logit"
    for group, observers in winners.items():
        for index, (rule, bits) in enumerate((("decstep", 5.0), ("popstep", 5.1),
                                              ("decsmooth", 6.0), ("popsmooth", 6.1))):
            row = _best("1/2", rule, "double", group, group, bits)
            row["observer"] = observers[index]
            best.append(row)
    with (out / "summary" / "best.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(best)
    with (out / "summary" / "groups.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows([_best("1/2", rule, "double", group, group, 9.0 + index / 10)
                          for group in ("keep", "drain")
                          for index, rule in enumerate(("decstep", "popstep", "decsmooth", "popsmooth"))])
    with (out / "summary" / "paired.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        for observer in ("combinatorial", "best"):
            for group in ("keep", "drain"):
                for index, fixed in enumerate(("decstep", "decsmooth")):
                    value = -0.11 - index * 0.01 if observer == "best" else -0.05
                    row = dict.fromkeys(PAIRED_COLUMNS, "0")
                    row.update({"observer": observer, "adopters": group, "rho": "1/2",
                                "contrast": "m_model", "minuend": "balance", "subtrahend": "double",
                                "fixed": fixed, "relative_hit": "0.1", "relative_lo": "-0.1",
                                "relative_hi": "0.3", "relative_bits": repr(value),
                                "relative_bits_lo": repr(value - 0.01),
                                "relative_bits_hi": repr(value + 0.01)})
                    writer.writerow(row)


def test_figures_command_draws_the_composite_comparison(tmp_path: Path) -> None:
    # GIVEN: An output whose best table wins each cell for a different observer.
    out = tmp_path / "out"
    _composite_summary(out)
    # WHEN: The command draws from the composite tables.
    written = Lab().figures(str(out), comparison_observer="best")
    # THEN: All three charts are produced under <out>/figures as PNGs.
    assert written == {name: str(out / "figures" / f"{name}.png") for name in FIGURE_NAMES}
    for name in FIGURE_NAMES:
        assert (out / "figures" / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _write_observer_path(out: Path, replicate: int, m_model: str, adopters: str,
                         hits: dict[str, list[float]], k: int = 1) -> None:
    """Write one `decstep` path folder holding a row per (observer, hit value)."""
    folder = out / f"r{replicate:03d}_rho1-2_decstep_{m_model}_{adopters}"
    folder.mkdir(parents=True)
    spec = {"replicate": replicate, "rho": "1/2", "rule": "decstep", "m_model": m_model,
            "adopters": adopters}
    (folder / "path.json").write_text(json.dumps({"spec": spec, "K": k}), encoding="utf-8")
    write_rows(folder / "rows.csv.gz", [
        {"path": "p", "observer": observer, "public_index": index, "source": 0, "kind": KEEP,
         "x": 100, "y": 100, "p_true": 0.5, "true_rank": 1.0, "top1": value,
         "max_posterior": 0.5}
        for observer, values in hits.items() for index, value in enumerate(values)])


def _four_observer_out(tmp_path: Path) -> Path:
    """Two replicates of a balance/double cell whose combined winner differs by arm."""
    out = tmp_path / "out"
    arms = {
        (0, "balance"): {"historical": [1.0, 1.0], "trained-logit": [0.0, 0.0],
                         "combinatorial": [0.0, 0.0], "group": [0.0, 0.0]},
        (0, "double"): {"historical": [0.0], "trained-logit": [1.0],
                        "combinatorial": [0.0], "group": [0.0]},
        (1, "balance"): {"historical": [1.0, 0.0], "trained-logit": [1.0, 1.0],
                         "combinatorial": [0.0, 0.0], "group": [0.0, 0.0]},
        (1, "double"): {"historical": [0.0], "trained-logit": [0.0],
                        "combinatorial": [0.0], "group": [0.0]},
    }
    for (replicate, m_model), hits in arms.items():
        _write_observer_path(out, replicate, m_model, "keep", hits)
    return out


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the real Fire entry point, as the installed `lab` console script does."""
    return subprocess.run([sys.executable, "-c", "from lab.cli import main; main()", *args],
                          capture_output=True, text=True, check=False)


def test_summarize_cli_parses_the_json_list_and_writes_the_combined_best(tmp_path: Path) -> None:
    # GIVEN: A synthetic four-observer output whose per-arm merged winners differ.
    out = _four_observer_out(tmp_path)
    # WHEN: The real CLI receives the candidate set as a JSON list literal.
    result = _run_cli("summarize", str(out), f"--best-observers={json.dumps(list(FOUR))}")
    # THEN: Fire parsed it into a list, summarize accepted it, and the combined table is real:
    # balance pools historical at 3/4, double pools trained-logit at 1/2.
    assert result.returncode == 0, result.stderr
    with (out / "summary" / "best.csv").open(newline="", encoding="utf-8") as fh:
        best = list(csv.DictReader(fh))
    assert [(row["m_model"], row["observer"], float(row["hit_rate"])) for row in best] == [
        ("balance", "historical", 0.75), ("double", "trained-logit", 0.5)]
    with (out / "summary" / "paired.csv").open(newline="", encoding="utf-8") as fh:
        composite = [row for row in csv.DictReader(fh) if row["observer"] == "best"]
    assert len(composite) == 1
    assert (composite[0]["minuend"], composite[0]["subtrahend"]) == ("balance", "double")
    assert float(composite[0]["diff_hit"]) == 0.25
