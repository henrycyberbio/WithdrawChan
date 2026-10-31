"""Tests for the figures: the slices behind figures 1-3 and the PNG output."""

from __future__ import annotations

import csv
import math
import warnings
from fractions import Fraction
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ft2font import FT2Font
from matplotlib.transforms import Bbox

import lab.report.figures
from lab.report.figures import figure1_series, figure2_bars, figure3_bars, figures, plot_bars
from lab.report.summary import GROUPS_COLUMNS, PAIRED_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.text import Annotation

# The three PNGs a formal run produces; the one-off overview is not part of this module.
FIGURE_NAMES = ("fig1", "fig2", "fig3")


def _best(rho: str, rule: str, m_model: str, adopters: str, group: str, bits: float) -> dict[str, str]:
    """One `best.csv` row whose interval is `bits` +- 0.5."""
    row = dict.fromkeys(GROUPS_COLUMNS, "0")
    row.update({"rho": rho, "rule": rule, "m_model": m_model, "adopters": adopters, "group": group,
                "observer": "combinatorial", "bits": repr(bits), "bits_lo": repr(bits - 0.5),
                "bits_hi": repr(bits + 0.5)})
    return row


ROWS = [
    _best("0/1", "need", "double", "keep", "non", 1.0),
    _best("1/2", "decsmooth", "double", "keep", "non", 1.5),
    _best("1/2", "decsmooth", "double", "keep", "keep", 6.0),
    _best("1/10", "decsmooth", "double", "keep", "keep", 5.0),
    _best("1/2", "decsmooth", "double", "drain", "drain", 3.0),
    # Rows figure 1 must leave out: the drain run's non-adopters, another rule, another maximum model.
    _best("1/2", "decsmooth", "double", "drain", "non", 9.0),
    _best("1/2", "decstep", "double", "keep", "keep", 9.0),
    _best("1/2", "decsmooth", "balance", "keep", "keep", 9.0),
]


def _write_summary(out: Path) -> None:
    """Write the three summary tables a formal run leaves behind, and nothing else."""
    (out / "summary").mkdir(parents=True, exist_ok=True)
    with (out / "summary" / "best.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(ROWS)
    with (out / "summary" / "groups.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows([_best("1/2", rule, "double", group, group, bits)
                          for group in ("keep", "drain")
                          for rule, bits in (("decstep", 5.0), ("popstep", 5.1),
                                             ("decsmooth", 6.0), ("popsmooth", 6.1))])
    with (out / "summary" / "paired.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        for observer in ("historical", "trained-logit", "combinatorial"):
            for group in ("keep", "drain"):
                for contrast, minuend, subtrahend, fixed in (
                        ("m_model", "balance", "double", "decstep"),
                        ("m_model", "balance", "double", "decsmooth"),
                        ("rule", "popstep", "decstep", "double"),
                        ("rule", "popsmooth", "decsmooth", "double")):
                    row = dict.fromkeys(PAIRED_COLUMNS, "0")
                    row.update({"observer": observer, "adopters": group, "rho": "1/2",
                                "contrast": contrast, "minuend": minuend, "subtrahend": subtrahend,
                                "fixed": fixed, "relative_hit": "0.1", "relative_lo": "-0.1",
                                "relative_hi": "0.3"})
                    row.update({"relative_bits": "-0.05", "relative_bits_lo": "-0.08",
                                "relative_bits_hi": "-0.02"})
                    writer.writerow(row)


def test_figure1_series_take_need_at_zero_and_the_chosen_setting_elsewhere() -> None:
    # GIVEN: Best rows for several rules, maximum models and runs.
    # WHEN: Slicing figure 1 for decsmooth under the default M.
    series = figure1_series(ROWS, rule="decsmooth", m_model="double")
    # THEN: Non-adopters come from the keep run with the need rule at rho = 0, adopters from their
    # own run's own group, every line sorted by rho, and no other setting leaks in.
    assert series == {
        "Non-WithdrawChan": [(Fraction(0), 1.0, 0.5, 1.5), (Fraction(1, 2), 1.5, 1.0, 2.0)],
        "Keep": [(Fraction(1, 10), 5.0, 4.5, 5.5), (Fraction(1, 2), 6.0, 5.5, 6.5)],
        "Drain": [(Fraction(1, 2), 3.0, 2.5, 3.5)],
    }


def test_figure2_bars_use_relative_entropy_at_one_adoption_rate() -> None:
    # GIVEN: Entropy contrasts for both rules and groups, plus an unwanted observer.
    rows = [{"contrast": "m_model", "minuend": "balance", "subtrahend": "double",
             "rho": rho, "fixed": rule, "adopters": group, "observer": observer,
             "relative_bits": "-0.05", "relative_bits_lo": "-0.08", "relative_bits_hi": "-0.02"}
            for rho in ("1/2", "1/10") for rule in ("decstep", "decsmooth")
            for group in ("keep", "drain") for observer in ("combinatorial", "historical")]
    # WHEN: Taking the main bar chart slice.
    bars = figure2_bars(rows, rho=Fraction(1, 2), observer="combinatorial")
    # THEN: Each group has step and random-walk entropy changes, not hit-rate changes.
    assert bars == {"Keep": [(-0.05, -0.08, -0.02)] * 2,
                    "Drain": [(-0.05, -0.08, -0.02)] * 2}


def test_figure3_bars_show_each_ladders_absolute_bound() -> None:
    # GIVEN: Both ladders under both maximum models and two adoption rates.
    rows = [_best(rho, rule, cap, group, group, bits)
            for rho in ("1/2", "1") for cap in ("double", "balance") for group in ("keep", "drain")
            for rule, bits in (("decstep", 5.0), ("popstep", 5.1), ("decsmooth", 6.0), ("popsmooth", 6.1))]
    # WHEN: Selecting absolute bounds from the combinatorial observer at 50% adoption.
    bars = figure3_bars(rows, rho=Fraction(1, 2), observer="combinatorial", m_model="double")
    # THEN: Each behaviour panel compares decimal and popular bounds in bits for both rules.
    for panel in bars.values():
        assert panel == {"Decimal ladder": [(5.0, 4.5, 5.5), (6.0, 5.5, 6.5)],
                         "Popular ladder": [(5.1, 4.6, 5.6), (6.1, 5.6, 6.6)]}


def test_bar_renderer_uses_columns_and_keeps_the_zero_baseline(tmp_path: Path) -> None:
    # GIVEN: Four relative entropy changes with intervals.
    panels = {"": {"Keep": [(-0.06, -0.08, -0.04), (-0.01, -0.02, -0.005)],
                    "Drain": [(-0.08, -0.1, -0.06), (-0.02, -0.03, -0.01)]}}
    # WHEN: Rendering the relative comparison as a bar chart.
    figure = plot_bars(panels, tmp_path / "bars.png", relative=True,
                       colors=(("#7cb5ec", "#f7a35c"),))
    # THEN: All four estimates are columns anchored at zero, and zero is visible.
    axes = figure.axes[0]
    assert len(axes.patches) == 4
    assert all(bar.get_y() == 0 for bar in axes.patches)
    assert axes.get_ylim()[0] < 0 < axes.get_ylim()[1]


def test_bar_renderer_keeps_estimates_when_intervals_are_undefined(tmp_path: Path) -> None:
    # GIVEN: Finite estimates whose bootstrap intervals are undefined.
    nan = float("nan")
    panels = {"": {"Keep": [(-0.06, nan, nan), (-0.01, nan, nan)],
                    "Drain": [(-0.08, nan, nan), (-0.02, nan, nan)]}}
    # WHEN: Rendering the comparison.
    figure = plot_bars(panels, tmp_path / "undefined-intervals.png", relative=True,
                       colors=(("#7cb5ec", "#f7a35c"),))
    # THEN: The four finite estimates remain columns, without inventing intervals.
    assert len(figure.axes[0].patches) == 4


def test_bar_renderer_marks_undefined_estimates_without_zero_columns(tmp_path: Path) -> None:
    # GIVEN: A legitimate zero-baseline comparison with no defined estimates or intervals.
    nan = float("nan")
    panels = {"": {"Keep": [(nan, nan, nan)] * 2, "Drain": [(nan, nan, nan)] * 2}}
    # WHEN: Rendering the unavailable comparison.
    figure = plot_bars(panels, tmp_path / "undefined-estimates.png", relative=True,
                       colors=(("#7cb5ec", "#f7a35c"),))
    # THEN: No zero-effect columns are invented, and the unavailable state is explicit.
    assert not figure.axes[0].patches
    assert any(text.get_text() == "Estimate unavailable" for text in figure.axes[0].texts)


def test_bar_renderer_keeps_interval_free_estimates_visible(tmp_path: Path) -> None:
    # GIVEN: Defined intervals near zero and an interval-free estimate far below them.
    nan = float("nan")
    panels = {"": {"Keep": [(-0.8, nan, nan), (-0.01, -0.02, -0.005)],
                    "Drain": [(-0.08, -0.1, -0.06), (-0.02, -0.03, -0.01)]}}
    # WHEN: Rendering mixed defined and undefined intervals.
    figure = plot_bars(panels, tmp_path / "mixed-intervals.png", relative=True,
                       colors=(("#7cb5ec", "#f7a35c"),))
    # THEN: All finite estimates remain inside the displayed axis limits.
    lower, upper = figure.axes[0].get_ylim()
    assert lower < -0.8 < upper


def test_figures_write_the_same_pngs_twice(tmp_path: Path) -> None:
    # GIVEN: A summarized output: only the three summary tables, no run.json.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing the figures twice in this process.
    first = figures(str(out), comparison_rho="1/2", comparison_observer="combinatorial")
    png = (out / "figures" / "fig1.png").read_bytes()
    again = figures(str(out), comparison_rho="1/2", comparison_observer="combinatorial")
    # THEN: Figure 1 is a PNG under <out>/figures and a rerun reproduces its bytes.
    assert first == again == {name: str(out / "figures" / f"{name}.png") for name in FIGURE_NAMES}
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert (out / "figures" / "fig1.png").read_bytes() == png


def test_figures_need_only_the_three_summary_tables(tmp_path: Path) -> None:
    # GIVEN: A summarized output without run.json, the deposit count the overview used to read.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing the formal figures.
    written = figures(str(out))
    # THEN: Exactly figures 1-3 are produced, and no run.json is required or created.
    assert written == {name: str(out / "figures" / f"{name}.png") for name in FIGURE_NAMES}
    assert sorted(path.name for path in (out / "figures").iterdir()) == \
        [f"{name}.png" for name in FIGURE_NAMES]
    assert not (out / "run.json").exists()


def test_figures_leave_a_pre_existing_overview_untouched(tmp_path: Path) -> None:
    # GIVEN: An output folder that already holds a stale one-off overview.png.
    out = tmp_path / "out"
    _write_summary(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    stale = out / "figures" / "overview.png"
    stale.write_bytes(b"stale-overview-bytes")
    # WHEN: Drawing the formal figures.
    figures(str(out))
    # THEN: The stale overview is neither rewritten nor deleted, and nothing else is swept away.
    assert stale.read_bytes() == b"stale-overview-bytes"
    assert sorted(path.name for path in (out / "figures").iterdir()) == \
        ["fig1.png", "fig2.png", "fig3.png", "overview.png"]


def test_figures_read_figure1_from_a_named_observer(tmp_path: Path) -> None:
    # GIVEN: A group run whose envelope table is header-only and whose groups table also holds the
    # usual comparison rows under another observer.
    out = tmp_path / "out"
    _write_summary(out)
    with (out / "summary" / "best.csv").open("w", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, GROUPS_COLUMNS).writeheader()
    rows = []
    for rho, rule, m_model, adopters, group, value in (
            ("0/1", "need", "double", "keep", "non", 2.0),
            ("1/10", "decsmooth", "double", "keep", "keep", 5.5),
            ("1/2", "decsmooth", "double", "keep", "keep", 7.0),
            ("1/2", "decsmooth", "double", "drain", "drain", 4.0)):
        row = _best(rho, rule, m_model, adopters, group, value)
        row["observer"] = "group"
        rows.append(row)
    with (out / "summary" / "groups.csv").open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, GROUPS_COLUMNS).writerows(rows)
    # WHEN: Drawing with the group observer selected for figure 1, on the usual figure 1 renderer.
    with patch.object(lab.report.figures, "plot_figure1") as plot:
        figures(str(out), figure1_observer="group")
    # THEN: Figure 1 gets that observer's rows through the usual slices, not best.csv's data.
    assert plot.call_args.args[0] == {
        "Non-WithdrawChan": [(Fraction(0), 2.0, 1.5, 2.5)],
        "Keep": [(Fraction(1, 10), 5.5, 5.0, 6.0), (Fraction(1, 2), 7.0, 6.5, 7.5)],
        "Drain": [(Fraction(1, 2), 4.0, 3.5, 4.5)],
    }


def test_figures_reject_an_unavailable_figure1_observer(tmp_path: Path) -> None:
    # GIVEN: A public run with a full envelope table but no group observer in its groups table.
    out = tmp_path / "out"
    _write_summary(out)

    # WHEN: Asking figure 1 for that observer.
    with pytest.raises(ValueError, match="group"):
        figures(str(out), figure1_observer="group")
    # THEN: The selection is rejected instead of silently falling back to best.csv.
    assert not (out / "figures" / "fig1.png").exists()


def test_plot_figure1_marks_an_unbounded_interval_side(tmp_path: Path) -> None:
    # GIVEN: A finite estimate whose upper interval bound is positive infinity.
    inf = float("inf")
    series = {"Non-WithdrawChan": [(Fraction(0), 1.0, 0.5, 1.5)],
              "Keep": [(Fraction(1, 2), 6.0, 5.0, inf)],
              "Drain": [(Fraction(1, 2), 3.0, 2.5, 3.5)]}
    captured: list[Figure] = []
    original = Figure.savefig

    def record(figure: Figure, *args: object, **kwargs: object) -> object:
        captured.append(figure)
        return original(figure, *args, **kwargs)

    # WHEN: Drawing figure 1.
    with patch.object(Figure, "savefig", record):
        lab.report.figures.plot_figure1(series, tmp_path / "unbounded.png")
    # THEN: The point and its finite lower bound stay, the axis stays finite, and the open side is
    # marked visibly above the point.
    axes = captured[0].axes[0]
    assert math.isfinite(axes.get_ylim()[1])
    assert axes.get_ylim()[1] >= 6.0
    assert any(tuple(float(value) for value in line.get_ydata()) == (6.0,) for line in axes.lines)
    assert any(annotation.arrow_patch is not None for annotation in axes.texts)


def test_plot_figure1_marks_an_unbounded_side_clear_of_a_shared_x_bar(tmp_path: Path) -> None:
    # GIVEN: The group slice at rho = 0.01, where Keep's upper bound is infinite and Drain's bar at
    # the same rho is long and finite.
    inf = float("inf")
    rho = Fraction(1, 100)
    series = {"Non-WithdrawChan": [(rho, 1.9533073483556984, 1.8821170093534172, 2.0270526191232974)],
              "Keep": [(rho, 5.643856189774724, 4.689299160535891, inf)],
              "Drain": [(rho, 6.409222878004727, 5.065344227840601, 12.20945336562895)]}
    captured: list[Figure] = []
    original = Figure.savefig

    def record(figure: Figure, *args: object, **kwargs: object) -> object:
        captured.append(figure)
        return original(figure, *args, **kwargs)

    # WHEN: Drawing figure 1.
    with patch.object(Figure, "savefig", record):
        lab.report.figures.plot_figure1(series, tmp_path / "collinear.png")

    # THEN: The finite lower whisker still runs from 4.689 bits to Keep's point, Drain's finite bar
    # is drawn at the same rho, and the infinity mark is clear of that bar instead of being crossed.
    figure = captured[0]
    canvas = FigureCanvasAgg(figure)
    figure.set_canvas(canvas)
    canvas.draw()
    axes = figure.axes[0]
    segments = [segment for line in axes.collections for segment in line.get_segments()]
    assert any(_vertical_segment(segment, rho, 4.689299160535891, 5.643856189774724)
               for segment in segments)
    assert any(_vertical_segment(segment, rho, 5.065344227840601, 12.20945336562895)
               for segment in segments)
    marks = [text for text in axes.texts if text.get_text() == "∞"]
    assert len(marks) == 1
    bar_x = axes.transData.transform((float(rho), 0.0))[0]
    assert marks[0].get_window_extent(canvas.get_renderer()).x0 > bar_x + 1.0


def test_plot_figure1_infinity_mark_uses_a_face_with_the_glyph(tmp_path: Path) -> None:
    # GIVEN: An unbounded upper side, on a platform whose UI sans-serif may not cover U+221E.
    inf = float("inf")
    series = {"Non-WithdrawChan": [(Fraction(0), 1.0, 0.5, 1.5)],
              "Keep": [(Fraction(1, 2), 6.0, 5.0, inf)],
              "Drain": [(Fraction(1, 2), 3.0, 2.5, 3.5)]}
    captured: list[Figure] = []
    original = Figure.savefig

    def record(figure: Figure, *args: object, **kwargs: object) -> object:
        captured.append(figure)
        return original(figure, *args, **kwargs)

    # WHEN: Drawing and saving the figure for real, watching for a missing-glyph warning.
    with patch.object(Figure, "savefig", record), warnings.catch_warnings(record=True) as drawn:
        warnings.simplefilter("always")
        lab.report.figures.plot_figure1(series, tmp_path / "infinity-face.png")

    # THEN: The mark's own face, resolved under the figure's rc, carries U+221E, so the PNG shows an
    # infinity sign instead of a box.
    mark = next(text for text in captured[0].axes[0].texts if text.get_text() == "∞")
    with rc_context(lab.report.figures.RC):
        face = font_manager.findfont(mark.get_fontproperties())
    assert FT2Font(face).get_char_index(0x221E) > 0, face
    assert not [item for item in drawn if "8734" in str(item.message)]


def _point_labels(axes: object) -> list[Annotation]:
    """The two-decimal point labels of figure 1, which are every text except the infinity marks."""
    return [text for text in axes.texts if text.get_text() != "∞"]


def _real_point_series() -> dict[str, list[tuple[Fraction, float, float, float]]]:
    """The formal USDC figure 1 slice: each cell's combined best estimate and its 95% interval.

    These are the real `best.csv` values of the plot (`decsmooth` under `double`, and `need` at
    `rho = 0`), copied to full precision.  The 1% and 2.5% Keep intervals are much taller than the
    short fixed label offsets, so a label placed above its own point lands inside its own whisker:
    the shape that forces the label aside instead of only away from other labels.
    """
    non = [
        (Fraction(0), 1.5438383882053663, 1.5077723712888187, 1.5865262549573322),
        (Fraction(1, 100), 1.5463818428037828, 1.5121392502039024, 1.5878484670317945),
        (Fraction(1, 40), 1.5539218986369439, 1.5180383433312326, 1.5955356652796102),
        (Fraction(1, 10), 1.5580136180831803, 1.5174545354067928, 1.6001169386971434),
        (Fraction(1, 4), 1.5618152618131707, 1.5130719550447917, 1.6175561697364023),
        (Fraction(1, 2), 1.5670120573376418, 1.5054717251030834, 1.618294968916241),
        (Fraction(3, 4), 1.5811177374653789, 1.4957595193315367, 1.6371059282803586),
    ]
    keep = [
        (Fraction(1, 100), 6.845490050944375, 5.824428435416546, 8.562452020316442),
        (Fraction(1, 40), 6.656362073778205, 6.012871807279168, 7.5242416286076175),
        (Fraction(1, 10), 6.802386182661051, 6.368215521093075, 7.256177903709661),
        (Fraction(1, 4), 7.418470345336644, 7.047886652866816, 7.713240876656231),
        (Fraction(1, 2), 7.652726036591851, 7.455640939468474, 7.8964932983776635),
        (Fraction(3, 4), 7.560728075821118, 7.267762878776705, 7.867622919176808),
        (Fraction(1, 1), 7.375533471709177, 7.17969814759368, 7.597160566762177),
    ]
    drain = [
        (Fraction(1, 100), 3.917537839808027, 3.3660604806394474, 4.429782566781849),
        (Fraction(1, 40), 3.860286858177412, 3.662849369493117, 4.0265341638401475),
        (Fraction(1, 10), 3.834719184489455, 3.755144036591892, 3.9312119624030992),
        (Fraction(1, 4), 3.8093843016693274, 3.7154278532943223, 3.904965262718998),
        (Fraction(1, 2), 3.8188677628446044, 3.7494426818979383, 3.8962461362395775),
        (Fraction(3, 4), 3.815190872450726, 3.780305173473867, 3.8490538391711335),
        (Fraction(1, 1), 3.802974849012769, 3.7825750943924787, 3.821735226467691),
    ]
    return {"Non-WithdrawChan": non, "Keep": keep, "Drain": drain}


def _interval_boxes(axes: object,
                    series: dict[str, list[tuple[Fraction, float, float, float]]]) -> list[Bbox]:
    """The displayed area of every finite interval side: its whisker and the cap across it.

    Each drawn side is a vertical line from the estimate to the bound, `capsize` wide at the bound;
    the box spans the cap's width so a label must clear the whole mark, not just its centre line.
    """
    style = lab.report.figures.STYLE
    pixels = axes.figure.dpi / 72
    half_cap = style["capsize"] * pixels
    line = max(style["errorbar_width"] * pixels, 1.0)
    boxes = []
    for points in series.values():
        for rho, bits, lo, hi in points:
            x, y = axes.transData.transform((float(rho), bits))
            for bound in (lo, hi):
                if not math.isfinite(bound):
                    continue
                _x, edge = axes.transData.transform((float(rho), bound))
                low, high = min(y, edge - line / 2), max(y, edge + line / 2)
                boxes.append(Bbox.from_extents(x - half_cap, low, x + half_cap, high))
    return boxes


def test_plot_figure1_labels_clear_every_interval_whisker(tmp_path: Path) -> None:
    # GIVEN: The real USDC slice, whose tall low-adoption intervals cross the short label offsets.
    series = _real_point_series()
    captured: list[Figure] = []
    original = Figure.savefig

    def record(figure: Figure, *args: object, **kwargs: object) -> object:
        captured.append(figure)
        return original(figure, *args, **kwargs)

    # WHEN: Drawing figure 1 and measuring it at the saved image's dpi.
    with patch.object(Figure, "savefig", record):
        lab.report.figures.plot_figure1(series, tmp_path / "whiskers.png")
    figure = captured[0]
    figure.set_dpi(lab.report.figures.STYLE["dpi"])
    canvas = FigureCanvasAgg(figure)
    figure.set_canvas(canvas)
    canvas.draw()
    renderer = canvas.get_renderer()
    axes = figure.axes[0]
    labels = _point_labels(axes)
    boxes = [text.get_window_extent(renderer) for text in labels]
    intervals = _interval_boxes(axes, series)
    axes_box = axes.get_window_extent(renderer)

    # THEN: All 21 points keep their own two-decimal label ...
    expected = {(float(rho), bits): f"{bits:.2f}"
                for points in series.values() for rho, bits, _lo, _hi in points}
    assert len(expected) == 21
    assert sorted((tuple(float(value) for value in text.xy), text.get_text()) for text in labels) == \
        sorted(expected.items())
    # ... every label stays inside the axes, out of the layout ...
    assert all(text.get_in_layout() is False for text in labels)
    assert all(axes_box.x0 <= box.x0 and box.x1 <= axes_box.x1
               and axes_box.y0 <= box.y0 and box.y1 <= axes_box.y1 for box in boxes)
    # ... and no label touches another label or any whisker, cap or marker it does not belong to.
    assert len(intervals) == 42
    crossed = [(labels[index].get_text(), index)
               for index, box in enumerate(boxes)
               for interval in intervals if box.overlaps(interval)]
    assert not crossed
    overlapping = [(labels[first].get_text(), labels[second].get_text())
                   for first in range(len(boxes)) for second in range(first + 1, len(boxes))
                   if boxes[first].overlaps(boxes[second])]
    assert not overlapping


def _vertical_segment(segment: tuple[tuple[float, float], tuple[float, float]],
                      rho: Fraction, low: float, high: float) -> bool:
    """Whether an errorbar segment is the vertical bar at `rho` spanning `low` to `high` bits."""
    (x0, y0), (x1, y1) = segment
    if abs(float(x0) - float(rho)) > 1e-9 or abs(float(x1) - float(rho)) > 1e-9:
        return False
    return (math.isclose(min(float(y0), float(y1)), low, rel_tol=0, abs_tol=1e-9)
            and math.isclose(max(float(y0), float(y1)), high, rel_tol=0, abs_tol=1e-9))


def test_plot_figure1_rejects_a_nonfinite_point_estimate(tmp_path: Path) -> None:
    # GIVEN: A cell whose own bound is not finite, so there is no point to place.
    inf = float("inf")
    series = {"Non-WithdrawChan": [(Fraction(0), 1.0, 0.5, 1.5)],
              "Keep": [(Fraction(1, 2), inf, inf, inf)],
              "Drain": [(Fraction(1, 2), 3.0, 2.5, 3.5)]}
    # WHEN: Drawing it.
    with pytest.raises(ValueError, match="1/2"):
        lab.report.figures.plot_figure1(series, tmp_path / "nonfinite.png")


def test_figures_reject_an_empty_figure1_selection(tmp_path: Path) -> None:
    # GIVEN: An observer whose table has no row for the requested rule and maximum model.
    out = tmp_path / "out"
    _write_summary(out)
    rows = []
    for group in ("keep", "drain"):
        row = _best("1/2", "decstep", "double", group, group, 4.0)
        row["observer"] = "group"
        rows.append(row)
    with (out / "summary" / "groups.csv").open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, GROUPS_COLUMNS).writerows(rows)

    # WHEN: Slicing figure 1 at the default rule.
    with pytest.raises(ValueError, match="figure 1"):
        figures(str(out), figure1_observer="group")
    # THEN: It is reported unavailable rather than crashing on an empty maximum or index.


def _rendered_text(figure: Figure) -> set[str]:
    """Every text a saved figure shows: legend entries and in-axes annotations."""
    texts: set[str] = set()
    for axes in figure.axes:
        legend = axes.get_legend()
        if legend is not None:
            texts |= {entry.get_text() for entry in legend.get_texts()}
        texts |= {entry.get_text() for entry in axes.texts}
    return texts


def test_all_three_figures_render_the_short_group_labels(tmp_path: Path) -> None:
    # GIVEN: A summarized output holding the three summary tables and nothing else.
    out = tmp_path / "out"
    _write_summary(out)
    # WHEN: Drawing the formal figures, capturing each Figure on its normal savefig.
    captured: list[Figure] = []
    original = Figure.savefig

    def record(figure: Figure, *args: object, **kwargs: object) -> object:
        captured.append(figure)
        return original(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", record):
        figures(str(out), comparison_rho="1/2", comparison_observer="combinatorial")
    # THEN: Figure 1 names all three groups by their short labels, figures 2 and 3 name only the two
    # adopter groups (figure 3 through its panel headings), and no rendered text keeps the long form.
    assert len(captured) == 3
    rendered = [_rendered_text(figure) for figure in captured]
    assert {"Non-WithdrawChan", "Keep", "Drain"} <= rendered[0]
    assert {"Keep", "Drain"} <= rendered[1]
    assert {"Keep", "Drain"} <= rendered[2]
    long_form = ("Conventional withdrawals", "WithdrawChan · Keep remainder", "WithdrawChan · Drain")
    for texts in rendered:
        assert not any(name in text for text in texts for name in long_form)


# ==== the four-observer composite tables ====

#: In `best.csv`, each cell keeps the observer that actually won it; the four names below rotate so
#: no single observer column value can stand in for the combined row.
COMPOSITE_OBSERVERS = ("historical", "trained-logit", "combinatorial", "group")


def _composite(out: Path) -> None:
    """Write an output whose `best.csv` wins different cells for different observers.

    `groups.csv` holds the same eight figure-3 cells under one observer with deliberately different
    bounds (9.x bits), so a figure that reads the wrong table is visible in its values.  `paired.csv`
    holds the usual combinatorial rows plus four `observer="best"` composite rows.
    """
    (out / "summary").mkdir(parents=True, exist_ok=True)
    winners = {
        ("keep", "decstep"): (0, 5.0), ("keep", "popstep"): (1, 5.1),
        ("keep", "decsmooth"): (2, 6.0), ("keep", "popsmooth"): (3, 6.1),
        ("drain", "decstep"): (3, 4.0), ("drain", "popstep"): (2, 4.1),
        ("drain", "decsmooth"): (0, 5.0), ("drain", "popsmooth"): (1, 5.1),
    }
    best = [_best("0/1", "need", "double", "keep", "non", 1.0),
            _best("1/2", "decsmooth", "double", "keep", "non", 1.5)]
    best[0]["observer"], best[1]["observer"] = "historical", "trained-logit"
    for (group, rule), (observer, bits) in winners.items():
        row = _best("1/2", rule, "double", group, group, bits)
        row["observer"] = COMPOSITE_OBSERVERS[observer]
        best.append(row)
    with (out / "summary" / "best.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(best)
    groups = [_best("1/2", rule, "double", group, group, 9.0 + index / 10)
              for group in ("keep", "drain")
              for index, rule in enumerate(("decstep", "popstep", "decsmooth", "popsmooth"))]
    with (out / "summary" / "groups.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(groups)
    paired = []
    rates = {("keep", "decstep"): -0.11, ("keep", "decsmooth"): -0.12,
             ("drain", "decstep"): -0.21, ("drain", "decsmooth"): -0.22}
    for observer in ("combinatorial", "best"):
        for (group, rule), value in rates.items():
            row = dict.fromkeys(PAIRED_COLUMNS, "0")
            row.update({"observer": observer, "adopters": group, "rho": "1/2",
                        "contrast": "m_model", "minuend": "balance", "subtrahend": "double",
                        "fixed": rule, "relative_hit": "0.1", "relative_lo": "-0.1",
                        "relative_hi": "0.3", "relative_bits": repr(value),
                        "relative_bits_lo": repr(value - 0.01), "relative_bits_hi": repr(value + 0.01)})
            paired.append(row)
    with (out / "summary" / "paired.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(paired)


def _quiet_draw(series: object, bars: object, **kwargs: object) -> None:
    """Call :func:`figures` with both renderers patched, so only the sliced data is inspected."""
    with patch.object(lab.report.figures, "plot_figure1", series), \
            patch.object(lab.report.figures, "plot_bars", bars):
        figures(**kwargs)


def test_figure3_bars_combine_rows_without_an_observer_filter() -> None:
    # GIVEN: Eight figure-3 cells whose winning observer rotates between historical, trained-logit,
    # combinatorial and group.
    rows = [_best("1/2", rule, "double", group, group, 5.0 + index / 10)
            for group in ("keep", "drain")
            for index, rule in enumerate(("decstep", "popstep", "decsmooth", "popsmooth"))]
    for index, row in enumerate(rows):
        row["observer"] = COMPOSITE_OBSERVERS[index % len(COMPOSITE_OBSERVERS)]
    # WHEN: Reading the already combined rows, with no observer filter.
    bars = figure3_bars(rows, rho=Fraction(1, 2), observer=None, m_model="double")
    # THEN: Each cell comes from the row that won it, without rewriting its observer name.
    assert bars == {"Keep": {"Decimal ladder": [(5.0, 4.5, 5.5), (5.2, 4.7, 5.7)],
                             "Popular ladder": [(5.1, 4.6, 5.6), (5.3, 4.8, 5.8)]},
                    "Drain": {"Decimal ladder": [(5.0, 4.5, 5.5), (5.2, 4.7, 5.7)],
                              "Popular ladder": [(5.1, 4.6, 5.6), (5.3, 4.8, 5.8)]}}


def test_figure3_bars_reject_two_rows_for_one_combined_cell() -> None:
    # GIVEN: A combined table that lists the same cell twice under different winners.
    row = _best("1/2", "decstep", "double", "keep", "keep", 5.0)
    duplicate = dict(row, observer="group")
    # WHEN: Reading it as combined rows.
    # THEN: The ambiguity is refused instead of silently taking the first.
    with pytest.raises(ValueError, match="more than once"):
        figure3_bars([row, duplicate], rho=Fraction(1, 2), observer=None, m_model="double")


def test_figure2_bars_report_a_missing_comparison_slice() -> None:
    # GIVEN: Composite contrast rows for one behaviour only.
    rows = [{"contrast": "m_model", "minuend": "balance", "subtrahend": "double", "rho": "1/2",
             "fixed": rule, "adopters": "keep", "observer": "best", "relative_bits": "-0.1",
             "relative_bits_lo": "-0.2", "relative_bits_hi": "0.0"}
            for rule in ("decstep", "decsmooth")]
    # WHEN: Slicing the missing behaviour.
    # THEN: The missing key is named, not raised as a bare StopIteration.
    with pytest.raises(ValueError, match="drain"):
        figure2_bars(rows, rho=Fraction(1, 2), observer="best")


def test_figures_best_mode_reads_both_composite_tables(tmp_path: Path) -> None:
    # GIVEN: An output whose per-cell winners differ and whose paired table has composite rows.
    out = tmp_path / "out"
    _composite(out)
    drawn: list[object] = []

    def series(points: object, path: object) -> None:
        drawn.append(("fig1", points))

    def bars(panels: object, path: object, **kwargs: object) -> None:
        drawn.append(("fig2" if kwargs["relative"] else "fig3", panels))

    # WHEN: Drawing with the composite statistic selected for the comparison.
    _quiet_draw(series, bars, out=str(out), comparison_observer="best")
    # THEN: Figure 1 keeps the combined best table, figure 2 reads the `best` paired rows, and
    # figure 3 reads `best.csv`'s own per-cell winners rather than one observer's groups rows.
    fig1, fig2, fig3 = (payload for _name, payload in drawn)
    assert fig1["Keep"] == [(Fraction(1, 2), 6.0, 5.5, 6.5)]
    assert [tuple(round(value, 2) for value in bar) for bar in fig2[""]["Keep"]] == \
        [(-0.11, -0.12, -0.1), (-0.12, -0.13, -0.11)]
    assert [tuple(round(value, 2) for value in bar) for bar in fig2[""]["Drain"]] == \
        [(-0.21, -0.22, -0.2), (-0.22, -0.23, -0.21)]
    assert fig3["Keep"]["Decimal ladder"] == [(5.0, 4.5, 5.5), (6.0, 5.5, 6.5)]
    assert [tuple(round(value, 2) for value in bar)
            for bar in fig3["Drain"]["Popular ladder"]] == [(4.1, 3.6, 4.6), (5.1, 4.6, 5.6)]


def test_figures_best_mode_rejects_a_missing_composite_paired_row(tmp_path: Path) -> None:
    # GIVEN: An output whose paired table holds only the per-observer rows.
    out = tmp_path / "out"
    _composite(out)
    path = out / "summary" / "paired.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh) if row["observer"] != "best"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # WHEN: Asking for the composite comparison.
    # THEN: It is refused, naming the statistic, and no figure is half-written.
    with pytest.raises(ValueError, match="best"):
        figures(str(out), comparison_observer="best")
    assert sorted(p.name for p in (out / "figures").iterdir()) == []


def test_figures_best_mode_rejects_an_incomplete_paired_slice(tmp_path: Path) -> None:
    # GIVEN: Composite paired rows for the keep behaviour only.
    out = tmp_path / "out"
    _composite(out)
    path = out / "summary" / "paired.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh)
                if row["observer"] != "best" or row["adopters"] == "keep"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, PAIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # WHEN: Asking for the composite comparison at the sliced rate.
    # THEN: The missing behaviour is named and no figure is produced from a partial table.
    with pytest.raises(ValueError, match="drain"):
        figures(str(out), comparison_observer="best")
    assert sorted(p.name for p in (out / "figures").iterdir()) == []


def test_figures_best_mode_rejects_an_incomplete_figure3_slice(tmp_path: Path) -> None:
    # GIVEN: A combined best table missing one of figure 3's eight cells.
    out = tmp_path / "out"
    _composite(out)
    path = out / "summary" / "best.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.DictReader(fh)
                if not (row["group"] == "drain" and row["rule"] == "popsmooth")]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, GROUPS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    # WHEN: Asking for the composite figure 3.
    # THEN: The missing cell is named, instead of falling back to another observer's table.
    with pytest.raises(ValueError, match="popsmooth"):
        figures(str(out), comparison_observer="best")
    assert sorted(p.name for p in (out / "figures").iterdir()) == []


def test_figures_best_mode_keeps_the_winning_observer_names(tmp_path: Path) -> None:
    # GIVEN: A combined best table with a different winning observer per cell.
    out = tmp_path / "out"
    _composite(out)
    before = (out / "summary" / "best.csv").read_bytes()
    # WHEN: Drawing the composite figures.
    figures(str(out), comparison_observer="best")
    # THEN: The input rows still name their real winners; `best` is only the paired statistic's name.
    after = (out / "summary" / "best.csv").read_bytes()
    assert after == before
    with (out / "summary" / "best.csv").open(encoding="utf-8", newline="") as fh:
        observers = {row["observer"] for row in csv.DictReader(fh)}
    assert observers == set(COMPOSITE_OBSERVERS)
