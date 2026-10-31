"""Figures from an experiment's summary tables, written as PNG.

Notation local to this module; the tables are those of :mod:`lab.report.summary`.

- `S`: the source deposit of one withdrawal; `O`: what an observer sees of it.
- `H`: the plotted bound `-log2 hit` in bits, an upper bound on the conditional min-entropy
  `H_inf(S | O)`; `best.csv` takes it under the strongest observer of each cell.
- series: figure 1's point lines, one `Point` `(rho, estimate, lo, hi)` per point, in bits.
- bars: the two bar charts' columns, one `Bar` `(estimate, lo, hi)` each: figure 2's relative
  entropy-bound change, figure 3's absolute bound in bits.
"""

from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import Bbox, blended_transform_factory

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.text import Annotation

__all__ = ["FIGURE1_SERIES", "figure1_series", "figures", "plot_figure1"]

Point = tuple[Fraction, float, float, float]

#: Figure 1's lines: label, the run's `adopters` kind and the plotted `group`.
FIGURE1_SERIES: tuple[tuple[str, str, str], ...] = (
    ("Non-WithdrawChan", "keep", "non"),
    ("Keep", "keep", "keep"),
    ("Drain", "drain", "drain"),
)

# ==== style ====

# The look follows the FusionCharts "gammel" theme, measured in CSS pixels on a 600 x 400 chart and
# converted at 0.75 pt per pixel, so the figure keeps the chart's proportions at any dpi.
PX = 0.75
STYLE = {
    "figsize": (600 / 96, 400 / 96),
    "dpi": 300,
    # Series colours: the main one is the primary hue; the error bars use the theme's dark grey.
    "colors": ("#434348", "#7cb5ec", "#f7a35c"),
    "errorbar_color": "#434348",
    "errorbar_alpha": 0.5,
    "grid_color": "#e6e6e6",
    "label_color": "#666666",
    "legend_color": "#333333",
    "minor_tick_color": "#cccccc",
    "line_width": 2 * PX,
    # Dash and gap of 3 and 6 px, in units of the line width.
    "dashes": (1.5, 3.0),
    "errorbar_width": 2 * PX,
    "capsize": 4 * PX,
    "marker_size": 8 * PX,
    "marker_edge_width": 1 * PX,
    "legend_marker_size": 12 * PX,
    "tick_size": 11 * PX,
    "title_size": 12 * PX,
    "point_label_size": 11 * PX * 0.8,
    "min_label_gap": 0.1,
}

# The platform's UI sans-serif: Segoe UI on Windows, Helvetica Neue on macOS, Noto Sans on most Linux
# desktops; DejaVu Sans, bundled with matplotlib, closes the list.
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Noto Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
}

# The unbounded-side mark: U+221E is missing from Noto Sans, the platform's UI face on Linux, so the
# mark asks for DejaVu Sans, which matplotlib bundles and which covers the glyph on every platform.
INFINITY_FONT = "DejaVu Sans"

# Where a point's two-decimal label may sit, in points from the marker: first above, then below,
# then further out, then to either side.  The fixed order keeps the drawing deterministic and
# staggers the crowded points at 0, 1% and 2.5% adoption.  The last rows are only reached where a
# long confidence interval leaves no room above or below the marker.
POINT_LABEL_OFFSETS = ((0.0, 8.0), (0.0, -14.0), (0.0, 22.0), (0.0, -28.0),
                       (10.0, 8.0), (-10.0, 8.0), (10.0, -14.0), (-10.0, -14.0),
                       (16.0, 8.0), (-16.0, 8.0), (16.0, -14.0), (-16.0, -14.0),
                       (0.0, 34.0), (0.0, -40.0))

Y_LABEL = "$\\overline{H}_\\infty(S \\mid O)$\nconditional min-entropy (bits)"
X_LABEL = "$\\rho$\nadoption rate"

# ==== slices ====


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _cell(best: list[dict[str, str]], rho: Fraction, rule: str, m_model: str, adopters: str,
          group: str) -> dict[str, str] | None:
    """The `best` row of one cell, or `None`; `rho = 0` reads the `need` rule under `double`."""
    wanted = ("need", "double") if rho == 0 else (rule, m_model)
    for row in best:
        if Fraction(row["rho"]) == rho and (row["rule"], row["m_model"], row["adopters"], row["group"]) == (
                *wanted, adopters, group):
            return row
    return None


def figure1_series(best: list[dict[str, str]], *, rule: str, m_model: str) -> dict[str, list[Point]]:
    """Figure 1's lines from `best.csv` rows, or from one observer's `groups.csv` rows.

    Every `rho > 0` point uses `rule` and `m_model`; the `rho = 0` point, which only the
    non-adopters have, uses the `need` rule under `double`.  Points are sorted by `rho`; a `rho`
    whose cell is missing from the rows is skipped.

    Args:
        best: Rows of `best.csv`, or one observer's rows of `groups.csv`; both carry these columns.
        rule: The rounding rule of every `rho > 0` point.
        m_model: The maximum model of every `rho > 0` point.

    Returns:
        One list per label of :data:`FIGURE1_SERIES`, each of `Point` `(rho, bits, bits_lo, bits_hi)`.
    """
    rhos = sorted({Fraction(row["rho"]) for row in best})
    series: dict[str, list[Point]] = {}
    for label, adopters, group in FIGURE1_SERIES:
        rows = [(rho, _cell(best, rho, rule, m_model, adopters, group)) for rho in rhos]
        series[label] = [(rho, float(row["bits"]), float(row["bits_lo"]), float(row["bits_hi"]))
                         for rho, row in rows if row is not None]
    return series


Bar = tuple[float, float, float]


def _one(rows: list[dict[str, str]], predicate: Callable[[dict[str, str]], bool],
         description: str) -> dict[str, str]:
    """The single row matching `predicate`; refuse a missing or duplicated cell by name.

    The comparison tables carry one row per sliced cell.  Picking the first of several would hide a
    duplicated composite row, and a bare `StopIteration` would not say which cell failed, so both
    cases raise a `ValueError` naming the description.
    """
    found = [row for row in rows if predicate(row)]
    if not found:
        msg = f"{description} is missing"
        raise ValueError(msg)
    if len(found) > 1:
        msg = f"{description} appears more than once"
        raise ValueError(msg)
    return found[0]


def figure2_bars(paired: list[dict[str, str]], *, rho: Fraction,
                 observer: str) -> dict[str, list[Bar]]:
    """Step and random-walk relative entropy-bound changes at fixed adoption, by user behaviour.

    `observer="best"` reads the composite statistic's rows, which pair the highest pooled hit rate
    of each arm; the rows keep every real observer's name elsewhere in the table.

    Args:
        paired: Rows of `paired.csv`.
        rho: The adoption rate every contrasting row must name.
        observer: The observer every contrasting row must name.

    Returns:
        The two adopter behaviours of :data:`FIGURE1_SERIES`, each the step then random-walk `Bar`
        `(relative_bits, relative_bits_lo, relative_bits_hi)` of the balance-minus-double contrast.
    """
    bars: dict[str, list[Bar]] = {}
    for label, adopters, _group in FIGURE1_SERIES[1:]:
        values = []
        for rule in ("decstep", "decsmooth"):
            row = _one(paired, lambda row, rule=rule, adopters=adopters:
                       Fraction(row["rho"]) == rho and
                       (row["contrast"], row["minuend"], row["subtrahend"], row["fixed"],
                        row["adopters"], row["observer"]) ==
                       ("m_model", "balance", "double", rule, adopters, observer),
                       f"paired.csv observer {observer!r} {adopters!r} {rule!r} row at rho {rho}")
            values.append(tuple(float(row[key]) for key in ("relative_bits", "relative_bits_lo",
                                                            "relative_bits_hi")))
        bars[label] = values
    return bars


def figure3_bars(groups: list[dict[str, str]], *, rho: Fraction, observer: str | None,
                 m_model: str) -> dict[str, dict[str, list[Bar]]]:
    """Absolute observer-specific entropy bounds for both ladders, by behaviour and step/smooth rule.

    `observer=None` reads rows that are already one combined row per cell, such as the composite
    `best.csv`; each cell then keeps whichever observer won it, so no observer filter applies.

    Args:
        groups: Rows of `groups.csv`, or the combined rows of `best.csv`.
        rho: The adoption rate every row must name.
        observer: The observer every row must name, or `None` to accept each cell's own winner.
        m_model: The maximum model every row must use.

    Returns:
        The two adopter behaviours of :data:`FIGURE1_SERIES`, each ladder's name mapped to the step
        then random-walk `Bar` `(bits, bits_lo, bits_hi)`.
    """
    panels: dict[str, dict[str, list[Bar]]] = {}
    for label, adopters, group in FIGURE1_SERIES[1:]:
        panels[label] = {}
        for ladder, rules in (("Decimal ladder", ("decstep", "decsmooth")),
                              ("Popular ladder", ("popstep", "popsmooth"))):
            values = []
            for rule in rules:
                row = _one(groups, lambda row, rule=rule, adopters=adopters, group=group:
                           Fraction(row["rho"]) == rho and
                           (row["rule"], row["m_model"], row["adopters"], row["group"]) ==
                           (rule, m_model, adopters, group) and
                           (observer is None or row["observer"] == observer),
                           f"figure 3 observer {observer!r} {adopters!r} {rule!r} cell at rho {rho}"
                           f" under m_model {m_model!r}")
                values.append(tuple(float(row[key]) for key in ("bits", "bits_lo", "bits_hi")))
            panels[label][ladder] = values
    return panels


# ==== plots ====


def _point_label(axes: Axes, rho: Fraction, bits: float, dx: float, dy: float) -> Annotation:
    """One two-decimal estimate, anchored at its own point and offset by `(dx, dy)` points."""
    label = axes.annotate(f"{bits:.2f}", xy=(float(rho), bits), xytext=(dx, dy),
                          textcoords="offset points",
                          ha="center" if dx == 0 else ("left" if dx > 0 else "right"),
                          va="bottom" if dy > 0 else "top",
                          fontsize=STYLE["point_label_size"], color=STYLE["label_color"], zorder=6)
    # Labels are annotations on the data, not layout content: counting them would shrink the axes
    # (the dense left-hand labels sit past the axis edge) and move every point under its label.
    label.set_in_layout(False)
    return label


def _interval_boxes(axes: Axes, series: dict[str, list[Point]], pixels: float) -> list[Bbox]:
    """The displayed area of every finite interval side: its whisker, and the cap across its end.

    Each drawn side runs from the estimate to the bound and ends in a cap `capsize` wide, so a label
    must clear the cap's full width, not just the centre line of the whisker.
    """
    half_cap = STYLE["capsize"] * pixels
    half_line = max(STYLE["errorbar_width"] * pixels, 1.0) / 2
    boxes: list[Bbox] = []
    for points in series.values():
        for rho, bits, lo, hi in points:
            x, y = axes.transData.transform((float(rho), bits))
            for bound in (lo, hi):
                if not math.isfinite(bound):
                    continue
                _x, edge = axes.transData.transform((float(rho), bound))
                boxes.append(Bbox.from_extents(x - half_cap, min(y, edge - half_line),
                                               x + half_cap, max(y, edge + half_line)))
    return boxes


def _label_points(axes: Axes, series: dict[str, list[Point]]) -> None:
    """Write each estimate, two decimals, at its point in the first free :data:`POINT_LABEL_OFFSETS`.

    Every real point gets exactly one label, even when two estimates round alike: the text keeps the
    point's `(rho, bits)`, so it can be paired with the data instead of deduplicated.  A candidate is
    kept only where its rendered box stays inside the axes and touches no earlier label, marker,
    confidence whisker or cap, infinity mark or the legend; the offset order is fixed, so the layout
    is deterministic.  The whiskers bound where a label may sit: a long interval pushes its own
    point's label out to the side rather than letting the bar cross the digits.
    """
    figure = axes.figure
    figure.set_dpi(STYLE["dpi"])
    FigureCanvasAgg(figure)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    area = axes.get_window_extent(renderer)
    pixels = figure.dpi / 72
    radius = STYLE["marker_size"] / 2 * pixels
    occupied: list[Bbox] = []
    for points in series.values():
        for rho, bits, _lo, _hi in points:
            x, y = axes.transData.transform((float(rho), bits))
            occupied.append(Bbox.from_extents(x - radius, y - radius, x + radius, y + radius))
    occupied += _interval_boxes(axes, series, pixels)
    occupied += [text.get_window_extent(renderer) for text in axes.texts if text.get_text() == "∞"]
    legend = axes.get_legend()
    if legend is not None:
        occupied.append(legend.get_window_extent(renderer))
    for points in series.values():
        for rho, bits, _lo, _hi in points:
            for dx, dy in POINT_LABEL_OFFSETS:
                label = _point_label(axes, rho, bits, dx, dy)
                box = label.get_window_extent(renderer).expanded(1.05, 1.1)
                fits = (box.x0 >= area.x0 + 1 and box.x1 <= area.x1 - 1
                        and box.y0 >= area.y0 + 1 and box.y1 <= area.y1 - 1
                        and not any(box.overlaps(taken) for taken in occupied))
                if fits:
                    occupied.append(box)
                    break
                label.remove()
            else:
                # Every candidate touched something; still label the point rather than drop it.
                _point_label(axes, rho, bits, *POINT_LABEL_OFFSETS[0])


def plot_figure1(series: dict[str, list[Point]], path: Path) -> None:
    """Draw figure 1 into `path`: one line per series with 95% interval bars, against `rho`.

    Every point's estimate is annotated with two decimals at the point itself; crowded points move
    their label, never their marker.

    Args:
        series: The lines to draw, `Point` lists keyed by their legend label, as
            :func:`figure1_series` returns them.
        path: The PNG file to write; drawn on a fresh `Figure`, so nothing in `series` is mutated.
    """
    for points in series.values():
        for rho, bits, _lo, _hi in points:
            if not math.isfinite(bits):
                msg = f"figure 1 estimate at rho {rho} is not finite"
                raise ValueError(msg)
    if not any(series.values()):
        msg = "figure 1 has no data for the selected rule and maximum model"
        raise ValueError(msg)
    with rc_context(RC):
        figure = Figure(figsize=STYLE["figsize"], layout="constrained")
        axes = figure.add_subplot()
        rhos = sorted({point[0] for points in series.values() for point in points})
        handles = []
        unbounded: list[tuple[float, float, bool]] = []
        for (label, points), color in zip(series.items(), STYLE["colors"], strict=True):
            x = [float(rho) for rho, _bits, _lo, _hi in points]
            y = [bits for _rho, bits, _lo, _hi in points]
            below = [bits - lo for _rho, bits, lo, _hi in points]
            above = [hi - bits for _rho, bits, _lo, hi in points]
            axes.plot(x, y, color=color, linewidth=STYLE["line_width"], linestyle=(0, STYLE["dashes"]),
                      dash_capstyle="round", zorder=2)
            axes.plot(x, y, linestyle="none", marker="o", markersize=STYLE["marker_size"],
                      markerfacecolor="white", markeredgecolor=color,
                      markeredgewidth=STYLE["marker_edge_width"], zorder=3)
            # Above the markers: an interval shorter than a marker would otherwise vanish under it.
            closed = [index for index in range(len(points))
                      if math.isfinite(points[index][2]) and math.isfinite(points[index][3])]
            if len(closed) == len(points):
                axes.errorbar(x, y, yerr=(below, above), fmt="none", ecolor=STYLE["errorbar_color"],
                              alpha=STYLE["errorbar_alpha"], elinewidth=STYLE["errorbar_width"],
                              capsize=STYLE["capsize"], capthick=STYLE["errorbar_width"], zorder=4)
            else:
                # A bound at infinity is an open side: keep the defined side and the point, and mark
                # the open direction instead of passing an infinite bar to the renderer.
                if closed:
                    axes.errorbar([x[index] for index in closed], [y[index] for index in closed],
                                  yerr=([below[index] for index in closed],
                                        [above[index] for index in closed]),
                                  fmt="none", ecolor=STYLE["errorbar_color"],
                                  alpha=STYLE["errorbar_alpha"], elinewidth=STYLE["errorbar_width"],
                                  capsize=STYLE["capsize"], capthick=STYLE["errorbar_width"], zorder=4)
                for index in range(len(points)):
                    lo, hi = points[index][2], points[index][3]
                    if math.isfinite(lo) and math.isfinite(hi):
                        continue
                    if math.isfinite(lo):
                        axes.errorbar([x[index]], [y[index]], yerr=([below[index]], [0.0]), fmt="none",
                                      ecolor=STYLE["errorbar_color"], alpha=STYLE["errorbar_alpha"],
                                      elinewidth=STYLE["errorbar_width"], capsize=STYLE["capsize"],
                                      capthick=STYLE["errorbar_width"], zorder=1)
                    if math.isfinite(hi):
                        axes.errorbar([x[index]], [y[index]], yerr=([0.0], [above[index]]), fmt="none",
                                      ecolor=STYLE["errorbar_color"], alpha=STYLE["errorbar_alpha"],
                                      elinewidth=STYLE["errorbar_width"], capsize=STYLE["capsize"],
                                      capthick=STYLE["errorbar_width"], zorder=1)
                    if math.isinf(hi):
                        unbounded.append((x[index], y[index], True))
                    elif math.isinf(lo):
                        unbounded.append((x[index], y[index], False))
            handles.append(Line2D([], [], linestyle="none", marker="o", markersize=STYLE["legend_marker_size"],
                                  markerfacecolor="white", markeredgecolor=color,
                                  markeredgewidth=STYLE["marker_edge_width"], label=label))
        labelled = _spread(rhos, STYLE["min_label_gap"])
        axes.set_xticks([float(rho) for rho in labelled], [f"{float(rho) * 100:.2f}%" for rho in labelled])
        axes.set_xticks([float(rho) for rho in rhos if rho not in labelled], minor=True)
        top = max([hi for points in series.values() for _rho, _bits, _lo, hi in points
                   if math.isfinite(hi)] +
                  [bits for points in series.values() for _rho, bits, _lo, _hi in points
                   if math.isfinite(bits)])
        axes.set_ylim(0.0, math.ceil(top * 1.1) if unbounded else math.ceil(top))
        step = axes.get_ylim()[1] * 0.05
        for x_value, y_value, upward in unbounded:
            tip = y_value + step * (1.2 if upward else -1.2)
            axes.annotate("", xy=(x_value, tip), xytext=(x_value, y_value),
                          arrowprops={"arrowstyle": "-|>", "color": STYLE["errorbar_color"],
                                      "alpha": STYLE["errorbar_alpha"],
                                      "linewidth": STYLE["errorbar_width"]}, zorder=5)
            axes.annotate("∞", xy=(x_value, tip), xytext=(6, 2 if upward else -2),
                          textcoords="offset points", ha="left",
                          va="bottom" if upward else "top", fontsize=STYLE["tick_size"] * 0.8,
                          fontfamily=INFINITY_FONT,
                          color=STYLE["label_color"], zorder=6)
        axes.grid(axis="y", color=STYLE["grid_color"], linewidth=1 * PX)
        axes.set_axisbelow(True)
        axes.tick_params(which="major", length=0, labelsize=STYLE["tick_size"], labelcolor=STYLE["label_color"])
        axes.tick_params(axis="x", which="minor", length=3, color=STYLE["minor_tick_color"])
        axes.set_xlabel(X_LABEL, fontsize=STYLE["title_size"], color=STYLE["label_color"])
        axes.set_ylabel(Y_LABEL, fontsize=STYLE["title_size"], color=STYLE["label_color"])
        # The legend sits at the right edge, midway between the highest and lowest line that reach it.
        ends = [points[-1][1] for points in series.values() if points and points[-1][0] == rhos[-1]]
        axes.legend(handles=handles, loc="center right",
                    bbox_to_anchor=(1.0, (max(ends) + min(ends)) / 2 if ends else 0.5),
                    bbox_transform=blended_transform_factory(axes.transAxes, axes.transData), frameon=False,
                    prop={"size": STYLE["title_size"]}, labelcolor=STYLE["legend_color"], handletextpad=0.3)
        _label_points(axes, series)
        figure.savefig(path, dpi=STYLE["dpi"], metadata={"Software": None})


def plot_bars(panels: dict[str, dict[str, list[Bar]]], path: Path, *, relative: bool,
              colors: tuple[tuple[str, str], ...]) -> Figure:
    """Grouped step/random-walk columns with interval bars and shared panel scales.

    Args:
        panels: One panel per heading, each series label mapped to its two `Bar` `(estimate, lo, hi)`
            columns, as :func:`figure2_bars` and :func:`figure3_bars` return them.
        path: The PNG file to write.
        relative: Show percentage changes on an axis including zero (`True`), or absolute
            bit bounds with a zero lower limit (`False`). Relative limits need not be symmetric.
        colors: One series colour tuple per panel, aligned with `panels`; the strict `zip` refuses a
            count that does not match.

    Returns:
        The composed `Figure`, also written to `path`; nothing in `panels` is mutated.
    """
    values = [bar for series in panels.values() for bars in series.values() for bar in bars]
    estimates = [value for value, _lo, _hi in values if math.isfinite(value)]
    lowers = [lo for _value, lo, _hi in values if math.isfinite(lo)]
    uppers = [hi for _value, _lo, hi in values if math.isfinite(hi)]
    lower = min(lowers + estimates, default=0.0)
    upper = max(uppers + estimates, default=1.0)
    span = max(upper - min(lower, 0), 0.01)
    bottom = min(lower, 0) - span * 0.2 if relative else 0
    top = max(upper, 0) + span * (0.35 if relative else 0.25)
    with rc_context(RC):
        figure = Figure(figsize=(STYLE["figsize"][0] * len(panels), STYLE["figsize"][1]),
                        layout="constrained")
        for index, ((heading, series), palette) in enumerate(zip(panels.items(), colors, strict=True)):
            axes = figure.add_subplot(1, len(panels), index + 1)
            has_estimates = False
            for offset, ((label, bars), color) in enumerate(zip(series.items(), palette, strict=True)):
                x = [category + (offset - 0.5) * 0.32 for category in range(2)]
                columns = [(location, value) for location, (value, _lo, _hi) in zip(x, bars, strict=True)
                           if math.isfinite(value)]
                if columns:
                    has_estimates = True
                    axes.bar([location for location, _value in columns],
                             [value for _location, value in columns],
                             width=0.28, color=color, label=label, zorder=2)
                intervals = [(location, lo, hi) for location, (_value, lo, hi) in zip(x, bars, strict=True)
                             if math.isfinite(lo) and math.isfinite(hi)]
                if intervals:
                    axes.errorbar([location for location, _lo, _hi in intervals],
                                  [(lo + hi) / 2 for _location, lo, hi in intervals],
                                  yerr=[(hi - lo) / 2 for _location, lo, hi in intervals], fmt="none",
                                  ecolor=STYLE["errorbar_color"], alpha=STYLE["errorbar_alpha"],
                                  elinewidth=STYLE["errorbar_width"], capsize=STYLE["capsize"],
                                  capthick=STYLE["errorbar_width"], zorder=3)
                for location, (value, lo, hi) in zip(x, bars, strict=True):
                    if not math.isfinite(value):
                        continue
                    anchor = lo if relative and value < 0 else hi
                    if not math.isfinite(anchor):
                        anchor = value
                    distance = -span * 0.025 if relative and value < 0 else span * 0.025
                    axes.text(location, anchor + distance, f"{value:.2%}" if relative else f"{value:.2f}",
                              ha="center", va="top" if distance < 0 else "bottom",
                              fontsize=STYLE["tick_size"], color=STYLE["label_color"])
            if not has_estimates:
                axes.text(0.5, 0.5, "Estimate unavailable", transform=axes.transAxes,
                          ha="center", va="center", fontsize=STYLE["tick_size"], color=STYLE["label_color"])
            axes.set_xticks((0, 1), ("Step", "Random walk"))
            axes.set_xlim(-0.6, 1.6)
            axes.set_ylim(bottom, top)
            if relative:
                axes.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                axes.axhline(0, color=STYLE["minor_tick_color"], linewidth=STYLE["line_width"])
            axes.grid(axis="y", color=STYLE["grid_color"], linewidth=PX)
            axes.set_axisbelow(True)
            axes.tick_params(length=0, labelsize=STYLE["tick_size"], labelcolor=STYLE["label_color"])
            axes.set_ylabel("relative change in entropy bound (%)" if relative else Y_LABEL,
                            fontsize=STYLE["title_size"], color=STYLE["label_color"])
            if heading:
                axes.text(0.02, 0.98, heading, transform=axes.transAxes, va="top",
                          fontsize=STYLE["title_size"], color=STYLE["label_color"])
            handles, _labels = axes.get_legend_handles_labels()
            if handles:
                axes.legend(loc="upper right", frameon=False, prop={"size": STYLE["title_size"]},
                            labelcolor=STYLE["legend_color"])
        figure.savefig(path, dpi=STYLE["dpi"], metadata={"Software": None})
    return figure


def _spread(rhos: list[Fraction], gap: float) -> list[Fraction]:
    """The `rhos` that get a tick label: from the top down, each more than `gap` below the last.

    On a linear axis the low ratios sit closer than a label is wide; the skipped ones keep a minor tick.
    """
    labelled: list[Fraction] = []
    for rho in sorted(rhos, reverse=True):
        if not labelled or labelled[-1] - rho > gap:
            labelled.append(rho)
    return sorted(labelled)


# ==== output ====


def figures(out: str, *, rule: str = "decsmooth", m_model: str = "double",
            comparison_rho: str = "1/2", comparison_observer: str = "combinatorial",
            figure1_observer: str | None = None) -> dict[str, str]:
    """Write the figures of an experiment output into `<out>/figures`.

    Reads `groups.csv` and `paired.csv`, and `best.csv` for figure 1 unless `figure1_observer`
    names another observer.  With `comparison_observer="best"`, figure 2 reads the composite
    `observer="best"` rows of `paired.csv` and figure 3 reads `best.csv` itself, whose cells may be
    won by different real observers; `best` is the statistic's marker, never a rewritten winner.
    Figure 1 uses `rule` and `m_model`; figure 2 shows relative entropy-bound changes between
    maximum models. Figure 3 shows both ladders' absolute bounds under `m_model`. Both comparison
    bar charts use `comparison_rho` and `comparison_observer`.

    Every slice is resolved before the first PNG is written, so a missing composite row or cell
    raises `ValueError` instead of leaving a partial or empty chart behind.

    Args:
        out: An experiment output; its `summary` folder holds the three tables.
        rule: The rounding rule figure 1 plots.
        m_model: The maximum model figures 1 and 3 use.
        comparison_rho: The adoption rate both bar charts slice at.
        comparison_observer: The observer both bar charts slice at, or `best` for the composite
            statistic.
        figure1_observer: The observer figure 1's rows must name in `groups.csv`; `None` reads the
            strongest-observer envelope from `best.csv` instead.

    Returns:
        A mapping from `fig1`, `fig2` and `fig3` to the written PNG paths under `<out>/figures`.
    """
    source = Path(out)
    folder = source / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    best: list[dict[str, str]] | None = None
    groups: list[dict[str, str]] | None = None
    if figure1_observer is None:
        best = _read(source / "summary" / "best.csv")
        figure1_rows = best
    else:
        groups = _read(source / "summary" / "groups.csv")
        figure1_rows = [row for row in groups if row["observer"] == figure1_observer]
        if not figure1_rows:
            msg = (f"figure 1 observer {figure1_observer!r} is not in groups.csv "
                   f"(available: {sorted({row['observer'] for row in groups})})")
            raise ValueError(msg)
    series = figure1_series(figure1_rows, rule=rule, m_model=m_model)
    comparison = Fraction(comparison_rho)
    paired = _read(source / "summary" / "paired.csv")
    composite = comparison_observer == "best"
    if composite and not any(row["observer"] == "best" for row in paired):
        msg = ("paired.csv has no observer 'best' composite rows; summarize with the explicit "
               "observer candidates first")
        raise ValueError(msg)
    fig2_panels = {"": figure2_bars(paired, rho=comparison, observer=comparison_observer)}
    if composite:
        if best is None:
            best = _read(source / "summary" / "best.csv")
        # Figure 3's cells may each be won by another observer, so its combined rows carry no
        # single observer value to filter on.
        fig3_panels = figure3_bars(best, rho=comparison, observer=None, m_model=m_model)
    else:
        if groups is None:
            groups = _read(source / "summary" / "groups.csv")
        fig3_panels = figure3_bars(groups, rho=comparison, observer=comparison_observer,
                                   m_model=m_model)
    fig1 = folder / "fig1.png"
    plot_figure1(series, fig1)
    fig2 = folder / "fig2.png"
    plot_bars(fig2_panels, fig2, relative=True, colors=(STYLE["colors"][1:],))
    fig3 = folder / "fig3.png"
    plot_bars(fig3_panels, fig3, relative=False,
              colors=(("#7cb5ec", "#bddaf5"), ("#f7a35c", "#fbd1ad")))
    return {"fig1": str(fig1), "fig2": str(fig2), "fig3": str(fig3)}
