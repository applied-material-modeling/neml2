# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Render one log-log timing+fit figure per scenario.

For every scenario found across the supplied folders, plot:

* the measured ``median_ms`` per batch size as scatter points with
  asymmetric error bars from ``min_ms`` / ``max_ms`` (v2 sweeps that
  don't carry per-batch spread degenerate to plain scatter -- no bars);
* the fitted scaling curve ``t(N) = t* * (max(N - N*, 0) / N* + 1)``
  using ``(t*, N*)`` from the folder's ``fitting.csv`` (so we replay
  the SAME eq 7.1 fit ``benchmark.sweep fit`` already wrote, with no
  re-fit at plot time);
* a side-panel table summarising each folder's
  ``t*`` (asymptotic per-call wall time below the elbow, ms),
  ``N*`` (batch at the elbow), and
  optimal throughput ``N*/t*`` (batches/ms, the asymptotic ceiling --
  also the value reached at the elbow, since beyond it doubling N
  doubles t).

Config format (JSON, either a bare list or ``{"folders": [...]}``;
unknown top-level keys like ``comment`` are ignored):

    {
      "folders": [
        {
          "path": "benchmark/results/20260608T040019Z_cpu_float64_eager",
          "label": "v2 CPU (eager)",
          "kwargs": {"linestyle": "--", "marker": "x", "color": "tab:gray"}
        },
        {
          "path": "benchmark/results/20260610T193958Z_cpu_float64_aoti",
          "label": "v3 CPU (AOTI)",
          "kwargs": {"linestyle": "-",  "marker": "o", "color": "tab:blue"}
        }
      ]
    }

``kwargs`` are forwarded to BOTH the scatter ``errorbar`` and the fit
``plot``; the script pins ``linestyle="none"`` on the scatter and
``marker=None`` on the fit so a user-supplied ``marker`` lands on the
scatter and a user-supplied ``linestyle`` lands on the line. Anything
else (``color``, ``linewidth``, ``alpha``, ...) flows through unchanged
so the two visually belong together.

Usage:

    python -m benchmark.plot_throughput \\
        --config benchmark/plot_throughput.json \\
        --output-dir plots/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# Scenarios are discovered by globbing ``<folder>/*.csv``; these aren't
# per-scenario timing files (fitting.csv = the per-scenario eq 7.1 fits;
# summary.csv = aggregated medians) and must be excluded.
_NON_SCENARIO_CSVS = frozenset({"fitting", "summary"})


def _discover_scenarios(folder: Path) -> list[str]:
    """Return sorted scenario names (CSV stems) for one result folder."""
    return sorted(
        p.stem
        for p in folder.glob("*.csv")
        if p.stem not in _NON_SCENARIO_CSVS and not p.stem.endswith(".runs")
    )


def _read_scenario_csv(path: Path) -> list[tuple[int, float, float, float]]:
    """Parse a scenario CSV into ``[(nbatch, median_ms, min_ms, max_ms), ...]``.

    ``median_ms`` matches the fit's loss target so the scatter and the
    curve are on the same y. ``min_ms`` / ``max_ms`` are returned as NaN
    for v2 sweeps that didn't record per-batch spread; the caller skips
    error bars on a NaN spread.
    """
    rows: list[tuple[int, float, float, float]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                b = int(r["nbatch"])
                med = float(r["median_ms"])
            except (ValueError, KeyError, TypeError):
                continue
            if not math.isfinite(med) or med <= 0:
                continue
            try:
                lo = float(r.get("min_ms", "nan"))
            except ValueError:
                lo = float("nan")
            try:
                hi = float(r.get("max_ms", "nan"))
            except ValueError:
                hi = float("nan")
            rows.append((b, med, lo, hi))
    rows.sort()
    return rows


def _read_fits(folder: Path) -> dict[str, tuple[float, float]]:
    """Return ``{scenario: (t0_ms, N_star)}`` from the folder's fitting.csv.

    Missing file / row / NaN fit values are silently treated as "no fit
    for this scenario" -- the plot then just shows the scatter.
    """
    fits: dict[str, tuple[float, float]] = {}
    path = folder / "fitting.csv"
    if not path.exists():
        return fits
    with path.open() as f:
        for r in csv.DictReader(f):
            try:
                t0 = float(r["t0_ms"])
                ns = float(r["N_star"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (math.isfinite(t0) and math.isfinite(ns) and t0 > 0 and ns > 0):
                continue
            fits[r["scenario"]] = (t0, ns)
    return fits


def _has_stats(rows: list[tuple[int, float, float, float]]) -> bool:
    """True iff at least one row has finite min/max we can plot as a bar."""
    return any(math.isfinite(lo) and math.isfinite(hi) for _, _, lo, hi in rows)


def _fit_curve(t0: float, n_star: float, x_min: float, x_max: float, n: int = 200):
    """Sample the piecewise-linear fit on a log-spaced N grid for plotting."""
    import numpy as np  # noqa: PLC0415

    grid = np.geomspace(max(x_min, 1.0), max(x_max, x_min + 1), num=n)
    return grid, t0 * (np.maximum(grid - n_star, 0.0) / n_star + 1.0)


def plot_throughput(
    specs: list[dict[str, Any]],
    output_dir: Path,
    *,
    fmt: str = "png",
    dpi: int = 120,
    figsize: tuple[float, float] = (11.0, 4.5),
) -> list[Path]:
    """Render one log-log scatter+fit figure per scenario; return written paths.

    Each spec contributes (at most) one scatter + one fit-line per
    scenario plus one row in the right-side throughput table. Folders
    that don't cover a given scenario are silently skipped for that
    figure. Folders without a fitting.csv entry for a scenario draw
    scatter only and appear in the table with ``--`` for N*/t*.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)

    folder_paths: list[Path] = []
    for spec in specs:
        p = Path(spec["path"])
        if not p.is_dir():
            raise FileNotFoundError(f"benchmark folder does not exist: {p}")
        folder_paths.append(p)

    folder_fits = [_read_fits(p) for p in folder_paths]

    all_scenarios = sorted(
        {scenario for folder in folder_paths for scenario in _discover_scenarios(folder)}
    )
    if not all_scenarios:
        print("No scenario CSVs found in any folder; nothing to plot.", file=sys.stderr)
        return []

    written: list[Path] = []
    for scenario in all_scenarios:
        # gridspec splits horizontally: main plot on the left, throughput
        # table on the right. The table panel hides its own axes so
        # matplotlib.table renders against a clean canvas. The 1.6:1
        # width ratio is empirically wide enough for "v3 CUDA (AOTI)"-
        # length labels alongside four numeric columns without clipping.
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.3], wspace=0.05)
        ax = fig.add_subplot(gs[0])
        tax = fig.add_subplot(gs[1])
        tax.axis("off")

        table_rows: list[list[str]] = []
        table_colors: list[Any] = []  # matplotlib color spec (str | tuple)

        plotted_any = False
        x_min, x_max = math.inf, -math.inf
        for spec, folder, fits in zip(specs, folder_paths, folder_fits, strict=True):
            csv_path = folder / f"{scenario}.csv"
            if not csv_path.exists():
                continue
            rows = _read_scenario_csv(csv_path)
            if not rows:
                continue
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            x_min = min(x_min, min(xs))
            x_max = max(x_max, max(xs))

            label = spec.get("label", str(folder))
            user_kw = dict(spec.get("kwargs", {}))
            color = user_kw.get("color")

            # ---- scatter (with err bars when stats are present) ----
            # Force linestyle='none' so a user-supplied `--`/`-` lands
            # on the fit line below, not as a connect-the-dots through
            # the scatter. Marker stays whatever the user picked.
            scatter_kw = dict(user_kw)
            scatter_kw["linestyle"] = "none"
            if _has_stats(rows):
                lower = [
                    (y - lo) if math.isfinite(lo) else 0.0 for y, lo in ((r[1], r[2]) for r in rows)
                ]
                upper = [
                    (hi - y) if math.isfinite(hi) else 0.0 for y, hi in ((r[1], r[3]) for r in rows)
                ]
                container = ax.errorbar(
                    xs, ys, yerr=[lower, upper], label=label, capsize=3, **scatter_kw
                )
            else:
                container = ax.errorbar(xs, ys, label=label, **scatter_kw)
            # Lock in the assigned color so the matching fit line and
            # table row paint the same hue. Container's `lines[0]` is
            # the data-line artist; its color is the resolved auto color
            # when the user didn't pin one.
            if color is None:
                color = container.lines[0].get_color()

            plotted_any = True

            # ---- fit line (skip if no fit available) ----
            fit = fits.get(scenario)
            if fit is not None:
                t0, ns = fit
                # Extend the curve at least to the data range; if the
                # elbow is beyond the largest measured batch the fit was
                # already extrapolating, so giving the line an extra
                # octave keeps the elbow visible.
                grid, curve = _fit_curve(t0, ns, x_min, max(x_max, ns * 2.0))
                line_kw = dict(user_kw)
                line_kw["color"] = color
                line_kw["marker"] = None
                line_kw.setdefault("linestyle", "-")
                ax.plot(grid, curve, **line_kw)
                # Asymptotic throughput beyond the elbow == value at the
                # elbow; the piecewise model has zero curvature past N*.
                tput = ns / t0
                table_rows.append([label, f"{t0:.2f}", f"{ns:.1f}", f"{tput:.3g}"])
            else:
                table_rows.append([label, "--", "--", "--"])
            table_colors.append(color)

        if not plotted_any:
            plt.close(fig)
            continue

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Wall time per call (ms)")
        ax.set_title(scenario)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        # ---- side-panel table ----
        from matplotlib.colors import to_rgba  # noqa: PLC0415

        col_labels = ["", "t* (ms)", "N*", "N*/t* (batch/ms)"]
        # Color the leftmost cell of each row to match the line color
        # for at-a-glance pairing with the figure legend. Wash the color
        # with alpha=0.35 so the label text stays legible against a
        # tinted (rather than saturated) background. Other cells stay
        # white so all rows render with consistent backgrounds.
        cell_colors: list[list[Any]] = [
            [to_rgba(c, alpha=0.35), "white", "white", "white"] for c in table_colors
        ]
        table = tax.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
            # Wider for the label column (carries strings like "v3 CUDA
            # (AOTI)") and the throughput column (carries the longest
            # numeric header). Sum < 1 so the table fits inside the
            # axes; matplotlib renders at (0.5 - sum/2) horizontally
            # centred otherwise.
            colWidths=[0.42, 0.16, 0.16, 0.26],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        # Vertical scale > 1.0 makes the row gaps breathe; horizontal at
        # 1.0 keeps the colWidths exact.
        table.scale(1.0, 1.5)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold")

        # `matplotlib.table` doesn't participate in tight_layout /
        # constrained_layout, so use subplots_adjust to leave space for
        # the title + axis labels. The right margin sits past the table
        # panel.
        fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.13)
        out = output_dir / f"{scenario}.{fmt}"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        written.append(out)
        print(f"  wrote {out}")

    return written


def _load_config(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        specs = raw.get("folders")
        if specs is None:
            raise ValueError(
                f"{path}: dict config must have a top-level 'folders' key. "
                "Alternatively, make the whole document a bare list."
            )
    else:
        specs = raw
    if not isinstance(specs, list):
        raise ValueError(f"{path}: 'folders' must be a list of folder specs.")
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError(f"{path}: entry [{i}] is missing required `path` key.")
    return specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plot_throughput",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="JSON config file (see module docstring)."
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write per-scenario figures."
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure file format (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=120, help="Raster DPI (default: 120).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = _load_config(args.config)
    plot_throughput(specs, args.output_dir, fmt=args.format, dpi=args.dpi)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
