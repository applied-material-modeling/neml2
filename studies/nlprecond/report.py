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

"""Render the three headline tables from a sweep's ``summary.csv``.

Usage::

    python -m studies.nlprecond.report [results-dir]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .ablate import DEFAULT_OUT

Rows = list[dict[str, str]]


def _load(results_dir: Path) -> Rows:
    path = results_dir / "summary.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found; run `python -m studies.nlprecond.ablate` first")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "  (no rows)\n"
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*header), "  " + "  ".join("-" * w for w in widths)]
    out += [fmt.format(*r) for r in rows]
    return "\n".join(out) + "\n"


def _status(row: dict[str, str]) -> str:
    """Status with the divergence detail folded in."""
    if row["status"] != "diverged":
        return row["status"]
    n = row["n_failed_members"]
    where = f"@step{row['failed_step']}" if row["failed_step"] != "-1" else ""
    return f"diverged{where}" + (f" ({n} rows)" if n not in ("-1", "") else "")


def arms_table(rows: Rows) -> str:
    """How expensive the first step is, and what each globalization buys."""
    sel = [r for r in rows if r["sweep"] == "arms"]
    body = [
        [
            r["case"],
            r["arm"],
            _status(r),
            r["iters_step1"],
            r["median_iters_rest"] or "-",
            r["imbalance"] or "-",
            r["total_iters"],
            r["iters_by_step"],
            r["step1_overshoot"] or "-",
        ]
        for r in sel
    ]
    head = ["case", "arm", "status", "step1", "med_rest", "imb", "total", "per-step", "overshoot"]
    return _table(head, body)


def stiffness_table(rows: Rows) -> str:
    """Measured plateau contraction against Newton's ``(1-1/n)**-n`` on a power law.

    Diverged runs are shown but flagged: the theory describes a solve that
    passes *through* the plateau into the quadratic basin, so on a run that hit
    the iteration cap the detected "plateau" may be part of a diverging
    sequence rather than the linear phase. Their rel.err is not evidence
    either way, and dropping them silently would overstate the agreement.
    """
    sel = [r for r in rows if r["sweep"] == "stiffness" and r["step1_contraction"]]
    body = [
        [
            r["case"],
            r["arm"],
            r["flow_n"],
            r["iters_step1"],
            "DIV" if r["status"] == "diverged" else "",
            r["step1_contraction"],
            r["theory_contraction"],
            _relerr(r["step1_contraction"], r["theory_contraction"]),
        ]
        for r in sel
    ]
    head = ["case", "arm", "n", "step1", "", "measured", "(1-1/n)^-n", "rel.err"]
    return _table(head, body)


def _relerr(measured: str, theory: str) -> str:
    try:
        m, t = float(measured), float(theory)
    except ValueError:
        return "-"
    return "-" if t == 0 else f"{abs(m - t) / t:.2%}"


def stepsize_table(rows: Rows) -> str:
    """Where each arm falls off the convergence basin as the increment grows."""
    sel = [r for r in rows if r["sweep"] == "stepsize"]
    cases = sorted({r["case"] for r in sel})
    scales = sorted({int(r["dt_scale"]) for r in sel})
    arms = sorted({r["arm"] for r in sel})
    by = {(r["case"], r["arm"], int(r["dt_scale"])): r for r in sel}
    body = []
    for case in cases:
        for arm in arms:
            cells = []
            for s in scales:
                r = by.get((case, arm, s))
                if r is None:
                    cells.append("-")
                elif r["status"] == "ok":
                    cells.append(r["total_iters"])
                else:
                    cells.append("DIV")
            body.append([case, arm, *cells])
    head = ["case", "arm", *(f"x{s}" for s in scales)]
    return _table(head, body)


def render(results_dir: Path) -> str:
    rows = _load(results_dir)
    parts = [
        f"nlprecond baseline -- {results_dir}",
        "",
        "1. ARMS -- cost of the first step, and what predictor / line search each buy",
        "   per-step = Newton iterations at each step. The cold-start penalty is the",
        "   drop from the first entry to the rest -- a penalty a predictor cannot",
        "   remove, because at step 1 there is no previous solution to extrapolate.",
        "   imb = step1 / median(rest); a LOWER bound at small --nsteps (the state is",
        "   still settling, so median(rest) sits above the true steady state).",
        "",
        arms_table(rows),
        "2. STIFFNESS -- the plateau is Newton's linear rate on a power law",
        "   Undamped (-ls) arms only: line search damps the step and shifts the rate.",
        "",
        stiffness_table(rows),
        "3. STEPSIZE -- total Newton iterations vs increment size (DIV = diverged)",
        "   xN means each step is N times the parent scenario's increment; the step",
        "   count is unchanged. The boundary between numbers and DIV is the",
        "   convergence basin edge a nonlinear preconditioner would push outward.",
        "",
        stepsize_table(rows),
    ]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUT,
        help=f"sweep output directory (default: {DEFAULT_OUT})",
    )
    args = ap.parse_args(argv)
    print(render(args.results_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
