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

"""Sweep the ablation grid and write the results as CSV.

Three named sub-sweeps, each answering one question, run instead of a full
cross-product (which would be ~10x the runs for no extra insight):

``arms``
    All four predictor/line-search combinations at the parent scenario's own
    stiffness and step size. Answers: *how much does the first step cost, and
    how much of the fix is the predictor vs the line search?*

``stiffness``
    ``flow_n`` swept with line search off. Answers: *is the plateau really
    Newton's* ``(1 - 1/n)**-n`` *rate on a power law?*

``stepsize``
    ``dt_scale`` swept -- each step a multiple of the parent's increment, with
    the step *count* held fixed. Answers: *where is the edge of the
    convergence basin?* -- the target a nonlinear preconditioner would move.

Usage::

    python -m studies.nlprecond.ablate --smoke
    python -m studies.nlprecond.ablate --sweep arms --case vp_isoharden
    python -m studies.nlprecond.ablate --output-dir studies/nlprecond/results/baseline
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import cases as case_registry
from .harness import ARMS, DEFAULT_NSTEPS, Record, run

DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "baseline"

#: Rate-sensitivity exponents for the `stiffness` sweep.
FLOW_N_SWEEP = (2.0, 4.0, 8.0, 12.0, 20.0)
#: Per-step increment multipliers, relative to the parent scenario's increment.
#: Orthogonal to the step count -- these make each step bigger, not fewer.
DT_SCALE_SWEEP = (1, 2, 5, 10, 20)

SUMMARY_FIELDS = [
    "sweep", "case", "arm", "flow_n", "dt_scale", "nbatch", "nsteps", "tfrac", "max_its",
    "status", "iters_step1", "median_iters_rest", "imbalance", "total_iters", "iters_by_step",
    "step1_overshoot", "step1_contraction", "theory_contraction",
    "failed_step", "n_failed_members", "error",
]  # fmt: skip
STEP_FIELDS = ["sweep", "case", "arm", "flow_n", "dt_scale", "step", "iters"]
TRACE_FIELDS = ["sweep", "case", "arm", "flow_n", "dt_scale", "iteration", "residual"]


@dataclass(frozen=True)
class Point:
    """One grid point."""

    sweep: str
    case: str
    arm: str
    flow_n: float | None
    dt_scale: int


def _grid(sweeps: Sequence[str], case_names: Sequence[str], arms: Sequence[str]) -> Iterator[Point]:
    for name in case_names:
        for sweep in sweeps:
            if sweep == "arms":
                for arm in arms:
                    yield Point("arms", name, arm, None, 1)
            elif sweep == "stiffness":
                # Line search off: the theoretical plateau rate assumes an
                # undamped Newton step, so a damped arm cannot test it.
                for arm in [a for a in arms if a.endswith("-ls")]:
                    for n in FLOW_N_SWEEP:
                        yield Point("stiffness", name, arm, n, 1)
            elif sweep == "stepsize":
                for arm in arms:
                    for s in DT_SCALE_SWEEP:
                        yield Point("stepsize", name, arm, None, s)
            else:  # pragma: no cover - argparse restricts the choices
                raise SystemExit(f"unknown sweep {sweep!r}")


def _fmt(x: float) -> str:
    return "" if isinstance(x, float) and math.isnan(x) else f"{x:.6g}"


def _summary_row(pt: Point, rec: Record) -> dict:
    return {
        "sweep": pt.sweep,
        "case": rec.case,
        "arm": rec.arm,
        "flow_n": _fmt(rec.flow_n),
        "dt_scale": rec.dt_scale,
        "nbatch": rec.nbatch,
        "nsteps": rec.nsteps,
        "tfrac": _fmt(rec.tfrac),
        "max_its": rec.max_its,
        "status": rec.status,
        "iters_step1": rec.iters_step1,
        "median_iters_rest": _fmt(rec.median_iters_rest),
        "imbalance": _fmt(rec.imbalance),
        "total_iters": rec.total_iters,
        "iters_by_step": rec.iters_by_step,
        "step1_overshoot": _fmt(rec.step1_overshoot),
        "step1_contraction": _fmt(rec.step1_contraction),
        "theory_contraction": _fmt(rec.theory_contraction),
        "failed_step": rec.failed_step,
        "n_failed_members": rec.n_failed_members,
        "error": rec.error,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sweep",
        action="append",
        choices=["arms", "stiffness", "stepsize"],
        help="sub-sweep to run; repeatable (default: all three)",
    )
    ap.add_argument(
        "--case",
        action="append",
        choices=list(case_registry.CASES),
        help="restrict to these cases; repeatable (default: all)",
    )
    ap.add_argument(
        "--arm",
        action="append",
        choices=list(ARMS),
        help="restrict to these arms; repeatable (default: all)",
    )
    ap.add_argument("--nbatch", type=int, default=8, help="dynamic batch members (default: 8)")
    ap.add_argument(
        "--nsteps",
        type=int,
        default=DEFAULT_NSTEPS,
        help=f"time steps per run, at the parent's own increment (default: {DEFAULT_NSTEPS})",
    )
    ap.add_argument("--max-its", type=int, default=50, help="Newton iteration cap (default: 50)")
    ap.add_argument(
        "--output-dir", type=Path, default=None, help=f"write CSVs here (default: {DEFAULT_OUT})"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="fast wiring check: every case, every arm, 4 steps, no CSV",
    )
    args = ap.parse_args(argv)

    case_names = args.case or list(case_registry.CASES)
    arms = args.arm or list(ARMS)

    if args.smoke:
        # The parent's own increment (dt_scale 1), so the well-behaved arms
        # converge and the run exercises the per-step bookkeeping assertion --
        # which only fires on a clean run -- and not just the divergence path.
        points = [Point("smoke", c, a, None, 1) for c in case_names for a in arms]
        args.nsteps = min(args.nsteps, 4)
        out_dir = None
    else:
        points = list(_grid(args.sweep or ["arms", "stiffness", "stepsize"], case_names, arms))
        out_dir = args.output_dir or DEFAULT_OUT
        out_dir.mkdir(parents=True, exist_ok=True)

    # Open all three CSVs up front and flush per row: a sweep is long, and a
    # crash or Ctrl-C must not discard the points already measured.
    files, writers = [], {}
    if out_dir is not None:
        csvs = (("summary", SUMMARY_FIELDS), ("steps", STEP_FIELDS), ("traces", TRACE_FIELDS))
        for tag, fields in csvs:
            fh = (out_dir / f"{tag}.csv").open("w", newline="")
            files.append(fh)
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            writers[tag] = w

    n_bad = 0
    try:
        for i, pt in enumerate(points, start=1):
            case = case_registry.get(pt.case)
            rec = run(
                case,
                pt.arm,
                flow_n=pt.flow_n,
                dt_scale=pt.dt_scale,
                nbatch=args.nbatch,
                max_its=args.max_its,
            )
            if rec.status == "diverged":
                n_bad += 1
            print(
                f"[{i:3d}/{len(points)}] {pt.sweep:9s} {rec.case:13s} {rec.arm:10s} "
                f"n={rec.flow_n:<5g} dt x{rec.dt_scale:<3d} {rec.status:8s} "
                f"step1={rec.iters_step1:3d} med={_fmt(rec.median_iters_rest) or '-':>5s} "
                f"imb={_fmt(rec.imbalance) or '-':>6s} total={rec.total_iters:5d}",
                flush=True,
            )
            if out_dir is None:
                continue
            writers["summary"].writerow(_summary_row(pt, rec))
            base = {
                "sweep": pt.sweep,
                "case": rec.case,
                "arm": rec.arm,
                "flow_n": _fmt(rec.flow_n),
                "dt_scale": rec.dt_scale,
            }
            for step, iters in enumerate(rec.per_step_iters, start=1):
                writers["steps"].writerow({**base, "step": step, "iters": iters})
            # Only the step-1 trace: it is the object of study, and dumping
            # every solve would balloon the CSV by ~100x for no added insight.
            dom = rec.step1_dominant
            if dom is not None:
                for k, r in enumerate(dom.residuals):
                    writers["traces"].writerow({**base, "iteration": k, "residual": f"{r:.8e}"})
            for fh in files:
                fh.flush()
    finally:
        for fh in files:
            fh.close()

    if out_dir is not None:
        print(f"\nwrote {out_dir}/summary.csv, steps.csv, traces.csv")
    print(f"{len(points)} point(s), {n_bad} diverged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
