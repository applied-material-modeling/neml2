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

"""Run one testbed case under one ablation arm and measure Newton convergence.

Route: ``py-eager``, CPU, float64. Iteration counts and residual histories are
route-independent -- the Newton loop is the *same* shared C++ code for eager and
AOTI (``neml2/csrc/aoti/newton.cpp``, reached from ``Newton.solve`` via
``newton_solve_eager``). Eager avoids an Inductor compile per grid point. This
is a convergence study, not a wall-time benchmark, so the "benchmarks run
through AOTI" rule does not apply; ``benchmark/_p3_precond_study.py`` makes the
same call for its Krylov study.

``NEML2_LOGS`` must be live before ``neml2`` is imported, so importing this
module sets it (and calls ``neml2.log.reload()`` defensively in case some other
import got there first).
"""

from __future__ import annotations

import math
import os
import re
import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Forced, not setdefault: the harness cannot function without the per-iteration
# `newton` debug stream, so an inherited NEML2_LOGS must not silently disable it.
os.environ["NEML2_LOGS"] = "newton=debug"
# Enrich a divergence with .converged_mask / .unknowns so a failed grid point
# reports *which* batch rows died rather than just "it threw".
os.environ.setdefault("NEML2_CAPTURE_SOLVE_FAILURE", "1")

import torch  # noqa: E402

from neml2 import load_input, log  # noqa: E402
from neml2.solvers import ConvergenceError  # noqa: E402

from .cases import DEFAULT_FLOW_N, Case  # noqa: E402

log.reload()

#: The four ablation arms. ``pred`` = predictor wired up; ``ls`` = line search
#: enabled (``max_linesearch_iterations`` > 1).
ARMS = ("pred+ls", "pred-ls", "nopred+ls", "nopred-ls")

#: ``max_linesearch_iterations`` for the two line-search states. 1 takes the
#: full-step branch in the C++ loop, i.e. exactly plain ``Newton``.
LS_ITERS = {True: 5, False: 1}

#: Time points in the parent scenarios, i.e. ``BASE_NPOINT - 1`` steps spanning
#: the full load history. Used only to recover the parent's per-step increment.
BASE_NPOINT = 100

#: Steps to run per case by default. The pathology is a *cold start*: it lives
#: entirely in step 1, and steps 2..N are only there to establish the
#: steady-state cost to compare against. A handful is plenty, and truncating the
#: history (rather than stretching it) keeps every increment identical to the
#: parent scenario's -- the same trick as ``benchmark/_gen_solver_study.py``.
DEFAULT_NSTEPS = 6

# `ITERATION   3, |R| = 1.08e-19, |R0| = 3.26e-04`. The literal `, |R| = `
# suffix is load-bearing: it excludes the indented `  LS ITERATION   n,
# min(alpha) = ...` line-search sub-iteration lines, which must not be counted
# as Newton iterations.
_ITER_RE = re.compile(r"ITERATION\s+(\d+), \|R\| = ([0-9.eE+-]+), \|R0\| = ([0-9.eE+-]+)")

# Solve delimiters, emitted at `info` by the shared C++ loop (`newton=debug`
# implies `info`). These -- not the `ITERATION 0` line -- are what bounds a
# solve, because the failure-capture masked re-run (`Newton::solve_masked`,
# triggered by NEML2_CAPTURE_SOLVE_FAILURE on a divergence) replays the same
# iterations with *no* banner and *no* iteration-0 line. Splitting on ITERATION 0
# would silently concatenate that replay onto the throwing solve and double its
# iteration count.
_BEGIN = "begin newton solve"
_END = "end newton solve"

_PREDICTOR_LINE = re.compile(r"^[ \t]*predictor = '[^']*'[ \t]*\n", re.MULTILINE)


#: A plateau's ratios must all sit within this factor of the plateau median.
_PLATEAU_BAND = 1.5
#: ...and there must be at least this many of them.
_PLATEAU_MIN = 3


def _longest_plateau(ratios: list[float]) -> float:
    """Median of the longest run of ratios that are near-constant, else NaN.

    "Near-constant" means every value in the run is within ``_PLATEAU_BAND`` of
    the run's median. Scanning for the *longest* such run, rather than greedily
    extending from the start, keeps a couple of ragged leading iterations from
    truncating an otherwise clean plateau.
    """
    best: list[float] = []
    for i in range(len(ratios)):
        for j in range(i + _PLATEAU_MIN, len(ratios) + 1):
            run = ratios[i:j]
            med = statistics.median(run)
            if med <= 0 or any(q > med * _PLATEAU_BAND or q < med / _PLATEAU_BAND for q in run):
                break
            if len(run) > len(best):
                best = run
    return statistics.median(best) if best else math.nan


@dataclass
class SolveTrace:
    """The residual history of a single Newton solve."""

    #: 1-based driver step this solve belongs to (the driver skips step 0).
    step: int
    #: |R| at each Newton iteration, starting at iteration 0 (the initial guess).
    residuals: list[float]

    @property
    def iters(self) -> int:
        """Newton iterations taken (the iteration-0 line is the initial guess)."""
        return len(self.residuals) - 1

    @property
    def r0(self) -> float:
        return self.residuals[0]

    @property
    def peak(self) -> float:
        return max(self.residuals)

    @property
    def overshoot(self) -> float:
        """Worst |R| relative to the initial |R|. > 1 means the first Newton
        step made things worse -- the signature of an unbalanced system."""
        return self.peak / self.r0 if self.r0 > 0 else math.nan

    def contraction(self) -> float:
        """Median |R_k| / |R_{k+1}| across the linear-convergence plateau, or NaN.

        A *plateau* is a run of consecutive iterations whose residual ratio is
        near-constant -- the signature of Newton degenerating to linear
        convergence in the far field of a power law. It is bounded below by the
        overshoot peak (an overshoot is not a contraction) and above by the
        onset of quadratic convergence (where the ratio runs away).

        Returns NaN when no plateau exists -- a solve that converges
        quadratically from the start has wildly-varying ratios, and reporting
        their median as a "contraction" would be meaningless. This matters for
        the small, benign sub-systems in a multi-solve step.

        For a residual dominated by a power law of exponent n under an
        *undamped* Newton step, the plateau ratio is ``(1 - 1/n)**-n``. Line
        search damps the step and shifts it, so only the ``-ls`` arms are
        comparable against :attr:`Record.theory_contraction`.
        """
        r = self.residuals
        ratios: list[float] = []
        for k in range(r.index(max(r)), len(r) - 1):
            if r[k] <= 0.0 or r[k + 1] <= 0.0:
                break
            ratios.append(r[k] / r[k + 1])
        return _longest_plateau(ratios)


@dataclass
class Record:
    """Everything one (case, arm, knobs) run produced."""

    case: str
    arm: str
    flow_n: float
    #: Per-step increment relative to the parent scenario's.
    dt_scale: int
    nbatch: int
    #: Driver steps run (``npoint - 1``).
    nsteps: int
    #: Fraction of the parent's full load history covered.
    tfrac: float
    max_its: int
    #: "ok", "diverged", or "skipped" (the arm is not runnable for this case).
    status: str
    #: Populated on divergence.
    error: str = ""
    #: 1-based driver step the solve failed at, or -1.
    failed_step: int = -1
    #: Batch members that failed, from ConvergenceError.converged_mask.
    n_failed_members: int = -1
    traces: list[SolveTrace] = field(default_factory=list)

    # -- derived metrics ---------------------------------------------------
    @property
    def per_step_iters(self) -> list[int]:
        """Newton iterations summed per driver step (a step may hold >1 solve)."""
        by_step: dict[int, int] = {}
        for t in self.traces:
            by_step[t.step] = by_step.get(t.step, 0) + t.iters
        return [by_step[k] for k in sorted(by_step)]

    @property
    def iters_step1(self) -> int:
        s = self.per_step_iters
        return s[0] if s else -1

    @property
    def median_iters_rest(self) -> float:
        s = self.per_step_iters[1:]
        return statistics.median(s) if s else math.nan

    @property
    def imbalance(self) -> float:
        """How much more the first step costs than a typical later step.

        Large means the solver is paying a cold-start penalty that a predictor
        cannot remove (there is nothing to extrapolate from at step 1).

        A *lower bound* at small ``nsteps``: the cost of a step keeps falling
        for the first ~10 steps as the state settles, so a short run's median
        over steps 2..N sits above the true steady state and deflates the ratio.
        (``vp_isoharden`` reads 5 at 6 steps and 15 at 99.) The per-step
        sequence in :attr:`iters_by_step` shows the shape without this caveat.
        """
        rest = self.median_iters_rest
        return self.iters_step1 / rest if rest and rest > 0 else math.nan

    @property
    def iters_by_step(self) -> str:
        """Per-step iteration counts as ``15-5-4-3-3-3``.

        The cold-start signature is a shape, not a scalar, and a short run shows
        it directly where a single ratio would understate it.
        """
        return "-".join(str(i) for i in self.per_step_iters)

    @property
    def total_iters(self) -> int:
        return sum(t.iters for t in self.traces)

    @property
    def step1_dominant(self) -> SolveTrace | None:
        """The step-1 solve that does the most work.

        A step may hold several solves over *different* sub-systems (e.g.
        ``cp_decoupled`` solves elastic-strain+hardening, then orientation).
        They have unrelated conditioning, so per-solve shape metrics must come
        from one of them, not an average. The longest is the one whose
        convergence dominates the step and the one nonlinear preconditioning
        would target.
        """
        first = [t for t in self.traces if t.step == 1]
        return max(first, key=lambda t: t.iters) if first else None

    @property
    def step1_overshoot(self) -> float:
        t = self.step1_dominant
        return t.overshoot if t else math.nan

    @property
    def step1_contraction(self) -> float:
        t = self.step1_dominant
        return t.contraction() if t else math.nan

    @property
    def theory_contraction(self) -> float:
        """``(1 - 1/n)**-n`` -- Newton's asymptotic rate on a pure power law."""
        n = self.flow_n
        return math.nan if n <= 1 else (1.0 - 1.0 / n) ** (-n)


def _prepare_input(case: Case, *, drop_predictor: bool, tmp: Path) -> Path:
    """Materialize the case input for one arm, stripping the predictor if asked.

    The predictor cannot be removed after construction: ``ImplicitUpdate``
    derives its declared inputs from the predictor's outputs, so clearing the
    attribute leaves the wiring inconsistent (``KeyError`` on the unknown's
    name). It has to come out of the input file.
    """
    if not drop_predictor:
        return case.input_file
    text, n = _PREDICTOR_LINE.subn("", case.input_file.read_text())
    if n == 0:
        # Drift guard, mirroring benchmark/_gen_solver_study.py: fail loudly
        # rather than silently measuring the wrong configuration.
        raise SystemExit(
            f"{case.input_file}: no `predictor = '...'` line found; the case file has "
            f"drifted and the nopred arm would silently measure the pred arm."
        )
    out = tmp / "model.i"
    out.write_text(text)
    return out


def _split_solves(lines: list[str], solves_per_step: int) -> list[SolveTrace]:
    """Group captured ITERATION lines into per-solve traces, tagged by step.

    Bounded by the begin/end banners; ITERATION lines outside a banner pair are
    the failure-capture replay and are dropped (see ``_BEGIN``).
    """
    solves: list[list[float]] = []
    cur: list[float] | None = None
    for ln in lines:
        if _BEGIN in ln:
            cur = []
            solves.append(cur)
            continue
        if _END in ln:
            cur = None
            continue
        if cur is None:
            continue
        m = _ITER_RE.search(ln)
        if m is not None:
            cur.append(float(m.group(2)))
    # The driver skips the model call at step 0 (ICs only), so the first solve
    # belongs to step 1.
    return [
        SolveTrace(step=i // solves_per_step + 1, residuals=r) for i, r in enumerate(solves) if r
    ]


def run(
    case: Case,
    arm: str,
    *,
    flow_n: float | None = None,
    dt_scale: int = 1,
    nsteps: int = DEFAULT_NSTEPS,
    nbatch: int = 8,
    max_its: int = 50,
) -> Record:
    """Drive ``case`` through ``nsteps`` time steps and measure every Newton solve.

    ``nsteps`` and ``dt_scale`` are independent. ``nsteps`` truncates the
    parent's load history; ``dt_scale`` scales the per-step increment. At
    ``dt_scale=1`` every increment is byte-identical to the parent scenario's,
    whatever ``nsteps`` is -- so a 6-step run measures the same physics as the
    parent's 99-step run, at 1/16 the cost. (Stretching a short history over the
    full load range instead would silently conflate "fewer steps" with "bigger
    steps" and make the cheap runs a different problem.)

    A divergence is captured, not raised: the grid must keep going, and a
    diverging corner is itself a result (it marks the edge of the convergence
    basin that a nonlinear preconditioner would need to enlarge).
    """
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm!r}; choose from {', '.join(ARMS)}")
    if nsteps < 1:
        raise SystemExit(f"nsteps must be >= 1, got {nsteps}")
    drop_predictor = arm.startswith("nopred")
    linesearch = arm.endswith("+ls")
    if flow_n is None:
        flow_n = DEFAULT_FLOW_N[case.flow_law]
    # The parent covers the full history (tfrac 1.0) in BASE_NPOINT-1 steps, so
    # one parent increment is 1/(BASE_NPOINT-1) of it. Take nsteps of those,
    # each dt_scale times larger.
    tfrac = dt_scale * nsteps / (BASE_NPOINT - 1)

    torch.set_default_dtype(torch.float64)
    rec = Record(
        case=case.name,
        arm=arm,
        flow_n=flow_n,
        dt_scale=dt_scale,
        nbatch=nbatch,
        nsteps=nsteps,
        tfrac=tfrac,
        max_its=max_its,
        status="ok",
    )

    if drop_predictor and not case.supports_nopred:
        # Not a harness failure -- a genuine NEML2 limitation this case exposes.
        # Record it as a row so the gap stays visible in the results instead of
        # looking like the arm was never attempted.
        rec.status = "skipped"
        rec.error = case.no_pred_blocker
        return rec

    lines: list[str] = []
    with tempfile.TemporaryDirectory(prefix=f"nlp_{case.name}_") as td:
        model_i = _prepare_input(case, drop_predictor=drop_predictor, tmp=Path(td))
        f = load_input(
            model_i,
            pre=[
                f"nbatch={nbatch}",
                f"npoint={nsteps + 1}",
                f"tfrac={tfrac!r}",
                f"flow_n={flow_n}",
                f"ls_iters={LS_ITERS[linesearch]}",
                f"max_its={max_its}",
            ],
        )
        driver = f.get_driver("driver")

        def sink(level: str, line: str) -> None:
            del level  # every channel we enabled is one we want
            lines.append(line)

        log.set_sink(sink)
        try:
            driver.run()
        except ConvergenceError as err:
            rec.status = "diverged"
            rec.error = str(err).split("\n", 1)[0][:200]
            mask = getattr(err, "converged_mask", None)
            if mask is not None:
                rec.n_failed_members = int((~mask).sum().item())
        finally:
            log.set_sink(None)

    rec.traces = _split_solves(lines, case.solves_per_step)

    if rec.status == "ok":
        # Bookkeeping check: a clean run must produce exactly one solve per
        # ImplicitUpdate per step. A mismatch means the case rewired itself (an
        # extra sub-system, a substepped solve) and every per-step metric below
        # would be misattributed.
        expected = case.solves_per_step * nsteps
        if len(rec.traces) != expected:
            raise SystemExit(
                f"{case.name}/{arm}: captured {len(rec.traces)} Newton solves, expected "
                f"{expected} (= solves_per_step {case.solves_per_step} x {nsteps} steps). "
                f"Update Case.solves_per_step in cases.py if the wiring changed."
            )
    elif rec.traces:
        rec.failed_step = rec.traces[-1].step

    return rec


__all__ = [
    "ARMS",
    "BASE_NPOINT",
    "DEFAULT_NSTEPS",
    "LS_ITERS",
    "Record",
    "SolveTrace",
    "run",
]
