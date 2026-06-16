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

"""v3 benchmark sweep: per-scenario adaptive batch scaling study.

Decoupled subcommands so a sweep can be assembled one piece at a time:

    # Create the result folder and record environment + sweep config:
    python -m benchmark.sweep init --device cpu [--output-dir DIR]

    # Run one or more scenarios into an existing folder:
    python -m benchmark.sweep run --output-dir DIR --scenarios chaboche2 [--force]

    # Rebuild summary.csv + metadata.json:sweep.batch_ranges from per-scenario CSVs:
    python -m benchmark.sweep summarize --output-dir DIR

    # End-to-end shortcut (init + run all discovered scenarios + summarize):
    python -m benchmark.sweep all --device cpu [--output-dir DIR]

Scenarios are auto-discovered: any directory ``benchmark/<name>/model.i``
is a sweep scenario. To add one, drop a ``model.i`` next to the others;
no edit to this file is needed.

``run`` skips scenarios that already have a ``<scenario>.csv`` on disk
unless ``--force`` is passed; this is what makes incremental sweeps work
(add one scenario at a time without re-running the rest). ``summarize``
is always safe to run -- it only reads per-scenario CSVs and rewrites
the aggregate view.

Sweep tunables (``--warmup``, ``--repeats``, ``--max-batch``,
``--max-seconds``) are fixed at ``init`` time and stored in the folder's
``metadata.json``. ``run`` reads them from disk so every scenario in a
folder shares the same stop-conditions; to change them, ``init`` a new
folder.

Per-scenario stop conditions (any one trips, whichever comes first):
* log-log slope >= ``LINEAR_SLOPE_THRESHOLD`` for
  ``LINEAR_STREAK_REQUIRED`` consecutive points (linear regime reached);
* median call time > ``--max-seconds`` (saturation by absolute wall
  time);
* CUDA OOM / host MemoryError;
* batch size > ``--max-batch`` (absolute hard cap).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from benchmark._env import collect_environment, folder_timestamp
from benchmark._stats import (
    compute_stats,
    loglog_slope,
    record_memory,
    reset_memory_peak,
    time_callable,
)

_BENCHMARK_DIR = Path(__file__).resolve().parent
_REPO_DIR = _BENCHMARK_DIR.parent

# Stop-condition tunables. Hardcoded to keep the CLI surface small; a
# motivated user edits these.
LINEAR_SLOPE_THRESHOLD = 0.85
LINEAR_STREAK_REQUIRED = 2
SLOPE_WINDOW = 3

# The driver block's ``model = '...'`` field tells us what to compile --
# no separate per-scenario override map needed. ``neml2-compile --driver``
# does the same lookup; we call the helper directly to keep the export +
# stub-emit calls feeding off the same source of truth.


def _discover_scenarios() -> tuple[str, ...]:
    """Return every ``benchmark/<name>/model.i`` directory name, sorted.

    Filesystem discovery instead of a hardcoded list so adding a new
    scenario directory is the whole patch -- no second edit to keep this
    file in sync.
    """
    return tuple(sorted(p.parent.name for p in _BENCHMARK_DIR.glob("*/model.i")))


ALL_SCENARIOS = _discover_scenarios()

_CSV_FIELDS = [
    "scenario",
    "nbatch",
    "n_warmup",
    "n_runs",
    "median_ms",
    "mean_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "p10_ms",
    "p90_ms",
    # Memory columns (added in the privacy + memory refresh). ``cuda_peak_mb``
    # is NaN on CPU runs; ``cpu_rss_mb`` is the process resident-set after
    # the batch's measurement window.
    "cpu_rss_mb",
    "cuda_peak_mb",
]


# ---------------------------------------------------------------------------
# Folder + metadata
# ---------------------------------------------------------------------------


def _autoname_output_dir(device: str) -> Path:
    # Hostname intentionally omitted from the folder name: HPC nodes often
    # reveal the institution and the folder is committed to a public repo.
    # Distinct runs on the same day are disambiguated by the UTC timestamp's
    # second resolution.
    name = f"{folder_timestamp()}_{device}_float64_aoti"
    return _REPO_DIR / "benchmark" / "results" / name


def init_folder(
    out_dir: Path,
    *,
    device: str,
    warmup: int,
    repeats: int,
    max_batch: int,
    max_seconds: float,
) -> dict[str, Any]:
    """Create ``out_dir`` and write a fresh ``metadata.json``.

    Captures the host environment, records the sweep config (warmup,
    repeats, max-batch, max-seconds) so subsequent ``run`` invocations
    inherit identical stop conditions, and stamps ``started_utc``.
    Refuses to overwrite an existing ``metadata.json`` -- use ``run`` to
    add scenarios or ``summarize`` to refresh derived fields.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        raise SystemExit(
            f"{meta_path} already exists. Use `run` to add scenarios or "
            f"`summarize` to refresh derived fields; remove the folder by "
            f"hand if you really want a fresh init."
        )
    meta = collect_environment(device)
    meta["sweep"] = {
        "device": device,
        "dtype": "float64",
        "mode": "aoti",
        "warmup": warmup,
        "repeats": repeats,
        "max_batch": max_batch,
        "max_seconds": max_seconds,
        "linear_slope_threshold": LINEAR_SLOPE_THRESHOLD,
        "linear_streak_required": LINEAR_STREAK_REQUIRED,
        "slope_window": SLOPE_WINDOW,
        "last_run_wall_seconds": None,
        "batch_ranges": {},
        "failed_scenarios": [],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"Initialised {out_dir} (device={device}, warmup={warmup}, repeats={repeats}).")
    return meta


def _load_meta(out_dir: Path) -> dict[str, Any]:
    """Read ``metadata.json``; raise if the folder isn't init'd."""
    meta_path = out_dir / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found. Run `init --device <cpu|cuda> --output-dir {out_dir}` first."
        )
    return json.loads(meta_path.read_text())


def _write_meta(out_dir: Path, meta: dict[str, Any]) -> None:
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# Per-scenario sweep
# ---------------------------------------------------------------------------


def sweep_scenario(
    scenario: str,
    *,
    device: str,
    warmup: int,
    repeats: int,
    compile_dir: Path,
    out_dir: Path,
    max_batch: int,
    max_seconds: float,
) -> list[dict[str, Any]]:
    """Run an adaptive batch sweep on one scenario; write per-scenario CSVs."""
    from neml2 import load_input  # noqa: PLC0415
    from neml2.cli.aoti_compile import compile_and_emit_stub  # noqa: PLC0415

    model_i = _BENCHMARK_DIR / scenario / "model.i"
    if not model_i.exists():
        raise FileNotFoundError(f"no model.i under benchmark/{scenario}/")

    rows: list[dict[str, Any]] = []
    raw_runs: list[tuple[int, list[float]]] = []
    medians: list[float] = []
    batches: list[int] = []
    linear_streak = 0
    B = 1
    scenario_csv = out_dir / f"{scenario}.csv"
    runs_csv = out_dir / f"{scenario}.runs.csv"
    while B <= max_batch:
        # Compile per batch -- the per-input ``example_batch_shape`` declared
        # in each scenario's ``[Settings]`` block governs the trace shapes,
        # parameterised by ``${nbatch}``. The autotune block-size hint is
        # derived from the trace example, and the optimal hint is workload-
        # dependent in ways that aren't predictable from first principles
        # (measured: scpcoup at B=8192 takes 12.6 s when compiled with
        # example=2 vs 2.1 s with example=8192 -- a 6x difference from the
        # autotune choice alone). The earlier ``TRACE_BATCH=2`` shared compile
        # was unfair to CP scenarios at large batch; passing ``nbatch=B`` per
        # iter lets each scenario's ``[Settings]`` resolve correctly. NEVER
        # pass ``example_batch_shape=`` here -- as a CLI string it becomes a
        # UNIFORM override that wipes per-input declarations (e.g. mxpc's
        # ``elastic_strain~1 = '(2; ${nbatch}:grain)'`` would silently
        # collapse to ``(B,)`` for every input, treating the per-grain sub-
        # batch as dynamic batch).
        pre = (f"nbatch={B}", f"device={device}")
        try:
            t0 = time.perf_counter()
            run_input = compile_and_emit_stub(
                model_i,
                compile_dir,
                driver="driver",
                device=device,
                pre=pre,
            )
            compile_s = time.perf_counter() - t0
            print(f"[{scenario}] B={B:6d} compile: {compile_s:.1f} s", flush=True)

            factory = load_input(run_input, pre=pre)
            driver = factory.get_driver("driver")
            # Reset CUDA peak BEFORE the measurement window so the value we
            # report afterward is this batch's own peak (not contaminated by
            # warmup at smaller batches that's still in the allocator).
            reset_memory_peak(device)
            times = time_callable(driver.run, device=device, n_warmup=warmup, n_runs=repeats)
            mem = record_memory(device)
        except (RuntimeError, MemoryError) as exc:
            # CUDA OOM and host-OOM both end the sweep for this scenario.
            print(f"[{scenario}] B={B}: aborted ({type(exc).__name__}: {exc!s})", flush=True)
            break
        stats = compute_stats(times)
        row = {"scenario": scenario, "nbatch": B, "n_warmup": warmup, **stats, **mem}
        rows.append(row)
        raw_runs.append((B, [float(t * 1000.0) for t in times]))
        medians.append(stats["median_ms"])
        batches.append(B)

        # Flush per-batch so a torch C++ crash on a later batch doesn't take
        # the whole scenario's data with it. The full file is overwritten on
        # every successful batch.
        _write_scenario_csv(scenario_csv, rows)
        _write_runs_csv(runs_csv, raw_runs)

        slope = loglog_slope(batches[-SLOPE_WINDOW:], medians[-SLOPE_WINDOW:])
        slope_str = "n/a" if slope != slope else f"{slope:+.2f}"
        print(
            f"[{scenario}] B={B:6d}  median={stats['median_ms']:9.2f} ms  "
            f"std={stats['std_ms']:8.2f}  slope={slope_str}",
            flush=True,
        )

        if stats["median_ms"] / 1000.0 > max_seconds:
            print(f"[{scenario}] stop: per-call median exceeds {max_seconds:.1f} s", flush=True)
            break
        if slope == slope and slope >= LINEAR_SLOPE_THRESHOLD:  # noqa: PLR2004
            linear_streak += 1
        else:
            linear_streak = 0
        if linear_streak >= LINEAR_STREAK_REQUIRED:
            print(
                f"[{scenario}] stop: {LINEAR_STREAK_REQUIRED} consecutive slopes "
                f">= {LINEAR_SLOPE_THRESHOLD:.2f} (linear regime reached)",
                flush=True,
            )
            break
        B *= 2
    else:
        print(f"[{scenario}] stop: hit --max-batch={max_batch}", flush=True)

    return rows


def _write_scenario_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_runs_csv(path: Path, raw: list[tuple[int, list[float]]]) -> None:
    """One row per (nbatch, run_index, time_ms) -- long format."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["nbatch", "run_index", "time_ms"])
        for B, times_ms in raw:
            for i, t in enumerate(times_ms):
                writer.writerow([B, i, f"{t:.6f}"])


def run_scenarios(
    out_dir: Path,
    scenarios: list[str],
    *,
    force: bool = False,
) -> int:
    """Run ``scenarios`` into ``out_dir``; skip those already on disk unless ``force``.

    Returns the number of scenarios that actually executed (0 if every
    requested scenario was already present). Reads ``warmup`` / ``repeats``
    / ``max_batch`` / ``max_seconds`` / ``device`` from the folder's
    ``metadata.json`` so every scenario in a folder shares stop conditions.
    """
    meta = _load_meta(out_dir)
    sweep_cfg = meta.setdefault("sweep", {})
    device = sweep_cfg["device"]
    dtype = sweep_cfg["dtype"]

    import torch  # noqa: PLC0415

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            f"Folder is pinned to device={device!r} but torch.cuda.is_available() is False."
        )

    # Mirror v2's tool main(): set process-wide torch defaults so the [Tensors]
    # block Python expressions (``SR2.fill(0.1, ...)``,
    # ``torch.tensor([...])``, ...) build tensors on the right device and
    # dtype without needing per-expression annotations. The AOTI runtime
    # validates inputs at the model boundary (``AOTIModel._check_tensor``);
    # without these defaults the validation fires immediately. End
    # applications own placement; embedders set their own defaults.
    torch.set_default_dtype(getattr(torch, dtype))
    torch.set_default_device(device)

    # Reuse TORCHINDUCTOR_CACHE_DIR across sessions so AOTI compile is warm.
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        str(_REPO_DIR / "benchmark" / "results" / "_aoti_cache"),
    )

    t_session = time.perf_counter()
    n_ran = 0
    for scenario in scenarios:
        scenario_csv = out_dir / f"{scenario}.csv"
        if scenario_csv.exists() and not force:
            print(f"[{scenario}] skip (already on disk; pass --force to rerun)", flush=True)
            continue
        # Clear any stale failed-scenarios entry -- we're about to either
        # succeed or write a new failure.
        sweep_cfg["failed_scenarios"] = [
            f for f in sweep_cfg.get("failed_scenarios", []) if f.get("scenario") != scenario
        ]
        with tempfile.TemporaryDirectory(prefix=f"sweep_{scenario}_") as tmp:
            try:
                sweep_scenario(
                    scenario,
                    device=device,
                    warmup=sweep_cfg["warmup"],
                    repeats=sweep_cfg["repeats"],
                    compile_dir=Path(tmp),
                    out_dir=out_dir,
                    max_batch=sweep_cfg["max_batch"],
                    max_seconds=sweep_cfg["max_seconds"],
                )
                n_ran += 1
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                sweep_cfg["failed_scenarios"].append(
                    {"scenario": scenario, "error": f"{type(exc).__name__}: {exc!s}"}
                )

    session_wall = round(time.perf_counter() - t_session, 2)
    # Only stamp last_run_wall_seconds when something actually ran -- a
    # skip-everything invocation shouldn't overwrite the wall time of the
    # most recent meaningful run.
    if n_ran > 0:
        sweep_cfg["last_run_wall_seconds"] = session_wall
    _write_meta(out_dir, meta)
    # Refresh summary.csv + batch_ranges from disk so the artifacts stay
    # coherent even if a scenario crashed mid-loop and only partial CSV
    # data exists (per-batch flushing in sweep_scenario keeps the row data
    # alive across torch C++ crashes).
    summarize(out_dir)
    print(f"\nrun: {n_ran} scenario(s) executed. Session wall: {session_wall:.1f} s.")
    return n_ran


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _iter_scenario_csvs(folder: Path):
    """Yield ``(scenario_name, csv_path)`` for every per-scenario CSV on disk."""
    aggregates = {"summary.csv", "fitting.csv"}
    for csv_path in sorted(folder.glob("*.csv")):
        name = csv_path.name
        if name in aggregates or name.endswith(".runs.csv"):
            continue
        yield csv_path.stem, csv_path


def summarize(out_dir: Path) -> int:
    """Rebuild ``summary.csv`` and ``metadata.json:sweep.batch_ranges`` from disk.

    Returns the total number of rows written to ``summary.csv``.
    """
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist.")
    summary_path = out_dir / "summary.csv"
    all_rows: list[dict[str, str]] = []
    batch_ranges: dict[str, list[int]] = {}

    for scenario, csv_path in _iter_scenario_csvs(out_dir):
        with csv_path.open() as f:
            scenario_rows = list(csv.DictReader(f))
        all_rows.extend(scenario_rows)
        batch_ranges[scenario] = [int(r["nbatch"]) for r in scenario_rows]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    meta_path = out_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta.setdefault("sweep", {})["batch_ranges"] = batch_ranges
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    print(
        f"summarize: rebuilt summary.csv ({len(all_rows)} rows, "
        f"{len(batch_ranges)} scenarios) in {out_dir}."
    )
    return len(all_rows)


# ---------------------------------------------------------------------------
# Fit (vectorization efficiency, eq 7.1 from ANL-24/43)
# ---------------------------------------------------------------------------


def _fit_t0_Nstar(batches, times_ms):
    """Fit (t_0, N*) to the piecewise model

        t(N) = t_0 * (max(N - N*, 0) / N* + 1)

    i.e. flat at ``t_0`` for ``N <= N*`` and linear-with-slope-``t_0/N*``
    beyond. Fit in log-log space because the spread of ``t`` across
    batches typically ranges over several decades; log residuals weight
    every order of magnitude equally instead of letting the largest
    batch dominate the loss.

    Returns ``(t0_ms, N_star, residual_rms_log)`` where the residual is
    the root-mean-square of ``log(t_pred) - log(t_meas)`` in natural log
    units (so multiplying by ~0.434 gives ``log10`` decades, and the raw
    value is the typical relative error in t).
    """
    import numpy as np  # noqa: PLC0415
    from scipy.optimize import least_squares  # noqa: PLC0415

    N = np.asarray(batches, dtype=np.float64)
    t = np.asarray(times_ms, dtype=np.float64)
    if N.size < 2:
        return float("nan"), float("nan"), float("nan")

    def model(params, N):
        t0, Ns = params
        # ``max(N - Ns, 0) / Ns + 1`` — Macaulay bracket / ramp form
        return t0 * (np.maximum(N - Ns, 0.0) / Ns + 1.0)

    def residual(params, N, log_t):
        return np.log(model(params, N)) - log_t

    log_t = np.log(t)
    # Initial guesses: t_0 ≈ smallest-batch time; N* ≈ geometric midpoint
    # of the measured range. log-bounds on both parameters so the
    # optimizer doesn't wander into the negative side or to absurd
    # values that the model can't represent meaningfully.
    p0 = [float(t.min()), float(np.exp((np.log(N.min()) + np.log(N.max())) / 2))]
    bounds = ([1e-6, 1.0], [1e6, 1e12])
    try:
        result = least_squares(residual, p0, args=(N, log_t), bounds=bounds, method="trf")
        t0_fit, Ns_fit = float(result.x[0]), float(result.x[1])
        rms = float(np.sqrt((result.fun**2).mean()))
        return t0_fit, Ns_fit, rms
    except Exception:  # noqa: BLE001
        return float("nan"), float("nan"), float("nan")


def fit(out_dir: Path) -> int:
    """Fit eq 7.1 per scenario; write ``fitting.csv`` next to ``summary.csv``.

    Uses the per-batch median timings from each scenario's CSV.
    Returns the number of scenarios fitted.
    """
    if not out_dir.exists():
        raise SystemExit(f"{out_dir} does not exist.")
    fitting_path = out_dir / "fitting.csv"

    fields = ["scenario", "t0_ms", "N_star", "log_rms", "n_points"]
    rows: list[dict[str, str]] = []
    for scenario, csv_path in _iter_scenario_csvs(out_dir):
        batches: list[int] = []
        times: list[float] = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                try:
                    batches.append(int(row["nbatch"]))
                    times.append(float(row["median_ms"]))
                except (KeyError, ValueError):
                    continue
        if len(batches) < 2:
            print(f"  {scenario}: skipped ({len(batches)} data point(s); need >= 2)")
            continue
        t0, Ns, rms = _fit_t0_Nstar(batches, times)
        rows.append(
            {
                "scenario": scenario,
                "t0_ms": f"{t0:.6f}",
                "N_star": f"{Ns:.2f}",
                "log_rms": f"{rms:.4f}",
                "n_points": str(len(batches)),
            }
        )
        print(
            f"  {scenario:<14}  t0 = {t0:9.3f} ms   N* = {Ns:>10.0f}   "
            f"log_rms = {rms:.3f}   ({len(batches)} pts)"
        )

    with fitting_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nfit: wrote {len(rows)} fits to {fitting_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_init_args(p: argparse.ArgumentParser) -> None:
    """Args common to ``init`` and ``all`` (which forwards to init)."""
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument(
        "--max-batch",
        type=int,
        default=65536,
        help="Hard cap on the largest batch tried per scenario.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Stop a scenario sweep when median call time exceeds this.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Folder to create. Defaults to "
            "benchmark/results/<UTC-ts>_<device>_float64_aoti_<host>/."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.sweep")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="{init,run,summarize,fit,all}")

    p_init = sub.add_parser(
        "init",
        help="Create folder + metadata.json snapshot.",
        description=(
            "Initialise an empty result folder: write metadata.json with the "
            "host environment snapshot and the sweep config (warmup, repeats, "
            "max-batch, max-seconds). No scenarios are run."
        ),
    )
    _add_init_args(p_init)

    p_run = sub.add_parser(
        "run",
        help="Run one or more scenarios into an existing folder.",
        description=(
            "Run scenarios into a folder previously created by `init`. "
            "Skips scenarios that already have a <scenario>.csv unless "
            "--force is passed. Uses warmup/repeats/max-batch/max-seconds "
            "from the folder's metadata.json."
        ),
    )
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument(
        "--scenarios",
        nargs="+",
        default=list(ALL_SCENARIOS),
        help="Scenarios to run. Default: every benchmark/<name>/model.i.",
    )
    p_run.add_argument(
        "--force",
        action="store_true",
        help="Re-run scenarios that already have a <scenario>.csv on disk.",
    )

    p_sum = sub.add_parser(
        "summarize",
        help="Rebuild summary.csv from per-scenario CSVs.",
        description=(
            "Rebuild summary.csv and metadata.json:sweep.batch_ranges from "
            "the per-scenario CSVs currently on disk under --output-dir. "
            "Always safe to run; does not modify per-scenario files."
        ),
    )
    p_sum.add_argument("--output-dir", type=Path, required=True)

    p_fit = sub.add_parser(
        "fit",
        help="Fit eq 7.1 per scenario and write fitting.csv.",
        description=(
            "For each per-scenario CSV under --output-dir, nonlinear-least-"
            "squares-fit the vectorisation-efficiency model "
            "``t(N) = t_0 * (max(N - N*, 0) / N* + 1)`` (eq 7.1 of ANL-24/43) "
            "in log-log space and write the fitted (t_0, N*) plus the log-"
            "space RMS residual to fitting.csv alongside summary.csv."
        ),
    )
    p_fit.add_argument("--output-dir", type=Path, required=True)

    p_all = sub.add_parser(
        "all",
        help="End-to-end shortcut: init + run every discovered scenario + summarize.",
        description=(
            "Convenience wrapper: init a fresh folder, run every scenario, "
            "then summarize. Equivalent to running the three subcommands in "
            "sequence; use the decomposed form when you want incremental "
            "control or are recovering from a partial run."
        ),
    )
    _add_init_args(p_all)
    p_all.add_argument(
        "--scenarios",
        nargs="+",
        default=list(ALL_SCENARIOS),
        help="Scenarios to run. Default: every benchmark/<name>/model.i.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "init":
        out_dir = args.output_dir or _autoname_output_dir(args.device)
        init_folder(
            out_dir,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            max_batch=args.max_batch,
            max_seconds=args.max_seconds,
        )
        # Echo the resolved path so callers can capture it.
        print(out_dir)
        return 0

    if args.cmd == "run":
        run_scenarios(args.output_dir, args.scenarios, force=args.force)
        return 0

    if args.cmd == "summarize":
        summarize(args.output_dir)
        return 0

    if args.cmd == "fit":
        fit(args.output_dir)
        return 0

    if args.cmd == "all":
        out_dir = args.output_dir or _autoname_output_dir(args.device)
        init_folder(
            out_dir,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            max_batch=args.max_batch,
            max_seconds=args.max_seconds,
        )
        run_scenarios(out_dir, args.scenarios, force=False)
        summarize(out_dir)
        print(f"\nDone. Folder: {out_dir}")
        return 0

    print(f"unknown subcommand: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
