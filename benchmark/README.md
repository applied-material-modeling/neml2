# Benchmark suite

Two layers live here:

1. **Smoke ctests** (`benchmark/CMakeLists.txt` + `benchmark/run_benchmark.py`).
   One ctest entry per scenario at batch=2 — compile + run, just confirm the
   pipeline still works. Registered into CMake when `BUILD_TESTING` is on.

2. **Scaling sweeps** (`benchmark/sweep.py`). The thing you run when you
   want actual perf numbers across a range of batch sizes.

This page documents the sweep.

## Running a sweep

`sweep.py` is decomposed into four subcommands. Each is independently
useful; an end-to-end shortcut chains them together.

```bash
# Quickest path — fresh folder, all 12 scenarios, then summarise:
python -m benchmark.sweep all --device cpu

# Step by step (the same flow, manually):
python -m benchmark.sweep init      --device cpu --output-dir benchmark/results/myrun/
python -m benchmark.sweep run       --output-dir benchmark/results/myrun/   # all 12
python -m benchmark.sweep summarize --output-dir benchmark/results/myrun/

# Incremental work — add one scenario without rerunning the rest:
python -m benchmark.sweep run       --output-dir benchmark/results/myrun/ --scenarios scpdecoup
python -m benchmark.sweep summarize --output-dir benchmark/results/myrun/

# Re-run a single scenario after fixing something:
python -m benchmark.sweep run       --output-dir benchmark/results/myrun/ --scenarios chaboche2 --force
python -m benchmark.sweep summarize --output-dir benchmark/results/myrun/
```

### Subcommand reference

| Subcommand | What it does | Required | Optional |
|---|---|---|---|
| `init` | Create folder; write `metadata.json` with host env + sweep config (warmup, repeats, max-batch, max-seconds, neml2-source). Refuses to overwrite an existing `metadata.json`. | `--device {cpu,cuda}` | `--output-dir`, `--warmup` (5), `--repeats` (20), `--max-batch` (65536), `--max-seconds` (30), `--neml2-source` (`v3-HEAD`) |
| `run` | Run scenarios into a folder previously created by `init`. Skips scenarios with an existing `<scenario>.csv` unless `--force`. Reads warmup/repeats/max-batch/max-seconds from the folder's metadata. | `--output-dir` | `--scenarios elasticity isoharden ...` (default: all 12), `--force` |
| `summarize` | Rebuild `summary.csv` and `metadata.json:sweep.batch_ranges` from the per-scenario CSVs currently on disk. Read-only on per-scenario files. | `--output-dir` | — |
| `all` | Convenience wrapper: `init` + `run --scenarios <all>` + `summarize`. Same args as `init`, plus `--scenarios`. | `--device {cpu,cuda}` | same as `init`, plus `--scenarios` |

When `--output-dir` is omitted from `init` or `all`, the folder name is
auto-generated as
`benchmark/results/<UTC-ts>_<device>_float64_aoti_<host>/`.

### Defaults

The defaults bake into `metadata.json` at `init` time:

* 12 scenarios, 5 warmup + 20 measured calls per (scenario, batch).
* Adaptive batch doubling starting at 1, stopping when **any** of:
  log–log slope ≥ 0.85 for 2 consecutive points (linear regime); per-call
  median > 30 s; CUDA OOM / host MemoryError; batch > 65536 (hard cap).

To change them, `init` a new folder with explicit flags.

## Result folder

Auto-named `benchmark/results/<UTC-ts>_<device>_float64_aoti_<host>/`.
Contents:

| File | What it is |
|---|---|
| `metadata.json` | Host environment snapshot + sweep config + `batch_ranges` + `failed_scenarios`. See **metadata** below. |
| `summary.csv` | Long-format: one row per (scenario, batch) with stats columns. Rebuilt by every `run` and by `summarize`. |
| `<scenario>.csv` | Same columns, one scenario per file. Written atomically by `run`. |
| `<scenario>.runs.csv` | Raw per-call wall time (`nbatch,run_index,time_ms`). |

Stats columns (all in milliseconds):
`scenario, nbatch, n_warmup, n_runs, median_ms, mean_ms, std_ms, min_ms, max_ms, p10_ms, p90_ms`.

Folders are committed alongside source — they're small text files and the
permanent record of every historic perf number.

## Metadata captured

`metadata.json` is split into sections so future readers can diff a run
field-by-field against any other run on the same machine.

| Section | Notable fields |
|---|---|
| `python` | version, executable, implementation |
| `platform` | system, release, machine, libc, hostname, container_indicator |
| `torch` | version, version_cuda, version_cudnn, num_threads, num_interop_threads, allow_tf32_* |
| `env_vars` | OMP/MKL/OPENBLAS/BLIS thread counts, CUDA_VISIBLE_DEVICES, TORCHINDUCTOR_CACHE_DIR |
| `cpu` | model, vendor, cores (logical/physical), governor, min/max/base freq, NUMA nodes, affinity mask |
| `memory` | total_gb, available_gb |
| `cuda` (cuda only) | device_name, compute_capability, total_memory_gb, driver_version, cuda_runtime_version, nvcc_version, `nvidia-smi` snapshot (compute_mode, persistence_mode, power, clocks, temp, ECC) |
| `sweep` | neml2_source, device, dtype, mode (`"aoti"`), warmup, repeats, max_batch, max_seconds, linear_slope_threshold/streak/window, trace_batch, `batch_ranges` per scenario, `failed_scenarios`, `last_run_wall_seconds` |

What's intentionally NOT recorded:
* `pip freeze` — exposes the user's full Python install; not needed for
  reproducing on the same NEML2 commit.
* A separate `neml2.*` section — the NEML2 source is whatever commit the
  results folder is committed at. The `sweep.neml2_source` marker is the
  one exception, so v2 baseline folders sitting alongside v3 runs can be
  told apart.
* Wall-clock timestamps — the folder name carries the init UTC; the
  underlying scenarios are reproducible from the recorded config.

## Reproducibility

Two readers on the same machine running the same NEML2 commit + same
batch shapes get the same numbers within run-to-run noise. The
`metadata.json` shows what setup produced a particular folder; if a
re-run drifts, diff metadata first (most regressions show up as a
governor change, a thread-count change, or a CUDA driver update).

## Comparing v2 vs v3

A v2.1.6 baseline lives alongside the v3 runs under `benchmark/results/`.
The two are told apart by `sweep.neml2_source` inside `metadata.json`
(`"v2.1.6"` vs `"v3-HEAD"`). The folder schema and CSV column layout are
identical, so pandas-joining them on `(scenario, nbatch)` works.

The v2 baseline was generated via a one-time throwaway script and the
canonical v2.1.6 git tag, on the same host the v3 runs were taken on.
v2's `neml2-time` binary reports total wall time across `--num-runs`
iterations only, so v2 rows have `median_ms == mean_ms` and the spread
columns (`std_ms`, percentiles) are NaN. The apples-to-apples column is
`mean_ms`.

If you want to regenerate a v2 baseline yourself:

```bash
# 1. Worktree v2.1.6 (one time)
git worktree add .claude/worktrees/v2.1.6 v2.1.6

# 2. Build v2 in a separate conda env so it doesn't clobber the v3 install
conda create -n neml2-v2 --clone <your-v3-env> -y
/home/thu/.conda/envs/neml2-v2/bin/pip uninstall -y neml2
(cd .claude/worktrees/v2.1.6 && /home/thu/.conda/envs/neml2-v2/bin/pip install -e . -v)

# 3. Drop a throwaway script (the one we used is fossilised in the commit
#    that added the v2 baseline; salvage and edit paths). Run on each device:
/home/thu/.conda/envs/neml2-v2/bin/python /tmp/v2_sweep.py --device cpu
/home/thu/.conda/envs/neml2-v2/bin/python /tmp/v2_sweep.py --device cuda
```

## Adaptive batch stopping

The sweep doubles `nbatch` starting at 1. After each measured point it
fits an OLS line to `log(median_ms)` vs `log(nbatch)` over the trailing
3 points and stops when the slope has stayed ≥ 0.85 for 2 consecutive
points (the device is past saturation and adding more work scales
linearly). The actual stopping batch per scenario is recorded in
`metadata.json:sweep.batch_ranges`.

## Where the slope-detection can mislead

* **Noisy small batches** (sub-10 ms calls): per-call wall time is on the
  order of OS scheduling jitter, so the slope estimate is noisy. The
  sweep tolerates this by requiring two consecutive slopes above the
  threshold; if you see "stop: hit --max-batch" in the log for a
  scenario, the slope detector didn't converge — the actual scaling
  curve is still in `summary.csv`, just not flagged as linear.
* **Phase changes** (one batch shape happens to fit a cache, the next
  doesn't): the slope reads non-monotonic for a couple points then
  recovers. Same mitigation — the run keeps going.
