(benchmark)=
# Benchmarking

The `benchmark/` directory ships a curated set of representative
material models together with a sweep harness and a
vectorisation-efficiency fitter. The tooling answers the question
that comes up whenever a NEML2 user sizes a production run:

> *How does compute time scale with batch size for a given model on a
> given device?*

The same tooling is the substrate for tracking performance changes
across NEML2 releases; runtime-touching PRs are expected to keep the
headline numbers from regressing.

## Scenarios

The committed scenarios cover the breadth of the solid-mechanics
module, from trivial elasticity to fully-coupled crystal plasticity:

| Code name        | Description                                                    |
| :--------------- | :------------------------------------------------------------- |
| `elasticity`     | Linear isotropic elasticity without model composition          |
| `radret`         | $J_2$ viscoplasticity with radial-return integration           |
| `isoharden`      | Viscoplasticity with isotropic hardening                       |
| `chaboche2`      | Chaboche viscoplasticity, 2 back stresses                      |
| `chaboche4`      | Chaboche viscoplasticity, 4 back stresses                      |
| `chaboche6`      | Chaboche viscoplasticity, 6 back stresses                      |
| `gtntheig`       | GTN visco-poroplasticity with thermal eigenstrain              |
| `scpcoup`        | Fully-coupled implicit single-crystal plasticity               |
| `scpcoupmult`    | Coupled SCP with multiplicative $F = F^e F^p$ decomposition    |
| `scpdecoup`      | Decoupled implicit single-crystal plasticity                   |
| `scpdecoupexp`   | Decoupled SCP with explicit orientation update                 |
| `tcpsingle`      | Taylor polycrystal with a single orientation                   |
| `tcprandom`      | Taylor polycrystal with pseudo-random FCC orientations         |
| `mxpc`           | Mixed-control polycrystal: ``nbatch`` scales the *sub-batch* (grain) axis with a shared global mixed-control state, solved via `SchurComplement` |

Each scenario lives under `benchmark/<name>/model.i` as an HIT input
file with two HIT-substituted parameters: `${nbatch}` (the dynamic
batch size) and `${device}` (`cpu` or `cuda`). All scenarios except
`tcprandom` advance 100 prescribed time steps; `tcprandom` advances
500 to simulate severe plastic deformation from the standard rolling
experiment.

## Vectorisation-efficiency model

For a fixed device and model, compute time as a function of batch
size $N$ is fitted to ANL-24/43 eq 7.1:

$$
t(N;\, t_0,\, N^*) = t_0 \left( \frac{\langle N - N^* \rangle}{N^*} + 1 \right)
$$

where $\langle x \rangle = \max(x, 0)$ is the Macaulay bracket,
$t_0$ is the per-call cost when the device's vectorisation
bandwidth is not yet saturated (a constant time across all small
batches), and $N^*$ is the **optimal batch size** at which the
device starts trading constant time for linear-in-$N$ time.

Two limits of the model are intuitive:

- $N \ll N^*$: $t \to t_0$. Vectorisation lanes are unsaturated;
  per-call cost is dominated by launch overhead, fixed setup, and
  whatever per-step constant work the model carries. Increasing
  batch costs nothing.
- $N \gg N^*$: $t \to t_0 \cdot N / N^*$. The device is saturated;
  per-call cost scales linearly with the batch.

The fitted $(t_0,\, N^*)$ pair characterises the model+device pair
in one of two useful ways:

- On a fixed device, $t_0$ sets the floor on per-call cost — every
  measurement at $N \le N^*$ sits there, so a smaller $t_0$ means a
  cheaper baseline.
- For a fixed model, $N^*$ is the largest batch that still pays only
  $t_0$, so a larger $N^*$ means more useful work per call before the
  device falls out of the free region.

The **peak throughput** of a model+device pair is

$$
\mathrm{throughput} = \frac{N^*}{t_0}
$$

(material updates per ms), since by construction $t(N^*) = t_0$. This
is the right single-number summary when comparing two runtimes — pick
each runtime's own optimal batch, and the user can run the model at
that batch to get the headline rate. Comparing at a fixed batch can
flatter or penalise either side depending on whether that batch
happens to land in one runtime's free region.

## Running the sweep

The sweep tool is invoked through `python -m benchmark.sweep`:

```bash
# Init a fresh result folder (just records sweep config + env snapshot)
python -m benchmark.sweep init --device cuda

# Run every scenario into that folder (or one folder at a time)
python -m benchmark.sweep run --output-dir benchmark/results/<TS>_cuda_float64_aoti

# Rebuild summary.csv from per-scenario CSVs
python -m benchmark.sweep summarize --output-dir <DIR>

# Fit eq 7.1 per scenario into fitting.csv
python -m benchmark.sweep fit --output-dir <DIR>

# End-to-end shortcut (init + run every discovered scenario + summarize)
python -m benchmark.sweep all --device cuda
```

Per-scenario stop conditions trip the inner adaptive-batch loop
whichever comes first: (a) log-log slope $\ge 0.85$ for two
consecutive points (the model has entered its linear regime so
more data points add little information), (b) measured median call
time exceeds 30 s (absolute wall-time saturation), (c) CUDA OOM /
host MemoryError, or (d) batch exceeds the `--max-batch` cap
(default 65536).

Each batch is compiled separately, with the scenario's own
`[Settings]/[example_batch_shape]` block resolved at `nbatch=B`. For
most scenarios that block is just `'(${nbatch},)'`, so the trace
shape ends up `(B,)`; mxpc declares per-input shapes so its grain
sub-batch stays distinct from the dynamic-batch axis. This matters:
Inductor's per-kernel autotune block-size hint is seeded from the
example shape and the optimal hint is workload-dependent. Reusing a
single compile across batches mirrors the cost of an ahead-of-time
deployment but can mis-report scaling behaviour at batches far from
the example. Per-batch compile is the methodology a production user
gets from `neml2-compile` invoked at the batch they actually intend
to run.

### Output

Each `run` populates four file kinds under `<output-dir>`:

| File                       | Contents                                                  |
| :------------------------- | :-------------------------------------------------------- |
| `metadata.json`            | Host + GPU snapshot, sweep config, per-scenario batch ranges, last-run wall time |
| `<scenario>.csv`           | One row per measured batch: median / mean / std / min / max / p10 / p90 / RSS / peak CUDA memory |
| `<scenario>.runs.csv`      | Raw per-iteration timings (warmup discarded, all measurement runs kept) |
| `summary.csv`              | Aggregate view across every scenario, regenerated by `summarize` |
| `fitting.csv`              | Per-scenario $(t_0, N^*)$ from `fit`, plus log-space RMS residual and number of fit points |

The CUDA compile pipeline saturates available CPU cores while
running; co-locating a CPU sweep on the same box double-occupies
the cores and taints both. Run one device at a time on a quiet
machine. The harness itself does not detect contention.
