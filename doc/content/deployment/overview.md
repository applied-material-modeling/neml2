(deployment-overview)=
# Ways to evaluate a model

A NEML2 model is authored once, in Python — a `Model` is a plain
`torch.nn.Module` composed from smaller registered pieces, and Python is the
*only* authoring surface. Once a model exists, the same model can be evaluated
through several different runtimes, depending on whether you are iterating
interactively, training, or deploying into a host application.

:::{note}
If you are an *end user* of an application built on NEML2, you evaluate models
through whatever interface that application exposes — you do not choose a runtime
and can stop reading here. This page is for developers integrating NEML2 into
their own Python or C++ workflow, who need to pick the right runtime for the job.
:::

All runtimes operate on the same starting point: a
[HIT](https://github.com/applied-material-modeling/neml2-hit) input file that
names one or more models. The minimal example used below lives at
`tutorials/models/running_your_first_model/input.i`:

```{literalinclude} ../tutorials/models/running_your_first_model/input.i
:language: ini
:caption: input.i
```

## One authoring surface, many runtimes

```{figure} ../../asset/neml2_evaluation_routes.svg
:alt: The six NEML2 evaluation routes as a host × mode matrix
:align: center

The six routes by host (Python / C++) and mode (eager / `torch.compile` / AOTI).
A model is authored once in Python; the empty cell is real (there is no in-process
`torch.compile` route in C++).
```

The four families differ in *where* the model runs and *what it costs to start*.
The AOTI family is one artifact (`.pt2` package + metadata + HIT stub) consumed
by two different hosts — running a compiled model from C++ and from Python is the
same compile, not two.

## At a glance

Each runtime has a short **codename** of the form `host-mode` (`py` / `cpp`
crossed with `eager` / `jit` / `aoti`). Use it in bug reports and discussion
threads to say which path you're on without a paragraph of description — "this
reproduces on `cpp-aoti` but not `py-eager`" is unambiguous.

Every runtime supports `forward`. They differ on sensitivities and on whether
they accept *sub-batch* models (e.g. crystal plasticity, which carries a per-slip
inner batch dimension):

| Codename | Entry point | Compile | Host | jvp / jacobian | Sub-batch | Primarily for |
|---|---|---|---|:--:|:--:|---|
| `py-eager` | `neml2.load_model` | none | Python | ✓ | ✓ | dev, testing, autograd training |
| `py-jit` | `neml2.compile` | in-process JIT | Python | ✓ (native) | ✓ | pyzag training loops |
| `py-aoti` | `neml2.aoti.Model` | offline | Python (pybind) | ✓ | ✓ | compiled model from Python |
| `cpp-aoti` | `neml2::aoti::Model` | offline | C++ | ✓ | ✓ | production C++ deployment |
| `cpp-dispatch` | `neml2::aoti::DispatchedModel` | offline | C++ | ✓ | ✓ | multi-device throughput |
| `cpp-eager` | `neml2::eager::Model` | none | C++ + embedded Python | ✓ | ✗ | compile-free C++ tests |

## `py-eager` — eager Python

The interactive path. `pip install neml2`, load the model with one call, and
evaluate it like any `torch.nn.Module`:

```python
import neml2
from neml2.types import SR2

model = neml2.load_model("input.i", "elasticity")
stress = model(SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
```

Inputs and outputs are typed tensor wrappers (`Scalar`, `SR2`, `R2`, …) from
`neml2.types`. There is no compile step; the model runs eager PyTorch and
supports the full `forward` / `jvp` / `jacobian` surface plus autograd, on any
device, including sub-batch models. This is the path you reach for during
development, in tests, and when training with PyTorch autograd. The line-by-line
walkthrough is in [](tutorials-models-running-your-first-model).

Most shell and Python entry points are layered on top of this runtime — see
[Runtimes vs. consumers](#runtimes-vs-consumers) below.

## `py-jit` — in-process `torch.compile`

The same eager model, accelerated in the running interpreter:

```python
import neml2

model = neml2.load_model("input.i", "elasticity")
neml2.compile(model)   # in place; the object is returned, now compiled
```

`neml2.compile` wraps the model's feed-forward `forward` in `torch.compile`. It
produces no artifact — the compiled graph lives in the interpreter and recompiles
lazily once per distinct input shape. Sensitivities still flow through the native
chain rule. It accepts a `Model`, a `ModelNonlinearSystem`, or a pyzag wrapper,
and exists primarily to speed up pyzag training loops (the residual / Jacobian
assembly). See [](tutorials-optimization-pyzag).

## Ahead-of-time compilation (AOTI)

The deployment path. Once a model is locked in, `neml2-compile` lowers it through
`torch.export` + AOT-Inductor into a self-contained artifact set — one or more
`.pt2` graphs, a metadata sidecar, and a drop-in HIT stub:

```bash
neml2-compile input.i --model elasticity
```

This writes compiled graphs and metadata into `aoti/elasticity/<device>/` (one
subfolder per target device) and drops a standalone stub `aoti/elasticity_aoti.i`
next to it. The artifact loads quickly, skips the HIT parser on every call, and
carries no NEML2 Python dependency at the C++ runtime. The package format itself
is documented in [](aoti-packages); the internals of the lowering are in
[](model-compilation-pipeline).

The *same* artifact is consumed by two hosts:

### `cpp-aoti` — from C++

`neml2::aoti::Model` (in `libneml2.so`) loads the package and exposes
`forward` / `jvp` / `jacobian` keyed by the structural names recorded in the
metadata. Wire it into your build with CMake or `pkg-config` —
[](external-project-integration).

### `cpp-dispatch` — across multiple devices

`neml2::aoti::DispatchedModel` wraps one pinned `aoti::Model` per device and an
injected `WorkScheduler`, chunking a batched call across CPU + GPU(s) and
stitching the results back. Schedulers range from a single-device chunk loop
(`SimpleScheduler`, `MPISimpleScheduler`) to a concurrent thread-per-device pool
(`StaticHybridScheduler`). This is a multi-device configuration of the C++ path,
not a separate compile — see [](model-dispatch).

### `py-aoti` — from Python

`neml2.aoti.Model` binds the same C++ runtime through pybind, and the
`AOTIModel` HIT shim lets a driver or input file load a compiled model in place
of a native one. Use this to run a compiled model from Python without re-exporting,
or to reproduce C++ numerics from a notebook. See [](aoti-packages).

## `cpp-eager` — C++ embedded eager Python

`neml2::eager::Model` runs a model straight from its original `.i` with **no
compile step**: it embeds a CPython interpreter (in the separate
`libneml2_eager.so`), imports the `neml2.eager._EagerModel` adapter, and marshals
tensors across the boundary. It mirrors `neml2::aoti::Model`'s
`forward` / `jvp` / `jacobian` surface, so it is a drop-in for the AOTI C++ path —
fast to start, slow to run. It is intended for downstream C++ unit tests that
cannot afford an Inductor compile on every build.

This runtime is **plain-batch only**: a model that produces sub-batch output
(e.g. crystal plasticity) is rejected, because there is no slot to declare
per-input sub-batch shapes at this boundary. See [](model-eager-cpp).

## Choosing a runtime

- **Iterating, debugging, or training in Python** → `py-eager`. Reach for
  `py-jit` (`neml2.compile`) only inside a pyzag loop where the residual is the
  hot path.
- **Deploying into a C++ application** → `cpp-aoti`. Switch to `cpp-dispatch`
  when you need to saturate multiple GPUs (or CPU + GPU) with one batched call.
- **Calling a compiled model from Python**, or reproducing C++ numerics without a
  NEML2 source dependency → `py-aoti`.
- **C++ tests that can't pay the compile cost** → `cpp-eager`.
- **Sub-batch models** (crystal plasticity and friends) run on every runtime
  *except* `cpp-eager`. If you need a compiled sub-batch model in C++, use
  `cpp-aoti`.

(runtimes-vs-consumers)=
## Runtimes vs. consumers

Several entry points are not runtimes themselves — they are *consumers* layered
on one of the runtimes above:

- `neml2-run <input.i> <driver>` and the `Driver` classes (`TransientDriver`,
  `ModelUnitTest`, `TransientRegression`, `Verification`) step a model through a
  load history on `py-eager`. If the input file names an `AOTIModel`, the same
  driver runs on `py-aoti` instead.
- The **pyzag** adapter (`NEML2PyzagModel`) drives calibration on `py-eager`,
  optionally accelerated to `py-jit` via `neml2.compile`.
- `neml2-inspect <input.i> <model>` resolves and prints a model's input/output
  graph but does **not** evaluate it.

The full tool reference is in [](cli-utilities).

## See also

- [](aoti-packages) — the on-disk AOTI package format and loading API.
- [](model-dispatch) — the multi-device work scheduler / dispatcher.
- [](model-eager-cpp) — the C++ embedded-eager runtime in detail.
- [](external-project-integration) — CMake / `pkg-config` wiring for C++ hosts.
- [](model-compilation-pipeline) — what `neml2-compile` does, stage by stage.
- [](tutorials) — end-to-end walkthroughs, including the compiled-model round trip.
- [](migration-guides) — what changed across NEML2 versions.
