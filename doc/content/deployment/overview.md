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
their own Python, C++, or command-line workflow. The deployment guides
([](python-integration), [](external-project-integration), [](cli-utilities))
cover getting set up; the reference pages linked below cover each route's
evaluation API.
:::

All runtimes operate on the same starting point: a
[HIT](https://github.com/applied-material-modeling/neml2-hit) input file that
names one or more models. The minimal example referenced throughout lives at
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

**Parameter derivatives.** All six routes additionally expose the same
`param_jacobian(inputs) -> (outputs, {out: {param: block}})` and
`param_vjp(inputs, cotangents) -> {param: grad}` surface for `d(output)/d(parameter)`,
computed by reverse-mode autograd — a separate path from the forward-mode input
chain rule that backs `jvp` / `jacobian`. The eager routes (`py-eager`,
`py-jit`, `cpp-eager`) differentiate w.r.t. every parameter at call time
(`params=` selects a subset); the AOTI routes (`py-aoti`, `cpp-aoti`,
`cpp-dispatch`) return only the `(output, parameter)` pairs whose graphs were
compiled in — promote the parameter with `-p NAME` and request the pair with
`-d OUT:NAME` at `neml2-compile` time (see [](aoti-packages)).

**Parameter values at runtime.** All six routes also share a parameter
read/write surface: `named_parameters()` returns the current values, and
`set_parameter(name, value)` updates one in place (a `torch.no_grad` copy on the
eager routes; a slot replacement on the AOTI routes, broadcast across devices for
`cpp-dispatch`). The new value is used on the next call.

**Batched (per-batch-element) parameters.** `set_parameter` also accepts a value
that carries a leading batch dimension — e.g. a `Scalar` parameter set to shape
`(B,)`, a spatially varying material property and the common MOOSE
inverse-optimization case. The call batch is then
`broadcast(input batches, parameter batches)`, and all of `forward` / `jvp` /
`jacobian` / `param_jacobian` / `param_vjp` return per-batch-element results on
every route (`cpp-dispatch` slices each batched parameter to each chunk's rows).
Both parameter-derivative surfaces follow the parameter's shape, matching the
eager semantics: `param_jacobian` returns a `d(output)/d(parameter)` block whose
trailing axes are `(*param_base)` (per batch element when the parameter is
batched), and `param_vjp` returns the adjoint `dL/d(param)` **summed over the
batch for a global (unbatched) parameter** but **per-element `(B, *param_base)`
for a batched one** — the latter being what reverse-mode autograd accumulates when
each batch element depends only on its own copy of the parameter. (`cpp-dispatch`
concatenates each batched parameter's per-chunk adjoint and sums each global one.)

## The routes

Each route has its own reference page with the loading-and-calling API; the
deployment guides cover the setup (install / build / artifacts) that comes first.

**Python** — set up with [](python-integration):

- [](py-eager) — load and call the model directly; the default for development,
  interactive work, and autograd training.
- [](py-jit) — `neml2.compile` accelerates the in-process graph, mainly for pyzag.
- [](py-aoti) — load and run a compiled `.pt2` package from Python.

**C++** — set up with [](external-project-integration):

- [](cpp-aoti) — load a compiled `.pt2` package via `libneml2.so`.
- [](model-dispatch) — the same artifact, chunked across CPU + GPU(s).
- [](model-eager-cpp) — run a model from its `.i` with no compile (for C++ tests).

The three compiled routes (`py-aoti`, `cpp-aoti`, `cpp-dispatch`) share one
artifact — see [](aoti-packages) for its format and
[](model-compilation-pipeline) for how `neml2-compile` produces it. The command
line is a fourth way to drive a model with no code at all: [](cli-utilities).

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
- The **pyzag** adapter (`NEML2PyzagFactory`) drives calibration on `py-jit` by
  default -- it compiles the residual model in place via `neml2.compile` on
  construction -- and falls back to `py-eager` when constructed with
  `compile=False` (a correctness oracle).
- `neml2-inspect <input.i> <model>` resolves and prints a model's input/output
  graph but does **not** evaluate it.

The full tool reference is in [](cli-utilities).

## See also

- [](python-integration) / [](external-project-integration) / [](cli-utilities) —
  set up neml2 in a Python app, a C++ build, or from the shell.
- [](aoti-packages) — the compiled-package format shared by the AOTI routes.
- [](model-compilation-pipeline) — what `neml2-compile` does, stage by stage.
- [](tutorials) — end-to-end walkthroughs.
- [](migration-guides) — what changed across NEML2 versions.
