(model-dispatch)=
# Dispatching across devices

Modern compute nodes are heterogeneous: one or more CPUs alongside one or more
GPUs. The **work scheduler / dispatcher** lets a single compiled model spread a
large batched evaluation across devices — slice the batch into sub-batches, run
each on a device, and stitch the results back together.

This is a **C++ runtime** feature: it serves the compiled
([](aoti-packages)) path embedded in a host application (e.g. MOOSE), where the
hot loop runs without Python. The Python authoring path (`neml2.load_model`,
`neml2-run`) stays eager and single-device — to spread work from Python, run
your own per-device loop.

:::{note}
Only CPU and CUDA devices are supported. Two scheduling modes are available:
**synchronous** single-device (`SimpleScheduler`, `MPISimpleScheduler`) and
**asynchronous** multi-device (`StaticHybridScheduler`, which runs CPU + GPU(s)
concurrently via a thread-per-device pool).
:::

## Compile one artifact per device

AOTI graphs are pinned to a device at export time, so a dispatcher that targets
several devices needs one artifact per device. `neml2-compile` takes multiple
`--device` values and emits them side by side:

```{literalinclude} ../../../tests/aoti/forward_single/model.i
:language: ini
:caption: tests/aoti/forward_single/model.i — the model to compile
```

```console
$ neml2-compile tests/aoti/forward_single/model.i --model model --device cpu cuda
```

produces a standalone stub next to a per-device artifact folder:

```text
aoti/
  model_aoti.i            # standalone stub; points at the folder via artifact_path
  model/
    cpu/   model_meta.json + *.pt2
    cuda/  model_meta.json + *.pt2
```

The loader resolves `<artifact_path>/<device>/` for the device it runs on. The
Python shim picks the subfolder matching `torch.get_default_device()` (so
`neml2-run --device cuda` loads `cuda/`); the C++ loader picks it from the
scheduler (below).

## Load and dispatch from C++

`neml2::aoti::load_model` mirrors Python's `load_model(path, name)` and returns a
`DispatchedModel` — a `Model`-shaped handle exposing the same
`forward` / `jvp` / `jacobian`. The optional scheduler is the dispatch opt-in; it
is supplied in C++ source, never from the `.i`.

```cpp
#include "neml2/csrc/dispatchers/factory.h"
#include "neml2/csrc/dispatchers/SimpleScheduler.h"

using namespace neml2::aoti;

// No scheduler -> no dispatch: runs the whole batch on cpu (zero-overhead
// pass-through over the underlying Model).
auto m = load_model("aoti/model_aoti.i", "model");
auto out = m.forward(inputs);

// With a scheduler -> the batch is chunked along its leading axis, each chunk
// moved to the compute device, run, and the results concatenated back on the
// input device.
auto sched = std::make_shared<SimpleScheduler>(SimpleScheduler::Config{"cuda", 4096});
auto md = load_model("aoti/model_aoti.i", "model", sched);
auto out_gpu = md.forward(inputs);   // cpu inputs -> gpu compute -> cpu results
```

When the scheduler's device equals the input device and the whole batch fits in
one chunk, `DispatchedModel` short-circuits to a direct `Model` call — so the
no-dispatch case carries no slicing or transfer cost.

## Schedulers

A scheduler decides which device(s) a workload runs on and how large each
sub-batch chunk is. All are plain C++ objects configured by a `Config` struct.
`DispatchedModel` picks its execution mode from the scheduler's type: a
**synchronous** scheduler (`SimpleScheduler`, `MPISimpleScheduler`) runs the
chunk loop on the calling thread; an **asynchronous** one
(`StaticHybridScheduler`) drives a thread-per-device pool.

### `SimpleScheduler`

Sends the whole workload to a single device, chunked. `Config{device,
batch_size}` (e.g. `{"cuda:0", 1024}`; `batch_size = 0` means "no chunking"). Use
it to:

- process a batch too large for device memory in fixed-size pieces;
- empirically tune the per-call batch size for a model + device; or
- drive one device per process when a host pins devices by hand.

### `MPISimpleScheduler`

For MPI jobs running one rank per GPU. `Config{devices, batch_sizes}` lists the
CUDA devices to choose from; each rank is assigned one based on its rank *within
its node* (ranks are grouped by hostname, then the local rank indexes into the
list), after which it chunks exactly like `SimpleScheduler`. Requires NEML2 built
with `-DNEML2_MPI=ON` and at least one device per rank per node; otherwise the
constructor throws.

### `StaticHybridScheduler`

Spreads one batch across several devices **concurrently** — a single
`DispatchedModel` runs CPU + GPU(s) at once via a thread-per-device pool.
`Config{devices, batch_sizes, capacities, priorities}` (the last two optional;
each broadcasts from length 1). Assignment is greedy: each chunk goes to the
highest-`priority` device that still has spare `capacity`
(`load + batch_size <= capacity`), so faster devices stay filled; `capacity`
controls how many chunks may be in flight per device (overlapping the next
chunk's host→device copy with the current chunk's compute).

```cpp
#include "neml2/csrc/dispatchers/StaticHybridScheduler.h"

StaticHybridScheduler::Config cfg;
cfg.devices     = {"cpu", "cuda:0", "cuda:1"};
cfg.batch_sizes = {512, 4096, 4096};   // tune per device, e.g. via SimpleScheduler
auto m = load_model("aoti/model_aoti.i", "model",
                    std::make_shared<StaticHybridScheduler>(cfg));
auto out = m.forward(inputs);          // dispatched across all three, gathered back
```

A hybrid pool admits **at most one CPU** plus distinct GPUs: each device's AOTI
graph already saturates torch's intra-op (OpenMP) thread pool, so two CPU
workers would only oversubscribe the same cores.

**Promoted parameters under hybrid.** `named_parameters()` is a single *master*
map; mutating it in place is broadcast to every device copy before the next
dispatch, so the usual single-device idiom keeps working:

```cpp
m.named_parameters().at("E").fill_(150e3);  // reflected on every device next call
```

## Error handling

Every exception that leaves `forward` / `jvp` / `jacobian` — on both the
synchronous and asynchronous paths — is a `neml2::aoti::Exception` (itself a
`std::runtime_error`) carrying a `recoverable()` flag. That flag is the contract
a downstream consumer branches on:

- **`ConvergenceError`** (`recoverable() == true`) — the nonlinear solve
  diverged or hit its iteration cap. A time-stepping consumer can cut the step
  and retry.
- **`FatalError`** (`recoverable() == false`) — a shape / device mismatch, a
  missing input, a malformed artifact. A retry would fail identically, so it
  must hard-fail. Foreign errors (a torch `c10::Error`, `std::bad_alloc`, ...)
  are normalized to this at the boundary, so a single `catch` covers everything.

```cpp
try
{
  auto out = m.forward(inputs);
}
catch (const neml2::aoti::Exception & e)
{
  if (e.recoverable()) { /* e.g. dt *= 0.5; retry */ }
  else                 { throw; } // fatal: give up
}
```

Under asynchronous dispatch this stays well-defined even when several chunks run
at once. A failing chunk is caught inside its worker (a C++ exception escaping a
`std::thread` would call `std::terminate`), the scheduler is still drained so the
pool can never deadlock, and only then does the dispatcher decide what to throw:

- one failure → it is re-thrown verbatim (its dynamic type, e.g.
  `ConvergenceError`, is preserved);
- several at once → an **`AggregateError`** carrying them all. It reports
  `recoverable()` only if *every* sub-error is recoverable, so a lone fatal among
  otherwise-recoverable failures still forces a hard stop. The individual errors
  are available via `AggregateError::errors()`.

Either way the `DispatchedModel` and its scheduler are left clean and reusable
for the next call.

## See also

- [](aoti-packages) — the per-device artifact + metadata layout.
- [](model-compilation) — the compile pipeline overview.
- [](tutorials-models-compiled) — the end-to-end compile-and-load how-to.
