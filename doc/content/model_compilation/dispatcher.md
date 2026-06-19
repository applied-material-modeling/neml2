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
Only CPU and CUDA devices are supported. This release ships the **synchronous**
path: `SimpleScheduler` (one device) and `MPIScheduler` (one GPU per MPI rank).
The asynchronous thread-per-device pool and the multi-device-in-one-instance
hybrid scheduler are planned follow-ups.
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
#include "neml2/csrc/aoti/factory.h"
#include "neml2/csrc/aoti/SimpleScheduler.h"

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

A scheduler decides which device a workload runs on and how large each
sub-batch chunk is. All are plain C++ objects configured by a `Config` struct.

### `SimpleScheduler`

Sends the whole workload to a single device, chunked. `Config{device,
batch_size}` (e.g. `{"cuda:0", 1024}`; `batch_size = 0` means "no chunking"). Use
it to:

- process a batch too large for device memory in fixed-size pieces;
- empirically tune the per-call batch size for a model + device; or
- drive one device per process when a host pins devices by hand.

### `MPIScheduler`

For MPI jobs running one rank per GPU. `Config{devices, batch_sizes}` lists the
CUDA devices to choose from; each rank is assigned one based on its rank *within
its node* (ranks are grouped by hostname, then the local rank indexes into the
list), after which it chunks exactly like `SimpleScheduler`. Requires NEML2 built
with `-DNEML2_MPI=ON` and at least one device per rank per node; otherwise the
constructor throws.

## See also

- [](aoti-packages) — the per-device artifact + metadata layout.
- [](model-compilation) — the compile pipeline overview.
- [](tutorials-models-compiled) — the end-to-end compile-and-load how-to.
