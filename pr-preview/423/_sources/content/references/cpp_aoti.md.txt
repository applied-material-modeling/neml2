(cpp-aoti)=
# `cpp-aoti` — compiled model from C++

`cpp-aoti` loads an AOT-Inductor package (`.pt2` + metadata) produced by
`neml2-compile` into a C++ host through `libneml2.so` — no Python at the
runtime. It is the production-deployment route: a single shared library, no
NEML2 Python dependency, the same `forward` / `jvp` / `jacobian` surface, and
sub-batch support. To spread one batched call across CPU + GPU(s), wrap it with
a scheduler — see [](model-dispatch).

The artifact format is documented once in [](aoti-packages); the build-system
wiring (CMake / pkg-config / linking) is in [](external-project-integration).
This page is the C++ loading-and-calling API.

## Loading and calling

The entry point that mirrors Python's `load_model(path, name)` is
`neml2::aoti::load_model`: hand it the stub `.i` and the model name, and
it parses the stub, resolves the per-device artifact folder from
`artifact_path`, and forwards any `[Solvers]` settings to the runtime:

```cpp
#include "neml2/csrc/dispatchers/factory.h"

auto model = neml2::aoti::load_model("aoti/elasticity_aoti.i", "elasticity");

auto outputs = model.forward({{"strain", strain_tensor}});
// J is nested: J["stress"]["strain"] is the (*B, *out_base, *in_base) block.
// jacobian()/jvp() return only the pairs the artifact was compiled with
// (`neml2-compile -d OUT:IN`); they throw a FatalError if none were requested.
// A batch-independent block (constant stiffness) is returned unbatched.
auto [outs, J] = model.jacobian({{"strain", strain_tensor}});

// Promoted parameters are mutable in place. Names are fully qualified: a bare
// leaf is wrapped at compile time, so `--model elasticity` exposes `elasticity.E`.
model.named_parameters().at("elasticity.E").fill_(210000.0);
```

`load_model` returns a `Model`-shaped handle. Passing an optional
scheduler turns on multi-device dispatch (chunking a batch across
CPU + GPU(s)); see [](model-dispatch) for the scheduler surface.

If you already have a per-device metadata path and want to skip HIT
parsing entirely, construct the bare `neml2::aoti::Model` directly. It
takes a single filesystem path to a `<device>/<name>_meta.json`; the
per-segment `.pt2` files are resolved relative to that path:

```cpp
#include "neml2/csrc/aoti/Model.h"

neml2::aoti::Model model("aoti/elasticity/cpu/elasticity_meta.json");
```

The bare `Model` is non-copyable and non-movable; hold it as a
`std::unique_ptr` / `std::shared_ptr` or as an automatic on the
stack.

## Errors

Public ops throw the `neml2::aoti` exception taxonomy: `ConvergenceError`
(recoverable — a Newton divergence / max-iters, so a caller can cut the time
step and retry), `FatalError` (non-recoverable — shape/device/config), and
`AggregateError` (concurrent dispatch failures). `Exception::recoverable()`
distinguishes the first from the rest.

## See also

- [](aoti-packages) — the `.pt2` package format this route loads.
- [](external-project-integration) — CMake / pkg-config wiring for the host build.
- [](model-dispatch) — spread a batched call across CPU + GPU(s).
- [](py-aoti) — the same artifact, loaded from Python.
- [](tutorials-models-compiled) — end-to-end how-to.
