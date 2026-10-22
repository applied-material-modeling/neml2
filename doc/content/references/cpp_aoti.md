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
it parses the stub, resolves the artifact folder from `artifact_path`,
and reads solver configuration directly from `metadata.json`:

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

If you want to skip HIT parsing entirely, construct the bare
`neml2::aoti::Model` directly from the artifact root. Device and dtype
default to `at::kCPU` and `at::kDouble`:

```cpp
#include "neml2/csrc/aoti/Model.h"

neml2::aoti::Model model("aoti/elasticity");
// or with explicit device / dtype:
neml2::aoti::Model model("aoti/elasticity", at::kCPU, at::kDouble);
```

The bare `Model` is non-copyable and non-movable; hold it as a
`std::unique_ptr` / `std::shared_ptr` or as an automatic on the
stack.

## The tensor contract

Every input and output tensor crossing this boundary is laid out as

```
(*B, *sub_batch_shape, *base_shape)
```

`*B` is the leading dynamic batch, shared across variables; `base_shape` is the
variable's type shape (`{}` for a Scalar, `{6}` for an SR2, `{3,3}` for an R2);
`sub_batch_shape` is the variable's structural per-site extent (per-grain,
per-slip-system, per-bin) and is empty for a plain-batch variable. See
[](sub-batch-declaration) for how the extent is declared.

`input_base_shapes()` and `input_sub_batch_shapes()` report the two trailing
regions, both parallel to `input_names()`:

```cpp
// Size input i without having a tensor for it yet.
std::vector<int64_t> shape{8};                            // batch
const auto & sub = model.input_sub_batch_shapes()[i];     // e.g. {100}, or {} if plain
const auto & base = model.input_base_shapes()[i];         // e.g. {6} for SR2
shape.insert(shape.end(), sub.begin(), sub.end());
shape.insert(shape.end(), base.begin(), base.end());
```

The runtime materializes every input to this full shape before the call (a
size-1 sub-batch axis is broadcast to its declared extent), so each batch entry
of an input is a contiguous block of `prod(sub_batch_shape) * prod(base_shape)`
values. Outputs are returned as the compiled graph produced them; the same
guarantee for outputs comes with `output_sub_batch_shapes()`. `DispatchedModel` and
`neml2::eager::Model` expose the same two accessors with identical signatures.

There is no `output_sub_batch_shapes()` yet. `metadata.json` carries a
`sub_batch_shape` for inputs only, so an output's extent is not known before the
first call; a host that must handle sub-batched outputs should reject them for
now.

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
