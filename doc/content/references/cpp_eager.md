(model-eager-cpp)=
# Eager evaluation from C++

The compiled path ([](aoti-packages)) is the right way to run a NEML2 model
from C++ in production: it is fast and needs no Python at the hot loop. But
`neml2-compile` runs an AOT-Inductor compile that takes **minutes** — too slow to
sit in a downstream C++ project's unit tests, where you just want to load a model
and check a few values.

The **eager C++ runtime** fills that gap. `neml2::eager::Model` embeds a CPython
interpreter, loads the model in Python via `neml2.factory.load_model` (the very
same path `neml2-run` uses), and evaluates `forward` by marshalling `at::Tensor`s
across the C++/Python boundary. There is **no compile step and no artifact** — it
consumes the *original* `.i` file directly. The trade-off is speed: every call
goes through Python, so it is far slower than a compiled graph. Use it for
fast-to-start tests; use the compiled / dispatched path ([](model-dispatch)) for
production.

:::{note}
This runtime lives in a **separate** shared library, `libneml2_eager.so`. The
main `libneml2.so` (the AOTI runtime + dispatcher) stays Python-free so it can
embed in pure-C++ hosts; link `libneml2_eager` only where you want the eager
fallback. `libneml2_eager` itself does **not** link libpython — like a CPython
extension module, it leaves the Python C-API symbols undefined and resolves them
at load time from the host's interpreter. A pure-C++ host must therefore link
libpython itself (e.g. CMake `Python3::Python` from
`find_package(Python3 COMPONENTS Development.Embed)`); a host that already runs
Python supplies it automatically.
:::

## Load and run from C++

`neml2::eager::load_model(input_file, model_name)` mirrors Python's
`load_model(path, name)` and returns a `neml2::eager::Model` whose operations
(`forward` / `jvp` / `jacobian`) and metadata accessors (`input_names` /
`output_names` / `input_base_shapes` / `output_base_shapes` / `device` / `dtype`)
have the **same signatures** as `neml2::aoti::Model`. So test code switches between
the eager and compiled runtimes by changing only the load call and the header.

```{literalinclude} ../../../tests/aoti/forward_single/model.i
:language: ini
:caption: tests/aoti/forward_single/model.i — loaded directly, no compile
```

```cpp
#include "neml2/csrc/eager/load_model.h"

using namespace neml2::eager;

// Loads the model named "model" straight from the .i by embedding Python.
auto m = load_model("tests/aoti/forward_single/model.i", "model");

// Build each input at its canonical (*B, *base_shape) shape, on the model's
// device + dtype (NEML2 runs in float64). For SR2 strain the base shape is {6}.
const auto opts = at::TensorOptions().dtype(m.dtype()).device(m.device());
std::map<std::string, at::Tensor> inputs;
for (std::size_t i = 0; i < m.input_names().size(); ++i)
{
  std::vector<int64_t> shape{8};                       // batch
  const auto & base = m.input_base_shapes()[i];        // e.g. {6} for SR2, {} for Scalar
  shape.insert(shape.end(), base.begin(), base.end());
  inputs.emplace(m.input_names()[i], at::randn(shape, opts));
}

auto out = m.forward(inputs);   // {"stress": (8, 6) tensor}
```

The reported `input_names` / `input_base_shapes` (and the outputs) are derived
through the same helper the AOTI metadata uses, so an eager model and its compiled
twin agree on the boundary contract — making `neml2::eager::Model` a true drop-in
for `neml2::aoti::Model`. Inputs must be passed at their canonical
`(*B, *base_shape)` shape; a non-canonical trailing shape (e.g. an SR2 strain
passed as `(*B, 1)` instead of `(*B, 6)`) is rejected with a `FatalError`.

## Sensitivities: `jvp` and `jacobian`

`jvp` and `jacobian` mirror `neml2::aoti::Model` and are computed on the native
model's chain rule (the same `v=` machinery the AOTI export traces), so they work
for forward and implicit (Newton) models alike. Both are **variable-native**: the
boundary works in variables, not flattened group vectors.

```cpp
// Jacobian-vector product: jvp_out[name] is the directional derivative at the
// output's natural (*B, *out_base) shape.
auto [out, jvp_out] = m.jvp(inputs, tangents);   // tangents keyed + shaped like
                                                 // inputs; a missing key is zero.

// Full Jacobian as unflattened variable-pair blocks: J[out_name][in_name] is
// (*B, *out_base, *in_base) -- e.g. SR2->SR2 -> (*B, 6, 6); Scalar->SR2 -> (*B, 6).
auto [out2, J] = m.jacobian(inputs);
```

:::{warning}
**Plain-batch only.** The raw-tensor `forward(map)` boundary has no slot to
declare per-input *sub-batch* shapes (in NEML2 those are caller-declared at
compile time), so the eager runtime supports only plain-batch models. A model
that carries BLOCK-aware / labelled sub-batch axes (e.g. crystal-plasticity
geometry) is **rejected** with a `neml2::aoti::FatalError` rather than silently
mishandled — use the compiled / dispatched path ([](model-dispatch)) for those.
:::

## Errors

A Python-side failure — a parse error, an unknown model name, a wrong-dtype
input, a sub-batch model — is normalized to `neml2::aoti::FatalError` (the same
taxonomy the AOTI runtime uses; see [](aoti-packages)). The one exception is a
**solver divergence / max-iterations**, which is re-raised as the *recoverable*
`neml2::aoti::ConvergenceError` (it originates as that type inside `libneml2.so`
and round-trips faithfully through the embedded interpreter), so a host can cut
the time step and retry:

```cpp
try { auto out = m.forward(inputs); }
catch (const neml2::aoti::Exception & e) {
  if (e.recoverable()) { /* cut dt, retry */ }
  else throw;
}
```

The exception types are exported from `libneml2.so`, so a
`catch (const neml2::aoti::Exception &)` works across the two libraries.

## Requirements

Because the model is loaded *in Python*, the host process must be able to import
the `neml2` package and its dependencies:

- A usable CPython interpreter. `neml2::eager::Model` starts one on first use
  (and never finalizes it — PyTorch does not support a clean re-import after
  `Py_Finalize`). If the host is *already* a Python process, the existing
  interpreter is reused and its lifecycle is left untouched.
- The `neml2` package importable from that interpreter (e.g. on `PYTHONPATH`, or
  installed in the interpreter's environment).

Each call holds the GIL, so concurrent `forward` calls from multiple host threads
are serialized. That is fine for unit tests; for parallel throughput use the
compiled / dispatched runtime ([](model-dispatch)).
