(aoti-packages)=
# AOTI packages

This is the on-disk reference for the artifacts that `neml2-compile`
produces and the thin runtime that loads them back. It is aimed at
downstream C++ embedders who need to know exactly what a `.pt2`
package contains.

The *how* — what `neml2-compile` does between reading your HIT
file and emitting these files — is covered in
[](model-compilation-pipeline). The *how-to* — compile a model,
load the result, take Jacobian-vector products — lives in
[](tutorials-models-compiled).

The CLI synopsis:

```
neml2-compile <input.i> --model <name>
                        [--output-dir <dir>]
                        [--device cpu|cuda [cpu|cuda ...]] [--dtype float64|float32]
                        [-p|--parameter NAME ...]
```

See [](cli-utilities) for the broader CLI surface.

## On-disk layout

`neml2-compile` writes a **per-device artifact folder** plus a single
standalone HIT stub that sits next to it:

```text
<output-dir>/                  # default ./aoti/
  <name>_aoti.i                # standalone HIT stub (points at the folder below)
  <name>/                      # per-device artifact folder
    cpu/   <name>_meta.json + *.pt2
    cuda/  <name>_meta.json + *.pt2
```

A compile targeting a single device (the default `--device cpu`) emits
just one `<device>/` subfolder; `neml2-compile --device cpu cuda` emits
both `cpu/` and `cuda/`, each a complete, self-contained artifact for
that device. The stub points at the `<name>/` folder via an absolute
`artifact_path` field, and the loader resolves
`<artifact_path>/<device>/<name>_meta.json` for the device it runs on
(see [The HIT stub](#the-hit-stub)).

Inside each `<device>/` subfolder, a forward single-segment model emits
the metadata plus a value graph and an optional JVP graph:

| File                  | Contents                                                                    |
| :-------------------- | :-------------------------------------------------------------------------- |
| `<name>.pt2`          | AOT-Inductor-compiled forward / value graph.                                |
| `<name>_jvp.pt2`      | Flat-Jacobian graph: returns the outputs plus a stacked `dout/din` block.   |
| `<name>_meta.json`    | Variable layout, dtype, device, promoted-parameter initial values.         |

Implicit single-segment models emit the per-segment set covered in the
segment table below in place of the forward graphs.

For models that contain an `ImplicitUpdate` — or a `ComposedModel`
whose leaves contain one — the export splits at each implicit
boundary into separate **segments**, each numbered `_seg{i}_`:

| File                            | Contents                                                |
| :------------------------------ | :------------------------------------------------------ |
| `<name>_seg{i}.pt2`             | Forward-segment value graph.                            |
| `<name>_seg{i}_jvp.pt2`         | Forward-segment flat Jacobian graph (optional).         |
| `<name>_seg{i}_rhs.pt2`         | Implicit-segment Newton residual `-r(u, g)`.            |
| `<name>_seg{i}_step.pt2`        | Implicit-segment fused assemble + solve + update.       |
| `<name>_seg{i}_ift.pt2`         | Implicit-function-theorem sensitivity `-A^{-1} B`.      |
| `<name>_seg{i}_predictor.pt2`   | Newton initial guess (only if the source had one).      |

The `_seg0_` infix is dropped in the single-segment shortcut, so
single-segment forward artifacts use the names in the first table
and single-segment implicit artifacts use the per-segment names from
the second table without the `_seg0_` prefix.

### What's in the metadata JSON

The metadata is the source of truth: the `.pt2` files are opaque to
NEML2, and re-loading an artifact in a new process never re-introspects
the Python source. It records, at a high level:

- **Device + dtype**, baked into the `.pt2` graphs at export. There
  is no runtime override.
- **Inputs and outputs** — master input/output order with per-variable
  storage size, base shape, and sub-batch shape.
- **Promoted parameters** — the `-p` set, with initial values. Empty
  in the fully-baked case (the artifact is then a frozen inference
  graph and `named_parameters()` is empty at load time).
- **Segments** — one entry per `ImplicitUpdate` boundary the exporter
  split on; executed in order at runtime, with each segment's outputs
  feeding the next segment's inputs via a shared `name → tensor` state
  map.

The exact field layout evolves alongside the export pipeline, so it is
not mirrored field-by-field here. The metadata carries an integer
`schema_version`, bumped on any breaking layout change. The C++ loader
refuses any non-matching version with a clear "regenerate via
`neml2-compile`" message; the only remediation is a re-compile.

<!-- dependencies: aoti.schema_version -->
The current schema version is `5`.

### Segment kinds

Two segment kinds appear inside `segments`:

- **Forward** segments lower to a single value graph (`_seg{i}.pt2`)
  plus an optional flat-Jacobian graph (`_seg{i}_jvp.pt2`). Call shape
  is `(*user_inputs, *promoted_params) -> outputs`.
- **Implicit** segments lower to three graphs — `_rhs.pt2` (Newton
  residual), `_step.pt2` (fused assemble + LU solve + update +
  post-update residual), `_ift.pt2` (`-A^{-1} B`
  implicit-function-theorem sensitivity at the converged state) —
  plus an optional `_predictor.pt2` graph when the source had a
  `Predictor`. The Newton loop body is one loader call per iteration
  plus a convergence sync; the IFT graph is consumed by `jacobian()`
  and `jvp()`.

  The Newton solve's convergence tolerances, iteration cap, and line-search
  settings are **not** baked into the metadata (schema v4+). They are carried
  by the HIT stub's `[Solvers]` block and forwarded to the C++ runtime at load
  time (see [The HIT stub](#the-hit-stub) below). Only the linear solver is
  baked — it lives inside the compiled `_step.pt2` / `_ift.pt2` graphs.

Each segment declares its inputs / outputs / promoted-parameter inputs
in the same per-variable structure as the top-level layout.

### Cross-segment state

At runtime each segment writes its outputs into a shared
`name → tensor` state map, and the next segment reads its declared
inputs from that map. The partitioning rule that decides where
segment boundaries land at export time is documented in
[](model-compilation-pipeline).

## The HIT stub

The `<name>_aoti.i` file is the original input with the
`[Models]/<name>` block surgically replaced by an `AOTIModel` shim.
Every other section (`[Tensors]`, `[Drivers]`, `[Settings]`, …) is
copied through verbatim, so the stub is a drop-in replacement
wherever a `Driver` consumes the model by name. A typical stub:

```ini
# Auto-generated by neml2-compile from input.i.
# Drop-in replacement for the original [elasticity] model.
# Do not edit; regenerate via `neml2-compile`.

[Models]
  [elasticity]
    type = AOTIModel
    artifact_path = '/abs/path/to/aoti/elasticity'
  []
[]
```

The `artifact_path` is an **absolute** path to the per-device artifact
folder (`<output-dir>/<name>/`). The loader appends `<device>/` for the
running device and loads `<artifact_path>/<device>/<name>_meta.json`.
Because the path is absolute and the stub lives outside that folder, the
artifacts are **not relocatable** — moving the folder requires editing
`artifact_path` or recompiling.

The shim has the same surface as a native model — same `input_spec`,
same `output_spec`, same call convention — but inside it dispatches
to the compiled `.pt2` instead of executing the Python `forward`.
Because the surface is identical, anything that consumes a model
through the normal HIT machinery (e.g. a `TransientDriver`) works
without modification.

For a model with an implicit (`ImplicitUpdate`) segment, a **minimal**
`[Solvers]` block is carried and the shim gains a `solver` field pointing
at it (schema v4+). At load the `AOTIModel` shim reads that solver's
convergence / line-search settings and forwards them to the C++ runtime,
so they can be tuned by editing the stub without recompiling:

```ini
[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-12
    rel_tol = 1e-10
    max_its = 25
  []
[]

[Models]
  [model]
    type = AOTIModel
    artifact_path = '/abs/path/to/aoti/model'
    solver = 'newton'
  []
[]
```

Only the knobs that take effect are carried. The `linear_solver` field is
**deliberately omitted**: the linear solver is baked into the compiled
`_step.pt2` / `_ift.pt2` at compile time, so editing it in the stub would
have no effect — leaving it out keeps the stub free of inert controls.
`[EquationSystems]` and `[Data]` are dropped — their state was baked in.

## Loading from Python

Two entry points, depending on what you need. To drive the artifact
through the normal HIT machinery (e.g. a `TransientDriver`), load the
stub with `neml2.load_model` — the `AOTIModel` shim resolves the
per-device subfolder for the current default device:

```python
import neml2

model = neml2.load_model("elasticity_aoti.i", "elasticity")
```

To work against the bare runtime directly — raw-tensor calls, JVP,
Jacobian, promoted-parameter mutation — construct `neml2.aoti.Model`
from a per-device metadata path:

```python
from neml2.aoti import Model

binding = Model("aoti/elasticity/cpu/elasticity_meta.json")
```

Its surface centers on three call paths plus introspection properties
(`input_names`, `output_names`, `input_base_shapes`, `output_base_shapes`,
`device`, `dtype`), a mutable `named_parameters()` map, and a
`set_parameter(name, tensor)` helper for replacing a promoted parameter
wholesale:

| Operation               | Returns                                             |
| :---------------------- | :-------------------------------------------------- |
| `forward(inputs)`       | One tensor per output name, at `(*B, *out_base)`.   |
| `jvp(inputs, tangents)` | `(outputs, J @ v)`, each at `(*B, *out_base)`. Missing tangent keys → zero. |
| `jacobian(inputs)`      | `(outputs, J)` with `J[out][in]` the block `(*B, *out_base, *in_base)`. |
| `named_parameters()`    | Mutable dict of promoted parameters; empty if baked. |

All three call paths take a dict keyed by the master input names
listed in `input_names`; missing keys throw.

```python
# Forward.
out = binding.forward({"strain": strain.data})

# JVP: tangent dict shares keys + (*B, *base) shapes with inputs; missing keys
# default to zero. jvp_out[name] is the directional derivative at (*B, *out_base).
out, jvp_out = binding.jvp({"strain": strain.data}, {"strain": tangent.data})

# Jacobian as unflattened variable-pair blocks: J[out_name][in_name] is
# (*B, *out_base, *in_base) (e.g. SR2->SR2 -> (*B, 6, 6); Scalar->SR2 -> (*B, 6)).
out, J = binding.jacobian({"strain": strain.data})
```

The promoted-parameter map is mutable; the next call sees the new
value:

```python
binding.named_parameters()["E"].fill_(100e3)
```

`inspect`-style diagnostics — variable names, dtypes, shapes — are
available via `neml2-inspect` against the stub, the same way as any
other model (see [](cli-neml2-inspect)).

## Parameter promotion (`-p`)

Every parameter and buffer is **baked** into the lowered graph as a
constant by default. Baked entries are immutable post-compile but
cost nothing per call — they're folded directly into the kernel.
Each `-p NAME` flag promotes one entry to a runtime-flexible
**graph input**:

```bash
neml2-compile input.i --model elasticity -p E
neml2-compile input.i --model viscoplasticity -p hardening.tau0 -p flow_rate.A
```

Names are fully qualified, exactly as `model.named_parameters(recurse=True)`
emits them.

| Aspect                          | Baked (default)                    | Promoted (`-p`)                        |
| :------------------------------ | :--------------------------------- | :------------------------------------- |
| On-disk representation          | Constant inside the `.pt2` graph.  | Initial value in `metadata.parameters`. |
| Runtime mutability              | None — re-compile to change.       | In-place via `named_parameters()`.     |
| Per-call cost                   | Zero (folded into kernel).         | One dict lookup + extra graph input.   |
| Appearance in `named_parameters()` | Absent.                          | Present.                               |

The trade-off: baked is the right default for shipped inference
artifacts; promotion is the escape hatch for training-loop weights,
what-if knobs, and calibration sweeps. If the model was compiled
with no `-p`, `named_parameters()` is empty and the artifact is
effectively a frozen inference graph.

### Constraint: no parameters inside `ImplicitUpdate`

Trying to promote a parameter that lives inside an `ImplicitUpdate`'s
`system.model` tree raises a `NotImplementedError` at compile time,
with a message pointing at the offending name. Parameters in forward
segments of a composed model promote normally; see
[](model-compilation-pipeline) for the underlying constraint on the
equation-system wrappers.

## Dynamic batch dimension

The leading batch dimension is compiled as a dynamic axis. The same
artifact handles any batch size from 1 to roughly a million without
recompilation. The cost is modest extra symbolic-shape machinery
inside the lowered kernel; the benefit is that a single artifact
serves both single-point evaluation (a unit-cell stress update) and
large fan-out runs (a finite-element kernel sweeping thousands of
integration points) with no extra moving parts.

Sub-batch axes — the structured per-site dimensions some models
carry — are baked into the artifact at export time. To change them,
re-compile.

## Device and dtype pinning

The `.pt2` graphs are pinned to the device and dtype they were
exported with, so the artifact does not expose a runtime `to()`:
any move would silently desync the graph from its parameters. To
target a different device or dtype, re-run `neml2-compile` with the
new `--device` / `--dtype`. Promoted parameter tensors are placed
on the same device as the graph at load time.

## C++ runtime

The C++ surface is the same `forward` / `jvp` / `jacobian` triple. The
entry point that mirrors Python's `load_model(path, name)` is
`neml2::aoti::load_model`: hand it the stub `.i` and the model name, and
it parses the stub, resolves the per-device artifact folder from
`artifact_path`, and forwards any `[Solvers]` settings to the runtime:

```cpp
#include "neml2/csrc/dispatchers/factory.h"

auto model = neml2::aoti::load_model("aoti/elasticity_aoti.i", "elasticity");

auto outputs = model.forward({{"strain", strain_tensor}});
// J is nested: J["stress"]["strain"] is the (*B, *out_base, *in_base) block.
auto [outs, J] = model.jacobian({{"strain", strain_tensor}});

// Promoted parameters are mutable in place.
model.named_parameters().at("E").fill_(210000.0);
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

For build-system wiring — `find_package(neml2)`, pkg-config,
`#include` paths, and the runtime library search — see
[](external-project-integration).

## See also

- [](model-compilation-pipeline) — what `neml2-compile` does between
  the HIT file and these artifacts.
- [](tutorials-models-compiled) — end-to-end how-to: compile, load,
  round-trip, JVP, parameter promotion, trade-offs against eager.
- [](cli-utilities) — `neml2-compile` and the rest of the console
  scripts.
- [](cli-neml2-inspect) — inspect a compiled stub the same way you
  inspect any other model.
- [](model-dispatch) — load these artifacts from C++ and spread a
  batched evaluation across CPU + GPU(s).
- [](external-project-integration) — CMake / pkg-config wiring for
  C++ projects that consume the bundled `libneml2.so` from the
  wheel.
