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
                        [--device cpu|cuda] [--dtype float64|float32]
                        [-p|--parameter NAME ...]
```

See [](cli-utilities) for the broader CLI surface.

## The four artifacts

For a single-segment model (pure forward or pure implicit, the common
case), `neml2-compile` emits exactly four files:

| File                  | Contents                                                                    |
| :-------------------- | :-------------------------------------------------------------------------- |
| `<name>.pt2`          | AOT-Inductor-compiled forward / value graph.                                |
| `<name>_jvp.pt2`      | Flat-Jacobian graph: returns the outputs plus a stacked `dout/din` block.   |
| `<name>_meta.json`    | Variable layout, dtype, device, promoted-parameter initial values.         |
| `<name>_aoti.i`       | HIT stub: drops into any input file as a replacement for the source model. |

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
single-segment artifacts always look like the first table.

### What's in the metadata JSON

The metadata is the source of truth: the `.pt2` files are opaque to
NEML2, and re-loading an artifact in a new process never re-introspects
the Python source. Schema version is currently `2`; the C++ loader
refuses any other value with a "regenerate via `neml2-compile`"
message.

```json
{
  "schema_version": 2,
  "type": "composed",            // always — kept for the unified loader path
  "device": "cpu",               // "cpu" | "cuda"
  "dtype":  "float64",           // "float64" | "float32"

  "inputs":  [{"name": "strain", "var_size": 6, "var_type": "SR2", ...}, ...],
  "outputs": [{"name": "stress", "var_size": 6, "var_type": "SR2", ...}, ...],

  "parameters": [                // promoted set only -- empty when fully baked
    {"name": "E", "dtype": "float64", "shape": [], "device": "cpu",
     "values": [200000.0], "origin": "parameter"}
  ],

  "segments": [/* one per segment, in dependency order */]
}
```

Field reference:

| Field                | Meaning                                                                                    |
| :------------------- | :----------------------------------------------------------------------------------------- |
| `schema_version`     | Currently `2`. Bumped on any breaking change.                                              |
| `type`               | Always `"composed"`. Kept for the unified loader path.                                     |
| `device` / `dtype`   | Baked into the `.pt2` graphs at export. There is no runtime override.                      |
| `inputs` / `outputs` | Master input/output order. `var_size` is flat storage (1 for `Scalar`, 6 for `SR2`, …).   |
| `parameters`         | Promoted set. Empty when no `-p` flags were given.                                         |
| `segments`           | Per-segment graph + IO layout, executed in order at runtime.                               |

The `sub_batch_shape` field (optional, per-variable) records the
variable's structured per-site axes; `sparsity` (optional, outputs
only) declares per-input exceptions to the default diagonal
cross-derivative pattern. Both are absent for the common bulk
constitutive case.

There is intentionally no `buffers` array: at the AOTI boundary every
parameter and buffer is either baked into the graph (default) or
promoted to a graph input (via `--parameter`). The `origin` field on
each promoted entry is purely diagnostic.

### Forward segment

```json
{
  "kind": "forward",
  "package": "<name>_seg{i}.pt2",
  "jvp_package": "<name>_seg{i}_jvp.pt2",      // optional
  "inputs":  [{"name": "...", "var_size": N}, ...],
  "outputs": [{"name": "...", "var_size": N}, ...],
  "param_inputs": ["E", "nu", ...]             // promoted names this segment consumes
}
```

- `package` — graph signature `(*user_inputs, *promoted_params) -> tuple[Tensor, ...]`.
- `jvp_package` — graph signature `(*user_inputs, *promoted_params) -> (*outputs, J)`,
  where `J` is a `(*B, sum(out_sizes), sum(in_sizes))` block-stacked
  Jacobian over the user inputs only.
- `param_inputs` is empty in the fully-baked case; the loader call
  shape then reduces to `(*user_inputs)`.

### Implicit segment

```json
{
  "kind": "implicit",
  "rhs_package":  "<name>_seg{i}_rhs.pt2",
  "step_package": "<name>_seg{i}_step.pt2",
  "ift_package":  "<name>_seg{i}_ift.pt2",
  "predictor_package": "<name>_seg{i}_predictor.pt2",  // omitted when no predictor

  "unknowns":   [{"name": "...", "var_size": N, "unflattened_shape": [...]}, ...],
  "givens":     [{"name": "...", "var_size": N, "unflattened_shape": [...]}, ...],
  "u_size": <int>, "g_size": <int>,

  "atol": <float>, "rtol": <float>, "miters": <int>,

  "predictor_inputs":  [...],   // present iff predictor_package is set
  "predictor_outputs": [...],

  "param_inputs": [...],
  "predictor_param_inputs": [...]
}
```

The three core graphs of an implicit segment are:

- `rhs_package` — `(u_flat, g_flat, *promoted) -> -r(u, g)`. Used for
  the baseline residual and the initial convergence check.
- `step_package` — `(u_flat, g_flat, *promoted) -> (u_new_flat, b_new_flat)`.
  Fuses assemble (`A`, `b`) + LU solve + `u_new = u + Δu` + post-update
  residual into one AOTI graph. The Newton loop body is one loader
  call plus an `at::all(b < atol).item<bool>()` convergence sync.
- `ift_package` — `(u_flat, g_flat, *promoted) -> (-A^{-1} B)` of shape
  `(*B, u_size, g_size)`. Implicit-function-theorem sensitivity at the
  converged state; consumed by `jacobian()` and `jvp()`.

`givens` pack into `g_flat` in declaration order; `unknowns` pack into
`u_flat` the same way. The `unflattened_shape` field records the
per-variable `(*sub_batch, *base)` layout for round-tripping through
the flat slot.

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
    meta = ./elasticity_meta.json
  []
[]
```

The shim has the same surface as a native model — same `input_spec`,
same `output_spec`, same call convention — but inside it dispatches
to the compiled `.pt2` instead of executing the Python `forward`.
Because the surface is identical, anything that consumes a model
through the normal HIT machinery (e.g. a `TransientDriver`) works
without modification.

## Loading from Python

The stub loads with the usual `neml2.load_model`:

```python
import neml2

model = neml2.load_model("elasticity_aoti.i", "elasticity")
```

The runtime surface on the underlying `neml2.aoti.Model` binding is
three operations plus a mutable parameter map:

| Operation               | Returns                                             |
| :---------------------- | :-------------------------------------------------- |
| `forward(inputs)`       | One tensor per output name.                         |
| `jvp(inputs, tangents)` | `(outputs, J @ v)`. Missing tangent keys → zero.    |
| `jacobian(inputs)`      | `(outputs, J)` with `J` of shape `(*B, n_out, n_in)`. |
| `named_parameters()`    | Mutable dict of promoted parameters; empty if baked. |

All three call paths take a dict keyed by the master input names
returned by `input_names()`; missing keys throw.

```python
binding = model._inner    # the bare neml2.aoti.Model runtime

# Forward.
out = binding.forward({"strain": strain.data})

# JVP: tangent dict shares keys with inputs; missing keys default to zero.
out, jvp_out = binding.jvp({"strain": strain.data}, {"strain": tangent.data})

# Dense Jacobian.
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

Promotion of any parameter that lives inside an `ImplicitUpdate`'s
`system.model` tree is **rejected up-front** with a clear
`NotImplementedError`. Parameters in forward segments of a composed
model promote normally; see [](model-compilation-pipeline) for the
underlying constraint on the equation-system wrappers.

## Dynamic batch dimension

The leading batch dimension is compiled as a dynamic axis:

```python
batch = Dim("batch", min=1, max=2**20)
```

The same artifact handles any batch size from 1 to roughly a million
without recompilation. The cost is modest extra symbolic-shape
machinery inside the lowered kernel; the benefit is that a single
artifact serves both single-point evaluation (a unit-cell stress
update) and large fan-out runs (a finite-element kernel sweeping
thousands of integration points) with no extra moving parts.

Sub-batch axes — the structured per-site dimensions some models
carry — are baked into the artifact at export time. To change them,
re-compile.

## Device and dtype pinning

The `.pt2` graphs are pinned to a specific device + dtype at export
time. There is intentionally no `to()`: offering one would either
be a misleading no-op (the graph stays put) or a contract-breaking
half-move (parameters shift, graph doesn't). To retarget, re-run
`neml2-compile` with a different `--device` / `--dtype`. Promoted
parameter tensors are placed on the same device as the graph at load
time.

## C++ runtime

The C++ surface is the same `forward` / `jvp` / `jacobian` triple,
exposed by a single header in the wheel:

```cpp
#include "neml2/csrc/aoti/Model.h"

neml2::aoti::Model model("path/to/elasticity_meta.json");

auto outputs = model.forward({{"strain", strain_tensor}});
auto [outs, J]  = model.jacobian({{"strain", strain_tensor}});

// Promoted parameters are mutable in place.
model.named_parameters().at("E").fill_(210000.0);
```

The constructor takes a single filesystem path to the metadata JSON;
the per-segment `.pt2` files are resolved relative to that path.
There is no participation in the `NEML2Object` / `Factory` /
`Registry` machinery — `neml2::aoti::Model` is a bare class, so
HIT-input parsing is the *loader*'s responsibility. The class itself
just consumes a metadata path.

The class is non-copyable and non-movable; hold it as a
`std::unique_ptr` / `std::shared_ptr` or as an automatic on the
stack.

For build-system wiring — `find_package(neml2)`, pkg-config,
`#include` paths, and the runtime library search — see
[](external-project-integration).

## Versioning

The supported schema version is `2`. Pre-v2 caches are not
backwards-compatible (the `parameters` array and per-segment
`param_inputs` field are required). When the schema bumps again the
C++ loader will refuse the old version with a clear
"regenerate via `neml2-compile`" message; that is the only
remediation path.

## See also

- [](model-compilation-pipeline) — what `neml2-compile` does between
  the HIT file and these artifacts.
- [](tutorials-models-compiled) — end-to-end how-to: compile, load,
  round-trip, JVP, parameter promotion, trade-offs against eager.
- [](cli-utilities) — `neml2-compile` and the rest of the console
  scripts.
- [](cli-neml2-inspect) — inspect a compiled stub the same way you
  inspect any other model.
- [](external-project-integration) — CMake / pkg-config wiring for
  C++ projects that consume the bundled `libneml2_aoti.so` from the
  wheel.
