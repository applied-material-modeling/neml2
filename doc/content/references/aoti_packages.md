(aoti-packages)=
# AOTI packages

This is the on-disk reference for the artifacts that `neml2-compile`
produces — the `.pt2` graphs, the metadata, and the HIT stub. It is the
shared substrate both compiled-model routes load: from Python via
[](py-aoti) and from C++ via [](cpp-aoti). Read it when you need to know
exactly what a `.pt2` package contains.

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
                        [-d|--derivative OUT:IN ...]
```

Derivative graphs are **opt-in**. With no `-d` flag only the `forward`
graph is compiled and the runtime `jvp` / `jacobian` raise. Each
`-d OUT:IN` requests the Jacobian/JVP for that output-input pair; omit a
side to select all on it (`stress:` = every input of `stress`, `:strain`
= every output w.r.t. `strain`, `:` = all pairs). The requested master
pairs are recorded in the metadata's top-level `derivatives` array.

Each derivative is one per-variable-pair block. A block that does not
depend on the dynamic batch (e.g. a constant stiffness tensor) is returned
**unbatched** at its natural `(*out_base, *in_base)` shape.

```{note}
For an **implicit (Newton-solve) model with sub-batched (per-grain) state**
— e.g. crystal plasticity — a derivative of a **non-sub-batched output
w.r.t. a non-sub-batched input** (e.g. a global stress w.r.t. a global
strain) is supported: the IFT solve handles the internal per-grain coupling
and the returned block has no grain axis. A derivative that *touches* a
per-grain variable (a sub-batched output or input) is not yet implemented
and fails fast at `neml2-compile` with a clear "… involves the sub-batched
variable …" error; use eager mode (`torch.autograd`) for per-grain
sensitivities. Forward evaluation of such models, and all plain-batch /
forward derivatives, are fully supported.
```

```{note}
**Parameter derivatives and saved-output ops.** Parameter-derivative graphs
(`-d OUT:PARAM` for a promoted `-p PARAM`) are compiled by reverse-mode
autograd, which is currently the only AD that lowers through AOTInductor. An
upstream PyTorch limitation
([pytorch/pytorch#187907](https://github.com/pytorch/pytorch/issues/187907))
prevents lowering a reverse-mode graph through an element-wise op whose backward
**saves its output** (`sqrt`, `exp`, `tanh`, `reciprocal`, and division by a
differentiated value — e.g. a von Mises norm or an Arrhenius/Kocks–Mecking
`exp`). NEML2 works around the common cases transparently: the typed-tensor
wrappers (`neml2/types/functions.py`) route these ops through input-recompute
`autograd.Function`s on the AD path while keeping the plain instruction on the
value path, so such models compile and match eager to machine precision. If a
model's differentiated path reaches a saved-output op that is *not* wrapped
(e.g. a raw `torch.<op>` inside a leaf), `neml2-compile` fails fast with an
actionable message; route that op through its wrapper, or compute the parameter
derivative through the eager runtime (Python `neml2.eager` / `cpp-eager`), which
never lowers a graph.
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
| `<name>_jvp.pt2`      | Per-pair Jacobian graph (only when `-d` requested a forward-segment pair): returns the outputs plus one block per requested `(out, in)`. A block that does not depend on the dynamic batch (e.g. a constant stiffness tensor) is emitted unbatched (`batch_independent` in the metadata). |
| `<name>_meta.json`    | Variable layout, dtype, device, promoted-parameter initial values, and the top-level `derivatives` array (master `[out, in]` pairs the artifact supports; empty = none). |

Implicit single-segment models emit the per-segment set covered in the
segment table below in place of the forward graphs.

For models that contain an `ImplicitUpdate` — or a `ComposedModel`
whose leaves contain one — the export splits at each implicit
boundary into separate **segments**, each numbered `_seg{i}_`:

| File                            | Contents                                                |
| :------------------------------ | :------------------------------------------------------ |
| `<name>_seg{i}.pt2`             | Forward-segment value graph.                            |
| `<name>_seg{i}_jvp.pt2`         | Forward-segment per-pair Jacobian graph (only when `-d` requested a pair this segment contributes to). |
| `<name>_seg{i}_rhs.pt2`         | Implicit-segment Newton residual `-r(u, g)`.            |
| `<name>_seg{i}_step.pt2`        | Implicit-segment fused assemble + solve + update.       |
| `<name>_seg{i}_ift.pt2`         | Implicit-function-theorem sensitivity `-A^{-1} B` (only when `-d` requested a pair routed through this segment). |
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
- **Promoted parameters** — the `-p` set, with initial values, the
  artifact's device, and each parameter's natural base shape
  (`param_base_shape`; e.g. `[]` for a `Scalar`, `[6]` for an `SR2`).
  Empty in the fully-baked case (the artifact is then a frozen inference
  graph and `named_parameters()` is empty at load time). The value /
  jvp / jacobian / param-Jacobian / param-VJP graphs take each promoted
  parameter as a **per-batch** `(B, *param_base)` input, so the runtime
  can broadcast a stored scalar up to the call batch — or pass a
  genuinely batched parameter through (see
  [Batched parameters](#batched-parameters)). The recorded device is the
  artifact's compile-target device (not the source snapshot's), since
  those graphs dereference the parameter on-device.
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
The current schema version is `7`.

### Segment kinds

Two segment kinds appear inside `segments`:

- **Forward** segments lower to a value graph (`_seg{i}.pt2`), plus a
  per-variable-pair Jacobian graph (`_seg{i}_jvp.pt2`) when `-d`
  requested a derivative pair this segment contributes to (a block that
  does not depend on the dynamic batch is emitted unbatched). Call shape
  is `(*user_inputs, *promoted_params) -> outputs`.
- **Implicit** segments always lower `_rhs.pt2` (Newton residual) and
  `_step.pt2` (fused assemble + LU solve + update + post-update
  residual), plus an optional `_predictor.pt2` graph when the source had
  a `Predictor`. They additionally lower `_ift.pt2` (`-A^{-1} B`
  implicit-function-theorem sensitivity at the converged state) **only
  when `-d` requested a pair whose derivative path runs through this
  segment**. The Newton loop body is one loader call per iteration plus a
  convergence sync; the IFT graph, when present, is consumed by
  `jacobian()` and `jvp()`.

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

(batched-parameters)=
### Batched parameters

A promoted parameter can be set to a **per-batch-element** value at runtime —
e.g. a `Scalar` parameter given shape `(B,)` via `set_parameter`, a spatially
varying material property (the common MOOSE inverse-optimization case). The
value / jvp / jacobian / param-Jacobian / param-VJP graphs all take each promoted
parameter as a per-batch `(B, *param_base)` input (a genuine per-batch input is
symbolic, so it is never re-pinned by Inductor's `constant_fold_uniform_value`
under the dynamic batch). The runtime broadcasts a stored *scalar* parameter up
to the call batch before the call, and passes a genuinely *batched* parameter
through unchanged; the call batch is `broadcast(input batches, parameter
batches)`.

The two parameter-derivative surfaces then follow the parameter's shape,
matching the eager runtime:

- **`param_jacobian`** returns a dense `d(out)/d(param)` block
  `(*B, *out_base, *param_base)` — per batch element, so a batched parameter's
  block varies across the batch.
- **`param_vjp`** returns the adjoint `dL/d(param)` **summed over the batch for a
  global (unbatched) parameter** (the gradient reverse-mode autograd accumulates
  for a shared leaf) and **per-element `(*B, *param_base)` for a batched one**
  (each batch element depends only on its own copy). The adjoint graph emits the
  per-element gradient and the runtime collapses it to the stored parameter's
  shape.

For `cpp-dispatch`, each chunk receives the batched parameter sliced to its own
rows; per-element `param_jacobian` blocks and batched `param_vjp` adjoints are
concatenated back across chunks, while a global `param_vjp` adjoint is summed.
Sub-batched (per-grain) parameters are not yet supported.

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

## See also

- [](py-aoti) — load and call a compiled package from Python.
- [](cpp-aoti) — load and call a compiled package from C++.
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
