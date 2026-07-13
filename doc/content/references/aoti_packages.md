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

`neml2-compile` writes an **artifact folder** plus an optional standalone
HIT stub that sits next to it:

```text
<output-dir>/                  # default ./aoti/
  <name>_aoti.i                # standalone HIT stub (optional; see --no-stub)
  <name>/                      # artifact folder
    metadata.json              # shared: structural info, promoted-param values, solver config
    cpu/float64/  *.pt2
    cuda/float64/ *.pt2
```

A compile targeting a single device (the default `--device cpu`) emits
just one `<device>/<dtype>/` subfolder; `neml2-compile --device cpu cuda`
emits both, each with its own `.pt2` binaries. The shared `metadata.json`
at the artifact root is device/dtype-independent: the loader derives the
device and dtype from the `<device>/<dtype>/` folder path, not from the
metadata. The stub points at the `<name>/` folder via an absolute
`artifact_path` field (see [The HIT stub](#the-hit-stub)).

The stub is **optional**: pass `--no-stub` to suppress it. The artifact
folder is self-describing — `cpp-aoti` and `cpp-dispatch` load directly
from it without the stub; the stub is a convenience for `neml2-run` and
`py-aoti` workflows.

The full set of files a compile will write — every `.pt2`, the shared
`metadata.json`, and the `<name>_aoti.i` stub — can be
enumerated ahead of time, without compiling, via
`neml2.cli.aoti_export.plan_export_artifacts`. `neml2-compile` prints
`[k/N]` progress against that count as each file lands, tagging every
file with its device (e.g. `cpu/float64/<name>_seg0.pt2`). The
independent compile units — each `(device, segment)` pair — are flattened
into one spawn process pool by `neml2-compile -j N`, so both axes
parallelize together: a `--device cpu cuda` build of a two-segment model
fully uses `-j 4`, and the emitted artifacts are identical to a serial
compile. `N` is capped at the number of `(device, segment)` cells, so a
single-segment single-device build ignores `-j`.

Inside each `<device>/<dtype>/` subfolder, a forward single-segment model
emits the value graph and an optional JVP graph:

| File                  | Contents                                                                    |
| :-------------------- | :-------------------------------------------------------------------------- |
| `<name>.pt2`          | AOT-Inductor-compiled forward / value graph.                                |
| `<name>_jvp.pt2`      | Per-pair Jacobian graph (only when `-d` requested a forward-segment pair): returns the outputs plus one block per requested `(out, in)`. A block that does not depend on the dynamic batch (e.g. a constant stiffness tensor) is emitted unbatched (`batch_independent` in the metadata). |

Implicit single-segment models emit the per-segment set covered in the
segment table below in place of the forward graphs.

For models that contain an `ImplicitUpdate` — or a `ComposedModel`
whose leaves contain one — the export splits at each implicit
boundary into separate **segments**, each numbered `_seg{i}_`:

| File                            | Contents                                                |
| :------------------------------ | :------------------------------------------------------ |
| `<name>_seg{i}.pt2`             | Forward-segment value graph.                            |
| `<name>_seg{i}_jvp.pt2`         | Forward-segment per-pair Jacobian graph (only when `-d` requested a pair this segment contributes to). |
| `<name>_seg{i}_residual.pt2`    | Implicit-segment Newton residual `-r(u, g)`.            |
| `<name>_seg{i}_jacobian.pt2`    | Implicit-segment residual Jacobian operator `A = ∂r/∂u` (+ `b`); no baked solve. Direct solve always; a Krylov solve only when a preconditioner or an input derivative needs the assembled `A`. |
| `<name>_seg{i}_solve.pt2`       | Implicit-segment linear solve `du = A^{-1} b` (the configured direct linear solver, un-baked from the operator). Direct solve only. |
| `<name>_seg{i}_matvec.pt2`      | Implicit-segment matrix-free operator `J·v = ∂r/∂u·v` (a Krylov solve only, in place of `_solve`); the C++ runtime drives GMRES/BiCGStab over it. |
| `<name>_seg{i}_jacobian_given.pt2` | IFT given-side operator `B = ∂r/∂g` (only when `-d` requested an input-derivative pair routed through this segment). |
| `<name>_seg{i}_solve_ift.pt2`   | IFT solve `-A^{-1} B` disassembled to per-`(unknown, given)` blocks (paired with `_jacobian_given`). |
| `<name>_seg{i}_dr_dparam.pt2`   | Parameter-derivative operators: dense `A` + `∂r/∂θ` (only when `-d` requested a promoted-parameter pair). |
| `<name>_seg{i}_solve_param.pt2` | Parameter-sensitivity solve `-A^{-1} ∂r/∂θ` per `(unknown, param)` block. |
| `<name>_seg{i}_predictor.pt2`   | Newton initial guess (only if the source had one).      |

The `_seg0_` infix is dropped in the single-segment shortcut, so
single-segment forward artifacts use the names in the first table
and single-segment implicit artifacts use the per-segment names from
the second table without the `_seg0_` prefix.

### What's in the metadata JSON

The `metadata.json` at the artifact root is the source of truth: the
`.pt2` files are opaque to NEML2, and re-loading an artifact in a new
process never re-introspects the Python source. It is
**device/dtype-independent** — those are derived from the `<device>/<dtype>/`
folder path. It records, at a high level:

- **Inputs and outputs** — master input/output order with per-variable
  storage size, base shape, and sub-batch shape.
- **Promoted parameters** — the `-p` set, with initial values and each
  parameter's natural base shape (`param_base_shape`; e.g. `[]` for a
  `Scalar`, `[6]` for an `SR2`). Floating parameters inherit the leaf
  dtype at load time; non-floating parameters carry an explicit `dtype`
  field. Empty in the fully-baked case (the artifact is then a frozen
  inference graph and `named_parameters()` is empty at load time). The
  value / jvp / jacobian / param-Jacobian / param-VJP graphs take each
  promoted parameter as a **per-batch** `(B, *param_base)` input, so
  the runtime can broadcast a stored scalar up to the call batch — or
  pass a genuinely batched parameter through (see
  [Batched parameters](#batched-parameters)).
- **Segments** — one entry per `ImplicitUpdate` boundary the exporter
  split on; executed in order at runtime, with each segment's outputs
  feeding the next segment's inputs via a shared `name → tensor` state
  map.
- **Solver config** (`solver_config`) — present for implicit (Newton-solve)
  models, absent for forward-only. Records convergence tolerances
  (`atol`, `rtol`, `miters`) and line-search settings (`ls_type`,
  `ls_max_iters`, `ls_cutback`, `ls_c`). The C++ runtime reads this
  directly from the metadata; it can be overridden at runtime via
  `set_solver_config`. The linear solve is un-baked from the residual Jacobian
  operator: the choice of linear solver lives in the `_solve.pt2` / `_solve_ift.pt2`
  / `_solve_param.pt2` graphs (recompile to change it), while the C++ runtime
  drives the operator → solve chain generically.
  `solver_kind` selects the implicit linear solve: `"direct"` (the default —
  `_jacobian` → `_solve`) or `"krylov"` (matrix-free GMRES/BiCGStab over
  `_matvec`). When `"krylov"`, a nested `krylov` block records the iterative
  settings — `method`, `restart`, `max_krylov_iters`, `krylov_abs_tol`,
  `krylov_rel_tol`, `preconditioner` (`none` / `jacobi` / `block_jacobi` /
  `full`), `cache_strategy` (`none` / `chord` / `quality_threshold`), and
  `cache_threshold`. The forward Newton solve honors it; the IFT / parameter
  derivative solves stay direct.
- **Boundary aliases** (optional `boundary_aliases`) — shallow renames
  applied at the interface only. Present only when the artifact was
  compiled with `--rename-input` / `--rename-output` / `--rename-parameter`;
  a map with `inputs` / `outputs` / `parameters` sub-maps of
  `{original_name: boundary_name}`. Every other field above keeps the
  **original** authored names; a loader reads them as the internal identity
  and applies the alias only when reporting names and keying the public
  `forward` / `jvp` / `jacobian` / `named_parameters` surface. Absent means
  the interface uses the authored names.

The exact field layout evolves alongside the export pipeline, so it is
not mirrored field-by-field here. The metadata carries an integer
`schema_version`, bumped on any breaking layout change. The C++ loader
refuses any non-matching version with a clear "regenerate via
`neml2-compile`" message; the only remediation is a re-compile.

<!-- dependencies: aoti.schema_version -->
The current schema version is `11`.

### Segment kinds

Two segment kinds appear inside `segments`:

- **Forward** segments lower to a value graph (`_seg{i}.pt2`), plus a
  per-variable-pair Jacobian graph (`_seg{i}_jvp.pt2`) when `-d`
  requested a derivative pair this segment contributes to (a block that
  does not depend on the dynamic batch is emitted unbatched). Call shape
  is `(*user_inputs, *promoted_params) -> outputs`.
- **Implicit** segments always lower `_residual.pt2` (Newton residual), plus an
  optional `_predictor.pt2` graph when the source had a `Predictor`. The
  remaining forward-solve graphs depend on `solver_config.solver_kind`:
  - a **direct** solve (`DenseLU` / `SchurComplement`) lowers `_jacobian.pt2`
    (the residual Jacobian operator `A = ∂r/∂u` + `b`) and `_solve.pt2` (the
    linear solve `du = A^{-1} b`). The linear solve is **un-baked** from the
    operator: the C++ runtime chains `_jacobian → _solve` per Newton iteration.
  - a **matrix-free Krylov** solve (`GMRES` / `BiCGStab`) lowers `_matvec.pt2`
    (`J·v = ∂r/∂u·v`) instead of `_solve`; the C++ runtime runs a batched Krylov
    iteration over it (never assembling `A`). `_jacobian.pt2` is lowered too only
    when a preconditioner (`jacobi` / `block_jacobi` / `full`) or an input
    derivative needs the assembled `A`.

  Either kind additionally lowers `_jacobian_given.pt2` (`B = ∂r/∂g`) +
  `_solve_ift.pt2` (`-A^{-1} B` implicit-function-theorem sensitivity at the
  converged state, reusing `_jacobian`'s `A`) **only when `-d` requested an
  input-derivative pair routed through this segment**, and `_dr_dparam.pt2` +
  `_solve_param.pt2` for a promoted-parameter derivative (the derivative solves
  are always direct, even under a Krylov forward). The IFT / param graphs, when
  present, are consumed by `jacobian()` / `jvp()` / `param_jacobian()`.

  The Newton solve's convergence tolerances, iteration cap, and line-search
  settings are recorded in `metadata.json` under `solver_config` and read
  directly by the C++ runtime (see [What's in the metadata JSON](#whats-in-the-metadata-json)
  above). The linear-solver choice (direct vs Krylov, preconditioner, cache
  strategy) is baked into the graph set + `solver_config`.

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
wherever a `Driver` consumes the model by name.

The stub is **optional**: `neml2-compile` emits it by default; pass
`--no-stub` to suppress it. The artifact folder is self-describing —
`cpp-aoti` and `cpp-dispatch` load directly from the folder and do not
require the stub.

A typical stub:

```ini
# Auto-generated by neml2-compile from input.i.
# Drop-in replacement for the original [elasticity] model.
# Do not edit; regenerate via `neml2-compile`.

[Models]
  [elasticity]
    type = AOTIModel
    artifact_path = 'elasticity'
  []
[]
```

The `artifact_path` is a path to the artifact folder (`<output-dir>/<name>/`),
written **relative to the stub** — usually just the folder name, since the stub
(`<output-dir>/<name>_aoti.i`) sits right next to it. The loader (Python shim and
C++ `load_model`) resolves a relative `artifact_path` against the stub's own
directory, reads `metadata.json` from that root, and resolves
`<artifact_path>/<device>/<dtype>/` for the device and dtype it runs on. Because
the path is relative, the stub and its artifact folder form a **portable bundle**:
copy or move the two together — to another directory, machine, or user — and it
still loads, with no machine-specific path baked in and no recompile. Moving the
stub *away* from its artifact folder is intentionally unsupported; keep them
together. (An absolute `artifact_path` is also accepted if you hand-edit one.)

The shim has the same surface as a native model — same `input_spec`,
same `output_spec`, same call convention — but inside it dispatches
to the compiled `.pt2` instead of executing the Python `forward`.
Because the surface is identical, anything that consumes a model
through the normal HIT machinery (e.g. a `TransientDriver`) works
without modification.

For a model with an implicit (`ImplicitUpdate`) segment the solver
configuration is read directly from `metadata.json` (`solver_config`).
The stub carries no `[Solvers]` block — solver settings travel with the
artifact, not the stub. To override them at runtime use
`set_solver_config` on the loaded `Model`. `[EquationSystems]` and
`[Data]` are dropped — their state was baked in.

## Parameter promotion (`-p`)

Every parameter and buffer is **baked** into the lowered graph as a
constant by default. Baked entries are immutable post-compile but
cost nothing per call — they're folded directly into the kernel.
Each `-p NAME` flag promotes one entry to a runtime-flexible
**graph input**:

```bash
neml2-compile input.i --model elasticity -p elasticity.E
neml2-compile input.i --model viscoplasticity -p hardening.tau0 -p flow_rate.A
```

Names are fully qualified. A bare (non-composed) leaf is wrapped in a
single-child `ComposedModel` at export — matching the eager runtime
(`neml2::eager::Model`) — so its parameters are qualified by the model name:
`--model elasticity` exposes `elasticity.E`. A composed model's parameters are
already qualified by their child-leaf names (e.g. `hardening.tau0`). Either way
the names match `model.named_parameters(recurse=True)` on the model the eager
runtime constructs, so `-p`, `metadata.parameters`, and `named_parameters()`
agree across the AOTI and eager routes.

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
