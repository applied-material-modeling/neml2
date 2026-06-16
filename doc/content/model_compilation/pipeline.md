(model-compilation-pipeline)=
# Compilation pipeline

This page is the developer reference for what `neml2-compile`
actually does between reading a HIT input file and writing the
`.pt2` artifacts described in [](aoti-packages). It's aimed at
people maintaining the export path; downstream users who only want
to *use* a compiled model should start with
[](tutorials-models-compiled).

## High-level shape

The entry point `neml2.cli.aoti_compile.main` orchestrates two
mostly-orthogonal subsystems:

1. `neml2.cli.aoti_export.export_model_for_aoti` — turns a live
   `nn.Module` into one or more `.pt2` graphs plus a metadata JSON.
2. `neml2.cli.aoti_compile.emit_aoti_stub` — rewrites the source
   HIT file so the original `[Models]/<name>` block becomes an
   `AOTIModel` shim that points at the metadata.

The shim is what `neml2.load_model` ends up loading; the runtime
binding (`neml2::aoti::Model`, exposed through `neml2.aoti.Model`)
reads the metadata and dispatches each call to the right `.pt2`
graph.

The export path is the deeper of the two and breaks down into seven
ordered steps:

1. Load the HIT file and instantiate the live model.
2. Validate and snapshot promoted parameters (`-p NAME`).
3. Re-route promoted parameters through the leaf's NLParam
   machinery so they surface as graph inputs.
4. Freeze remaining `nn.Parameter`s to buffers so torch.export
   bakes them as constants.
5. Seed per-variable sub-batch shapes from the HIT driver inputs.
6. Partition the model into forward / implicit segments at every
   `ImplicitUpdate` boundary.
7. Trace + lower each segment through `torch.export` and
   `aoti_compile_and_package`; collect per-segment metadata.

The metadata dict accumulated across these steps is what
[](aoti-packages) documents on disk. Below, each step expands on
*why* it exists in addition to *what* it does.

## Step 1 — Load the model

`neml2.factory.load_input(hit_path)` parses the HIT file via
`nmhit` and returns a `Factory` from which any registered object
can be instantiated by name. `factory.get_model(model_name)`
returns a live `Model` (i.e. an `nn.Module`) with all dependencies
already resolved and wired.

The compiled artifact is a *frozen* snapshot of this live object —
re-running `neml2-compile` is the only way to pick up source
changes.

## Step 2 — Validate and snapshot promoted parameters

`_validate_promoted` walks the model's fully-qualified parameter
namespace (the same one `model.named_parameters(recurse=True)`
exposes) and:

- rejects any name that doesn't resolve;
- rejects any name that lives inside an `ImplicitUpdate`'s
  `system.model` tree — the equation-system AOTI export wrappers
  (`RHS`, `NewtonStep`, `IFT`) have fixed
  `(u_flat, g_flat)` forward signatures and can't yet accept a
  `*nl_params` tail.

For each accepted name, `_snapshot_promoted` records the initial
value before any rewriting. Those tensors land in
`metadata.parameters[].values` so the C++ side can populate
`aoti::Model::named_parameters()` at construction.

## Step 3 — Re-route promoted parameters through NLParam

This step transitions each promoted name from a static field on the
leaf to a graph input the caller supplies on every invocation. Each
promoted name has to move from "static field on the leaf" to "graph input the
caller supplies on every invocation". `_promote_to_nl_params`:

- deletes the static slot from the leaf's `_parameters` dict;
- adds an entry to the leaf's `_nl_params` dict tagged with the
  qualified name as its input name;
- appends the qualified name to the leaf's `input_spec`.

Once that's done, the next `ComposedModel` wrap downstream picks
the new inputs up through normal dependency resolution; the
call-boundary `_coerce_to_input_type` re-wraps the incoming raw
tensor in the right `TensorWrapper` before handing it to the leaf's
`forward`, which reads the value through `_get_param`. The leaf
implementation is unchanged — the NLParam pack abstracts the
"baked vs promoted" distinction away.

## Step 4 — Freeze remaining parameters to buffers

`torch.export` treats `nn.Parameter` as a *graph input* by default
and `nn.Buffer` as a *constant*. We want every non-promoted entry
folded into the kernel; `_freeze_remaining_parameters_to_buffers`
walks the module tree and converts each surviving `nn.Parameter`
into a persistent buffer (deep-copy of the data, same dtype +
device). Promoted entries are already gone from `_parameters` so
this loop skips them naturally.

## Step 5 — Seed sub-batch shapes from HIT driver inputs

Some constitutive models — slip-system kinematics, multi-grain
plasticity — carry structured per-site dimensions on top of the
batch axis (the `sub_batch_shape`). The implicit-update tracer
needs that shape to build the right Schur/Block layout, but the
shape is only knowable from the driver inputs in the HIT file.

`_read_settings` parses `[Settings]/example_batch_shape` (uniform
field or per-variable subsection); `_resolve_example_shapes` expands
that into a `name → (dyn, sub)` map over the model's `input_spec`;
`_seed_implicit_subbatch` then calls `system.initialize(...)` on every
nested `ModelNonlinearSystem` with example inputs of the resolved
shape, so the system's per-variable `sub_batch_shape` is populated
before the tracer runs.

When no `example_batch_shape` is declared, the resolver falls back to
a small built-in default that's correct for the bulk-constitutive
case.

## Step 6 — Partition into segments

Every `ImplicitUpdate` becomes its own segment so the Newton loop
can be driven from the C++ orchestrator (one `.pt2` call per step
until convergence). `torch.export` traces eager Python and can't
trace `while not converged`, so the data-dependent loop boundary is
the natural split point.

`_partition_into_segments` dispatches:

- root is an `ImplicitUpdate` → **one implicit segment**;
- root is a `ComposedModel` containing any `ImplicitUpdate` →
  flatten the leaves and walk in topological order; consecutive
  non-implicit leaves coalesce into a single forward segment, every
  `ImplicitUpdate` becomes its own implicit segment;
- anything else → **one forward segment**.

At runtime each segment writes its outputs into a shared
`name → tensor` state map; the next segment reads its declared
inputs from that map. The shared map plus topological order is
what lets a multi-segment artifact present a single forward
contract to the loader.

## Step 7 — Trace, lower, package

Each segment routes through `neml2.models.export.compile_model`, a thin
adapter around `torch.export.export` + `torch._inductor.aoti_
compile_and_package`. The adapter handles three pieces of
machinery the raw torch APIs don't:

**Dynamic batch dim.** Every input gets axis 0 marked as a single
shared dynamic `Dim` so the lowered kernel specializes once and
serves any batch size in the supported range without recompilation.
The cost is modest extra symbolic-shape machinery inside the kernel;
the benefit is that a single artifact serves both single-point
evaluation and large fan-out runs.

**Pytree-structured signatures.** Typed wrappers (`Scalar`, `SR2`,
`R2`, ...) are pytree-registered dataclasses. The adapter mirrors
their pytree structure when building the dynamic-shape spec so
torch.export sees the right dynamism on the underlying tensor
leaves, while static metadata (e.g. sub-batch widths) stays out of
the trace boundary.

**Forward-signature variants.** Different leaves use different
forward signatures — pure `*inputs`, named positionals followed by
`*nl_params`, or named positionals only — and the adapter normalizes
the example list and dynamic-shape spec to whichever shape the
target leaf expects.

The lowered output is a `.pt2` package per graph. **Implicit
segments** lower three graphs each (`rhs`, `step`, `ift`) plus an
optional `predictor`; **forward segments** lower one value graph
plus an optional flat-Jacobian graph (`jvp`). Implicit segments
all route through the `RHS` / `NewtonStep` / `IFT` wrappers in
`neml2.es.implicit`; `NewtonStep` and `IFT` forward
to whichever `linear_solver` the source model is configured with
(`DenseLU` for the common single-group case; `SchurComplement` for
the `BLOCK + DENSE` 2-group factorisation).

The C++ orchestrator sees the same flat `(u_flat, g_flat) →
(u_new, b_new)` contract either way — the multi-group / sub-batch
structure is internal to the `.pt2`.

## After export: emit the HIT stub

Once the metadata JSON has been written, `emit_aoti_stub` rewrites
the source HIT file:

- parse the original with `nmhit`;
- find the top-level `[Models]` block containing `model_name`
  (HIT allows multiple `[Models]` blocks — we have to walk them);
- remove the original sub-block;
- add a new `[<name>]` sub-block with `type = AOTIModel` and
  `meta = <relative path to the metadata JSON>`;
- drop any legacy `[Settings]/aoti_*` keys (no-ops in v3 but they
  would fail the new factory's strict-key check);
- write the rendered tree to `<name>_aoti.i` with a generated
  header.

The metadata path is recorded *relative* to the stub's directory
so the artifact set is movable as a unit — drop the four files
into a new directory together and the stub still resolves
correctly.

## Where to look in the source

| Subsystem                            | Module / function                                                |
| :----------------------------------- | :--------------------------------------------------------------- |
| CLI entry point                      | `neml2/cli/aoti_compile.py::main`                                |
| HIT-stub emission                    | `neml2/cli/aoti_compile.py::emit_aoti_stub`                      |
| Top-level export                     | `neml2/cli/aoti_export.py::export_model_for_aoti`                |
| Promote / freeze / re-route          | `_validate_promoted`, `_promote_to_nl_params`, `_freeze_remaining_parameters_to_buffers` |
| Example-shape resolution + sub-batch seeding | `_read_settings`, `_resolve_example_shapes`, `_seed_implicit_subbatch` |
| Partitioning                         | `_partition_into_segments`, `_contains_implicit`, `_flatten_composed` |
| Forward / implicit segment lowering  | `_compile_forward_segment`, `_compile_implicit_segment`          |
| `torch.export` adapter               | `neml2/models/export.py::compile_model`                          |

## See also

- [](aoti-packages) — on-disk artifact format and runtime API.
- [](tutorials-models-compiled) — end-to-end how-to.
- [](cli-utilities) — the rest of the CLI surface, including
  `neml2-inspect` against a compiled stub.
