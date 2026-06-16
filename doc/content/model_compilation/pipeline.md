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

This is the deepest pipeline step. Each promoted name has to
transition from "static field on the leaf" to "graph input the
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

`_hit_driver_example_inputs` walks the `[Drivers]` section,
materializes one example tensor per declared force/state variable,
and `_seed_implicit_subbatch` propagates each tensor's
sub-batch shape into the matching `ModelNonlinearSystem`'s
per-variable spec.

The step is silently skipped when the HIT file has no `[Drivers]`
section — the bulk-constitutive default of `sub_batch_shape = ()`
is correct for that case.

## Step 6 — Partition into segments

`torch.export` traces eager Python; a Newton loop's
`while not converged` is not traceable. So every `ImplicitUpdate`
must become its own segment, with the Newton iteration driven by
the C++ orchestrator (one `.pt2` call per step until convergence).

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
shared `Dim("batch", min=1, max=2**20)` so the lowered kernel
specializes once and serves any batch size from 1 to ~1M without
recompilation. The cost is modest extra symbolic-shape machinery
inside the kernel; the benefit is that a single artifact serves
both single-point evaluation and large fan-out runs.

**Pytree-structured signatures.** Typed wrappers (`Scalar`, `SR2`,
`R2`, ...) are dataclasses registered with `torch.utils._pytree.
register_dataclass(field_names=..., drop_field_names=...)`. The
adapter walks each example input through `_dyn_spec`, mirroring
the registered pytree structure so the dynamic-shape spec reaches
the underlying `torch.Tensor` leaves. `drop_field_names` excludes
static metadata (e.g. `sub_batch_ndim`) from the trace boundary.

**Forward-signature variants.** Leaves come in three shapes —
pure `*inputs`, named positionals followed by `*nl_params`, or
named positionals only — and torch.export collapses
`VAR_POSITIONAL` packs into a single nested-tuple argument. The
adapter splits the example list at the named-positional count and
re-bundles the tail into the pack, so the `dynamic_shapes`
structure matches what `torch.export` actually sees.

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

### ELF GNU_STACK patch

`compile_model` post-processes every `.so` inside each emitted
`.pt2`: it walks the ELF program headers and clears the `PF_X`
bit on the `PT_GNU_STACK` entry. Without this, PyTorch 2.12's
constants-folding assembler omits the `.note.GNU-stack` section,
and ld.bfd / lld respond by marking the resulting shared object's
stack as executable. SELinux and modern hardened-kernel systems
refuse to `dlopen()` such an object.

The patch is one byte per `.so` and is reversible (the artifact
loader doesn't depend on the bit), but without it the C++ runtime
fails to open the artifact on roughly half of all production
Linux installs. The dispatch lives behind a single helper so the
day torch upstreams a fix, we delete the helper and the call
site.

## After export: emit the HIT stub

Once `_write_meta` has serialized the metadata JSON,
`emit_aoti_stub` rewrites the source HIT file:

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
| Sub-batch propagation                | `_hit_driver_example_inputs`, `_seed_implicit_subbatch`          |
| Partitioning                         | `_partition_into_segments`, `_contains_implicit`, `_flatten_composed` |
| Forward / implicit segment lowering  | `_compile_forward_segment`, `_compile_implicit_segment`          |
| `torch.export` adapter               | `neml2/models/export.py::compile_model`                          |
| ELF GNU_STACK patch                  | `neml2/models/export.py::_clear_elf_execstack`                   |

## See also

- [](aoti-packages) — on-disk artifact format and runtime API.
- [](tutorials-models-compiled) — end-to-end how-to.
- [](cli-utilities) — the rest of the CLI surface, including
  `neml2-inspect` against a compiled stub.
