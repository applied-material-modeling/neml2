# NEML2 AOTI package format (schema v2)

A running spec for the on-disk artifacts produced by `neml2-compile` and consumed by `neml2::aoti::Model` (the thin C++ runtime in `include/neml2/aoti/Model.h`). Same source-of-truth for the Python exporter, the C++ loader, and (eventually) the public artifact-format documentation.

## Goals of the format

1. **Bake by default; promote on demand.** Every parameter and buffer in the source model is folded into the compiled graph as a constant unless the user explicitly promotes it via `neml2-compile --parameter NAME`. Promoted entries become graph inputs and are mutable at runtime through `aoti::Model::named_parameters()`. The fully-baked artifact is the efficient default for inference; promotion is the escape hatch for training-loop weights and what-if knobs.
2. **Device-pinned at export time.** The artifact carries an explicit device and dtype, baked into the `.pt2` segments at trace time. There is no runtime `to()` — to retarget, re-run `neml2-compile`.
3. **One loader handles every shape.** A single `aoti::Model` class loads everything; the on-disk schema is uniform across pure forward, pure implicit, and composed-with-breakpoint shapes.
4. **Metadata is the source of truth.** The `.pt2` files are opaque to NEML2 — all variable names, sizes, orchestration parameters, and promoted-parameter initial values live in a sidecar JSON. Re-loading an AOTI artifact in a new process never requires re-introspecting the Python model.
5. **Derivative graphs travel with the value graph.** Each segment ships the auxiliary `.pt2`s its Jacobian path needs (`_jvp` for forward, `_ift` for implicit). The value-only path ignores them with zero overhead; `jacobian()` and `jvp()` consume them.
6. **Predictors are first-class.** When a Python `ImplicitUpdate` carries a predictor for its Newton initial guess, that predictor compiles to its own `.pt2` and survives across the cache-miss boundary just like the residual surface.

## Producing artifacts: `neml2-compile`

`neml2-compile` is the **only** way to produce an AOTI artifact. Synopsis:

```
neml2-compile <input.i> --model <name>
                        [--output-dir <dir>]
                        [--device cpu|cuda] [--dtype float64|float32]
                        [-p|--parameter NAME ...]
```

- `<input.i>` — HIT input file declaring the source model.
- `--model NAME` — name of the `[Models]` block to compile.
- `--output-dir DIR` — output directory (default: `./aoti/<NAME>/`).
- `--device`, `--dtype` — baked into the artifact at export time; no runtime override.
- `-p NAME` / `--parameter NAME` (repeatable) — promote a parameter or buffer (fully-qualified dotted name, as `model.named_parameters(recurse=True)` emits) to runtime-flexible status. Without any `-p`, the artifact is fully baked.

Produces:

```
<output-dir>/
  <name>_meta.json                  # schema-v2 metadata (always present)
  <name>_aoti.i                     # runnable HIT stub (replaces <name>'s [Models] block)
  <name>_seg{i}.pt2                 # forward-segment value graph
  <name>_seg{i}_jvp.pt2             # forward-segment flat Jacobian graph
  <name>_seg{i}_rhs.pt2             # implicit-segment Newton residual
  <name>_seg{i}_step.pt2            # implicit-segment fused (assemble+solve+update)
  <name>_seg{i}_ift.pt2             # implicit-segment IFT Jacobian (du/dg)
  <name>_seg{i}_predictor.pt2       # implicit-segment Newton initial guess (optional)
```

**Single-segment shortcut.** When the exported model has exactly one segment (pure forward or pure implicit), the `_seg0_` infix is dropped — files are `<name>.pt2`, `<name>_jvp.pt2`, `<name>_rhs.pt2`, `<name>_step.pt2`, `<name>_ift.pt2`, `<name>_predictor.pt2`. Schema is otherwise identical.

The `.i` stub is the original input with the `[Models]/<name>` block surgically replaced by an `AOTIModel` shim. Other sections (`[Tensors]`, `[Drivers]`, `[Settings]`, ...) are copied through verbatim. **Note:** the stub is a valid HIT file as of this iteration but is not yet end-to-end runnable -- driving it requires the Python pybind binding and native-driver dispatch, which are deferred to a follow-up PR. C++ unit tests bypass the stub and construct from a meta path directly.

## Metadata schema (v2)

```jsonc
{
  "schema_version": 2,                 // bumped in this iteration; v1 caches must
                                       // be re-exported via neml2-compile
  "type": "composed",                  // always — kept for the unified loader path
  "device": "cpu",                     // baked artifact target ("cpu" | "cuda")
  "dtype":  "float64",                 // baked artifact dtype  ("float64" | "float32")

  "inputs":  [{"name": "<var>", "var_size": <int>,
               "var_type": "<TypedWrapper>",                  // e.g. "Scalar", "SR2"
               "sub_batch_shape": [<int>, ...]?,              // optional
               "sparsity": {"<other>": "dense"}?              // optional (outputs only)
              }, ...],                                         // master input order
  "outputs": [...],                                            // master output order

  "parameters": [                      // promoted set only -- empty when the model
                                       // was compiled with no -p flags
    {"name": "<dotted.name>",
     "dtype": "float64",
     "shape": [<int>, ...],            // shape of the promoted tensor
     "device": "cpu",
     "values": [<float>, ...],         // flattened initial value (row-major)
     "origin": "parameter" | "buffer"  // diagnostic; the C++ runtime treats both the same
    }, ...
  ],

  "segments": [
    /* ... segment objects, see below ... */
  ]
}
```

**No `buffers` array.** The PyTorch parameter/buffer distinction does not survive the AOTI boundary. At export time everything is either baked into the graph (default) or promoted to a graph input (via `--parameter`); the metadata records only the promoted set. The `origin` field is purely diagnostic.

`var_size` is the flat storage size of the variable's tensor type — `1` for `Scalar`, `6` for `SR2`, `36` for `SSR4`, etc.

`sub_batch_shape` records the variable's structured per-site region (the trailing batch axes that broadcast like batch but carry block-diagonal derivative structure). Absent or empty means no sub-batch.

`sparsity` (output variables only) declares per-`{input_name: "dense"}` exceptions to the default `"diagonal"` cross-derivative pattern. Outputs with no dense edges omit the field.

### Forward segment

```jsonc
{
  "kind": "forward",
  "package": "<name>_seg{i}.pt2",
  "jvp_package": "<name>_seg{i}_jvp.pt2",     // optional
  "inputs":  [{"name": "<var>", "var_size": <int>}, ...],
  "outputs": [{"name": "<var>", "var_size": <int>}, ...],
  "param_inputs": ["<dotted.name>", ...]      // promoted names this segment's
                                              // graphs consume (graph-call order);
                                              // empty in the fully-baked case
}
```

- `package` — torch.export'd graph with signature `(*user_inputs..., *promoted_params...) -> tuple[Tensor, ...]`. User-input order matches `inputs`; promoted-param order matches `param_inputs`.
- `jvp_package` — graph with signature `(*user_inputs..., *promoted_params...) -> (*outputs..., J)`. `J` is a `(*B, sum(out.var_size), sum(in.var_size))` block-stacked Jacobian over **user inputs only** (promoted-param tangents are not exposed). Row blocks follow `outputs` order; column blocks follow `inputs` order.

When `param_inputs` is empty (no promotions), the loader call shape is identical to v1 — `(*user_inputs)` only.

### Implicit segment

```jsonc
{
  "kind": "implicit",
  "rhs_package":  "<name>_seg{i}_rhs.pt2",
  "step_package": "<name>_seg{i}_step.pt2",
  "ift_package":  "<name>_seg{i}_ift.pt2",          // optional
  "predictor_package": "<name>_seg{i}_predictor.pt2",  // omitted when no predictor

  "unknowns": [{"name": "<var>", "var_size": <int>, "unflattened_shape": [...]}, ...],
  "givens":   [{"name": "<var>", "var_size": <int>, "unflattened_shape": [...]}, ...],
  "u_size":  <int>,                                    // sum of unknown var_sizes
  "g_size":  <int>,                                    // sum of given   var_sizes

  // Newton solver knobs — read from the EquationSystem's solver block.
  "atol":   <float>,
  "rtol":   <float>,
  "miters": <int>,

  // Predictor inputs / outputs — present iff predictor_package is set.
  "predictor_inputs":  [{"name": "<var>", "var_size": <int>}, ...],
  "predictor_outputs": [{"name": "<var>", "var_size": <int>}, ...],

  "param_inputs": ["<dotted.name>", ...],           // promoted names consumed by
                                                    // rhs / step / ift (graph order)
  "predictor_param_inputs": ["<dotted.name>", ...]  // promoted names consumed by
                                                    // the predictor graph (may
                                                    // differ from param_inputs)
}
```

- `rhs_package` — `(u_flat, g_flat, *promoted) -> -r(u, g)`. Used for the baseline residual and the initial convergence check.
- `step_package` — `(u_flat, g_flat, *promoted) -> (u_new_flat, b_new_flat)`. Fuses assemble (A, b) + LU solve + `u_new = u + Δu` + post-update residual into one AOTI graph. The C++ Newton loop body is one loader call plus an `at::all(b < atol).item<bool>()` convergence sync.
- `ift_package` — `(u_flat, g_flat, *promoted) -> (-A⁻¹ B)` of shape `(*B, u_size, g_size)`. Implicit-function-theorem sensitivity at the converged state, consumed by `aoti::Model::jacobian()` and `aoti::Model::jvp()`.
- `predictor_package` — `(*predictor_inputs, *predictor_promoted) -> (*predictor_outputs)`. Outputs scatter into `u0_flat` by matching each `predictor_outputs[k].name` against the unknown names; predictor outputs that don't correspond to any unknown are dropped.

`givens` packs into `g_flat` in declaration order. `unknowns` packs into `u_flat` in declaration order. Both packings are flat concatenation along the trailing dimension; per-variable subtensors are read back by `narrow`-ing on offsets computed from the cumulative `var_size` prefix sum.

`unflattened_shape` records the per-variable `(*sub_batch, *base)` layout for round-tripping through the flat `(*B, var_size)` slot. Empty list means a Scalar with no trailing storage dim.

`predictor_inputs` can include history variables that aren't part of `givens` (e.g. `flow_rate~1` for a `ConstantExtrapolationPredictor` on `flow_rate`).

`predictor_param_inputs` is distinct from `param_inputs` because the predictor module's parameter space is independent of the system model's (the predictor is a separate `nn.Module`).

## Partitioning rule

The Python exporter's dispatch in `export_model_for_aoti`:

```
inner = unwrap(model)
if isinstance(inner, ImplicitUpdate):
    -> single implicit segment
elif inner is a ComposedModel whose flattened leaves contain any ImplicitUpdate:
    -> partition into segments at each ImplicitUpdate boundary
else:
    -> single forward segment
```

The partitioning algorithm (`_partition_into_segments`) walks the master ComposedModel's resolved leaf order, accumulating consecutive non-implicit leaves into a `forward` segment and breaking out a one-leaf `implicit` segment at each `ImplicitUpdate`. The resulting list is consumed in order at runtime — each segment writes its outputs into a shared `name → tensor` state map, and the next segment reads its declared inputs from that state.

Each forward segment's outputs are computed by wrapping its leaf list in a fresh `ComposedModel(payload, additional_outputs=...)`. The `additional_outputs` set is `(master_output_names ∪ ∪_{later seg} seg_inputs) ∩ seg_produced` — i.e. any variable produced by this segment that the master or some downstream segment will read, even if the segment's own dependency resolver would normally treat it as a consumed intermediate.

## C++ surface

`neml2::aoti::Model` is a **bare** class — no `NEML2Object` inheritance, no Registry registration, no `OptionSet`. The constructor takes a single filesystem path to the metadata JSON:

```cpp
#include <neml2/aoti/Model.h>

auto model = neml2::aoti::Model("path/to/my_model_meta.json");

auto outputs = model.forward({{"strain", strain_tensor}});
auto [outputs, J] = model.jacobian({{"strain", strain_tensor}});

// Promoted parameters are mutable via named_parameters():
model.named_parameters().at("E").fill_(210000.0);
```

HIT-input loading is the loader's responsibility, not the class's. Parse the `.i` with `nmhit`, walk to the `[Models]` block, filter for `type = AOTIModel`, extract the `meta` field, call the bare ctor. In this PR the only loaders are C++ unit tests; the §7 follow-up adds a Python pybind binding + native-driver dispatch that does the HIT-side parsing.

The artifact is device- and dtype-pinned. There is no `to()` — the graphs cannot be migrated post-load. To retarget, re-run `neml2-compile`.

## Promotion path

Promotion routes through NEML2's existing typed-parameter (`NLParam`) machinery -- the same mechanism that the pyzag calibration path uses. For each `--parameter NAME` the exporter:

1. Resolves the holder leaf via `model.named_parameters() ∪ named_buffers()`.
2. Verifies the holder is a native NEML2 `Model` (has `_nl_params`, `input_spec`, `_typed_storage_classes`) and that the holder is not inside any `ImplicitUpdate`'s `system.model` tree (see "Implicit-segment limitation" below).
3. Deletes the static `_parameters[local]` / `_buffers[local]` slot.
4. Adds `holder._nl_params[local] = NLParam(input_name=qname, tail_index=N)`.
5. Adds `qname` to the holder's per-instance `input_spec` with the original typed-wrapper class (e.g. `Scalar`).

Any wrapping `ComposedModel` picks the new input up via dependency resolution; the call boundary's `_coerce_to_input_type` wraps the raw input tensor in the right `TensorWrapper` before handing it to the leaf; the leaf's existing `self._get_param(local, nl_params, Scalar)` call reads from `nl_params[tail_index]` instead of `self.local`. No leaf-side code changes required.

For the JVP graph, identity seeds are constructed for structural inputs only (`_ForwardJacobianModule.structural_idx`); promoted inputs stay absent from the seed dict and contribute structural zeros to `J` via the default chain rule's `v.get(name, {})` empty fallback. `J`'s columns are over structural inputs only -- exactly what `aoti::Model::jacobian()` exposes.

Remaining (non-promoted) `nn.Parameter` instances are converted to persistent buffers via `_freeze_remaining_parameters_to_buffers` immediately after promotion, so `torch.export` bakes them as constants instead of lifting them as graph inputs.

### Implicit-segment limitation

Promotion of parameters that live inside an `ImplicitUpdate`'s `system.model` tree is **rejected up-front** with a clear `NotImplementedError`. The Dense/Block equation-system wrappers (`DenseRHS`, `DenseNewtonStep`, `DenseIFT`, `BlockRHS`, ...) have fixed `(u_flat, g_flat)` forward signatures; threading a `*nl_params` tail through them requires changing those signatures and updating the C++ Newton orchestrator's loader calls. That's a separate piece of work. Parameters in forward segments of a composed model promote normally.

## Versioning

The supported schema version is `2`. The C++ loader (`include/neml2/aoti/Model.h`) holds `kSupportedSchemaVersion = 2` and refuses any other value with a clear "regenerate via neml2-compile" message. Pre-v2 caches are not backwards-compatible — the `parameters` array and `param_inputs` per-segment field are required.

## References

- `python/neml2/native/cli/aoti_export.py` — exporter implementation (`export_model_for_aoti`).
- `python/neml2/native/cli/aoti_compile.py` — `neml2-compile` CLI + `.i`-stub emitter.
- `include/neml2/aoti/Model.h` + `src/neml2/aoti/Model.cxx` — bare C++ runtime.
- `python/neml2/native/export.py::compile_model` — torch.export → AOTInductor wrapper used by the segment compilers.
