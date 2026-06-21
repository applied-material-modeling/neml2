(py-aoti)=
# `py-aoti` — compiled model from Python

`py-aoti` runs a model that has already been lowered to an AOT-Inductor
package (`.pt2` + metadata) by `neml2-compile`, from Python — no NEML2
source model, no Newton-in-Python, no recompile per call. Use it to deploy a
calibrated model behind a Python service, or to reproduce the C++ runtime's
numerics from a notebook. It exposes the same `forward` / `jvp` / `jacobian`
surface as the eager routes and supports sub-batch models.

The on-disk artifact this route consumes — the `.pt2` layout, the metadata
schema, parameter promotion, device/dtype pinning — is documented once in
[](aoti-packages); how to produce it and set up your environment is in
[](python-integration). This page is the Python loading-and-calling API.

## Two entry points

To drive the artifact through the normal HIT machinery (e.g. a
`TransientDriver`), load the stub with `neml2.load_model` — the `AOTIModel`
shim resolves the per-device subfolder for the current default device:

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

## Call surface

The bare runtime centers on three call paths plus introspection properties
(`input_names`, `output_names`, `input_base_shapes`, `output_base_shapes`,
`device`, `dtype`), a mutable `named_parameters()` map, and a
`set_parameter(name, tensor)` helper for replacing a promoted parameter
wholesale:

| Operation               | Returns                                             |
| :---------------------- | :-------------------------------------------------- |
| `forward(inputs)`       | One tensor per output name, at `(*B, *out_base)`.   |
| `jvp(inputs, tangents)` | `(outputs, J @ v)` over the **requested** pairs. Missing tangent keys → zero. |
| `jacobian(inputs)`      | `(outputs, J)` with `J[out][in]` the block for each **requested** pair, `(*B, *out_base, *in_base)`. |
| `named_parameters()`    | Mutable dict of promoted parameters; empty if baked. |

`jvp` / `jacobian` only return the output-input pairs the artifact was
compiled with (`neml2-compile -d OUT:IN`); pairs absent from the
`derivatives` metadata are absent from the returned maps. An artifact
compiled with no `-d` raises from both. A batch-independent block (e.g. a
constant stiffness tensor) comes back unbatched (`(*out_base, *in_base)`).

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

## See also

- [](aoti-packages) — the `.pt2` package format this route loads.
- [](python-integration) — installing neml2 and obtaining a compiled package.
- [](cpp-aoti) — the same artifact, loaded from C++.
- [](tutorials-models-compiled) — end-to-end how-to: compile, load, round-trip.
- [](cli-utilities) — `neml2-compile`, `neml2-inspect`.
