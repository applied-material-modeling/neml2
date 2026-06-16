---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
mystnb:
  execution_mode: cache
---

(tutorials-models-compiled)=
# Compiled models

You'll take a model you can already run in Python, compile it ahead of
time into a self-contained package, and call the compiled version from
another process. The compiled form runs the kernels in pre-compiled
C++ instead of the eager PyTorch dispatcher, so it's the form to use
once a model is locked in and you want it fast.

## The pipeline

```
input.i ──► neml2-compile ──► elasticity.pt2          (AOTI-compiled kernels)
                              elasticity_jvp.pt2      (flat dout/din graph)
                              elasticity_meta.json    (variable layout, dtype, device)
                              elasticity_aoti.i       (drop-in HIT stub)
```

`neml2-compile` emits all four files into the output directory. The
`elasticity_aoti.i` stub is a copy of the original input with the
`[Models]/elasticity` block replaced by an `AOTIModel` shim pointing at
the metadata — everything else (drivers, settings, tensors) is copied
through verbatim, so the stub is a drop-in replacement anywhere a
`Driver` consumes a model by name.

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

## Compiling from the shell

`neml2-compile` (see [](cli-utilities)) builds the package in one
invocation:

```{code-cell} ipython3
!neml2-compile input.i --model elasticity
```

Compilation is a one-time cost (Inductor + C++ compile, typically
seconds); the resulting artifact loads quickly on every subsequent
run. With no `--output-dir` the artifacts land in `aoti/<model>/`:

```{code-cell} ipython3
!ls aoti/elasticity
```

## Loading the compiled model

The stub looks like any other HIT input file, so `neml2.load_model`
loads it the usual way:

```{code-cell} ipython3
import neml2

compiled = neml2.load_model("aoti/elasticity/elasticity_aoti.i", "elasticity")
compiled
```

The shim behaves like a native model — same inputs, same outputs,
same call convention — but underneath it dispatches to the compiled
`.pt2` instead of running Python:

```{code-cell} ipython3
print("inputs :", list(compiled.input_spec))
print("outputs:", list(compiled.output_spec))
```

## Round-trip equivalence

Evaluate the compiled and eager forms on the same strain and check
they agree. The compiled model always returns a tuple (even with a
single output), so we unpack it:

```{code-cell} ipython3
import torch
from neml2.types import SR2

eager = neml2.load_model("input.i", "elasticity")

strain = SR2(torch.tensor(
    [
        [0.01,  0.0,    0.0, 0.0, 0.0, 0.0  ],
        [0.005, -0.002, 0.0, 0.0, 0.0, 0.001],
    ],
    dtype=torch.float64,
))

(stress_compiled,) = compiled(strain)
stress_eager = eager(strain)

print("compiled:", stress_compiled.data)
print("eager   :", stress_eager.data)
print("match   :", torch.allclose(stress_compiled.data, stress_eager.data,
                                  rtol=1e-12, atol=1e-12))
```

The compiled artifact matches eager output to machine precision. The
leading batch dimension is dynamic, so the same artifact handles any
batch size from 1 to roughly a million without recompilation.

## Jacobian and Jacobian-vector product

Alongside `forward`, the compiled artifact also exposes `jvp` (outputs
plus a directional derivative `J @ v`) and `jacobian` (outputs plus
the dense Jacobian). Both live on the underlying C++ binding,
reachable through the shim's inner attribute:

```{code-cell} ipython3
binding = compiled._inner          # the bare ``neml2.aoti.Model`` runtime

# Tangent on the input — same shape as the input itself.
strain_t = torch.zeros_like(strain.data)
strain_t[:, 0] = 1.0               # probe d(stress)/d(epsilon_xx)

out, jvp = binding.jvp({"strain": strain.data}, {"strain": strain_t})
print("output[stress] :", out["stress"][:, :3])
print("J @ v [stress] :", jvp["stress"][:, :3])
```

```{code-cell} ipython3
out, J = binding.jacobian({"strain": strain.data})
print("output[stress] :", out["stress"][:, :3])
print("Jacobian shape :", J.shape)             # (batch, n_out, n_in)
print("J[0] (6x6):\n", J[0])
```

Use `jvp` when you only need one direction of the gradient; use
`jacobian` when you need the whole matrix.

## Promoting parameters

By default every parameter is baked into the compiled graph as a
constant. That's fastest, but it also means you can't change them at
runtime. To keep one mutable, promote it at compile time with `-p`:

```{code-cell} ipython3
!neml2-compile input.i --model elasticity --output-dir aoti_promoted -p E
```

The promoted artifact loads the same way; the difference is that the
named parameter is reachable from Python and editing it in place is
reflected on the next forward call:

```{code-cell} ipython3
m = neml2.load_model("aoti_promoted/elasticity_aoti.i", "elasticity")

# Initial value comes from the snapshot taken at compile time.
(stress0,) = m(strain)

# Mutate E in place on the underlying binding. The next call sees it.
m._inner.named_parameters()["E"].fill_(100e3)
(stress1,) = m(strain)

print("initial E=200e3 stress[0,0]:", stress0.data[0, 0].item())
print("after   E=100e3 stress[0,0]:", stress1.data[0, 0].item())
print("ratio (expect 0.5):         ", (stress1.data[0, 0] / stress0.data[0, 0]).item())
```

Each promoted name adds a small per-call cost (it becomes a graph
input rather than a baked constant), but the compiled artifact is
typically still much faster than eager. Promotion doesn't work for
parameters inside an `ImplicitUpdate` segment — the compiler will
refuse them with a clear error.

## Trade-offs vs eager

|                       | Eager                                  | AOTI                                            |
|---                    |---                                     |---                                              |
| Per-call overhead     | Python + PyTorch dispatcher per op     | Thin Python shim into pre-compiled C++ kernels  |
| Build cost            | Zero                                   | Seconds (Inductor + C++ compile, once)          |
| Parameter mutation    | Free — they're live attributes         | Only for `-p` names; rest are baked             |
| Autograd              | Yes (full eager AD)                    | No (forward-only; `jvp` / `jacobian` sidecars)  |
| Device/dtype          | Switchable at runtime via `.to(...)`   | Pinned at compile time (`--device`, `--dtype`)  |
| Portability           | Needs the full Python source           | Needs only `.pt2` + metadata + the C++ runtime  |

Rule of thumb: stay in eager mode while you're still iterating on a
model. Compile it once you're ready to run it at scale — driver
runs, large batch sweeps, the inner loop of a finite-element kernel.

## Where to go next

- [](tutorials-models-implicit-model) covers `ImplicitUpdate` models;
  the same compile flow handles them — each implicit segment produces
  its own residual and Newton-step artifacts and the runtime
  orchestrates the Newton solve internally.
- [](tutorials-models-transient-driver) shows how a compiled model
  plugs straight into a `TransientDriver` — the driver doesn't care
  whether its model is eager or AOTI-backed.
- [](tutorials-models-evaluation-device) covers the device choice
  itself; with AOTI the device is baked at compile time, so pick
  carefully.
