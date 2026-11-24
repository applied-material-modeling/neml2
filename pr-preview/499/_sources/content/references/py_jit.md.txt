(py-jit)=
# `py-jit` — in-process `torch.compile`

`py-jit` accelerates an eager model in the running interpreter with
`torch.compile`. It produces **no artifact** — the compiled graph lives in the
process and recompiles lazily once per distinct input shape. It exists mainly
to speed up pyzag training loops (the residual / Jacobian assembly); for a
shippable compiled package use [](py-aoti) instead.

## API

```python
import neml2

model = neml2.load_model("input.i", "elasticity")
neml2.compile(model)   # in place; the same object is returned, now compiled
```

`neml2.compile(target, *, fullgraph=False, mode=None, **compile_kwargs)` accepts:

- a `Model` — its feed-forward `forward` is compiled;
- a `ModelNonlinearSystem` — its residual model is compiled (so `assemble` runs
  compiled);
- a pyzag wrapper (anything exposing a `.sys` that is a `ModelNonlinearSystem`).

Compilation is applied **in place** and uses `dynamic=False` (hard-coded — the
only setting that reliably accelerates today; it recompiles once per distinct
input shape, which is fine for a fixed training configuration). `mode` and the
other keywords forward to `torch.compile`.

Sensitivities are unaffected: `py-jit` compiles the forward graph; first- and
second-order derivatives still flow through the native chain rule. Sub-batch
models are supported.

## See also

- [](tutorials-optimization-pyzag) — the calibration loop this route accelerates.
- [](py-eager) — the uncompiled runtime `neml2.compile` wraps.
- [](py-aoti) — the ahead-of-time compiled alternative (ships an artifact).
