(py-eager)=
# `py-eager` — eager Python

`py-eager` is the default runtime: load a model from its input file and call it
like any `torch.nn.Module`, executing eager PyTorch with no compile step. It is
the path you reach for during development, in tests, and when training with
PyTorch autograd. It supports the full `forward` / `jvp` / `jacobian` surface,
any device, and sub-batch models (e.g. crystal plasticity).

Set-up — installing neml2, device placement — is in [](python-integration).

## Load and call

```python
import neml2
from neml2.types import SR2

model = neml2.load_model("input.i", "elasticity")
stress = model(SR2.fill(0.01, 0.0, 0.0, 0.0, 0.0, 0.0))
```

Inputs and outputs are typed tensor wrappers (`Scalar`, `SR2`, `R2`, …) from
`neml2.types`; `Scalar(<number>)` accepts a plain Python number. The model is a
live `nn.Module`, so batched inputs, `.to(device=…)`, and `named_parameters()`
all behave the way they do for any torch module.

## Sensitivities

`model.jvp` and `model.jacobian` are the typed forward-mode derivative surface —
the same `forward` / `jvp` / `jacobian` API the compiled and embedded routes
expose, so derivative code reads identically on every route:

```python
model = neml2.load_model("input.i", "elasticity")

# Jacobian-vector product: one cheap directional derivative.
outputs, jvp = model.jvp({"strain": eps}, {"strain": deps})
#   jvp["stress"] is an SR2 — the directional derivative of stress along deps.

# Full Jacobian: every (output, input) block.
outputs, J = model.jacobian({"strain": eps})
#   J["stress"]["strain"] is a typed block d(stress)/d(strain).
```

Inputs and tangents are keyed by variable name and accept typed wrappers or raw
tensors. `jvp` returns each directional derivative as a wrapper of the output's
type; `jacobian` returns the nested `{output: {input: block}}` map, each block a
dynamic-base `Tensor` of shape `(*batch, *sub_batch, *out_base, *in_base)`.
Sub-batch models (e.g. crystal plasticity) are supported — the per-site axis
stays in the block's sub-batch region. The reverse-mode parameter derivatives
$\partial(\text{output})/\partial(\text{parameter})$ are a separate surface:
`param_jacobian` / `param_vjp`.

Two lower-level paths remain for when you need them:

- **Chain rule** — the first/second-order sensitivity dicts threaded through
  `forward(..., v=, v2=, vh=)` propagate tangents analytically through the
  composed model; `jvp` / `jacobian` are thin typed wrappers over exactly this.
- **Autograd** — because evaluation is eager, ordinary `torch.autograd` works
  end to end; this is what the calibration tutorials use. See
  [](tutorials-optimization-autograd).

## Consumers

Most shell and Python entry points run *on* this route rather than being
separate runtimes: `neml2-run` and the `Driver` classes (`TransientDriver`,
`ModelUnitTest`, `Verification`) step a model through a load history on
`py-eager`, and the pyzag adapter calibrates on it (optionally accelerated to
[](py-jit)).

## See also

- [](tutorials-models-running-your-first-model) — line-by-line first run.
- [](tutorials-models-transient-driver) — driving a model over a load history.
- [](python-integration) — install and set up neml2 in a Python app.
- [](cli-utilities) — `neml2-run` / `neml2-inspect`.
