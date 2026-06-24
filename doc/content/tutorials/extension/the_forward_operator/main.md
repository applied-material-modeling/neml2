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

(tutorials-extension-forward)=
# The forward operator

The previous tutorial declared the projectile model's inputs, outputs,
and parameters. This one fills in the math: a `forward()` method that
turns a velocity into an acceleration.

## The `forward` method

`forward()` is the function NEML2 calls to evaluate your model. The
positional arguments are the inputs in the order you declared them in
the schema — each one is already a typed wrapper (`Vec`, `SR2`, …) the
framework built for you. Return the outputs in schema order.

There's also a `v=None` keyword in the signature — that's the
chain-rule hook for first derivatives, covered later on this page.
Until you need derivatives, you can leave it alone; calls without
`v` get back just the outputs.

## Implementation

Recall the equation for the projectile model:

$$
  \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}.
$$

`self.g` is the gravity vector (a buffer, read directly), and the
velocity comes in as the first positional argument. The drag
coefficient `mu` is a *parameter*, so it is read through
`self._get_param("mu", nl_params, Scalar)` rather than `self.mu` —
more on that below. The whole forward is:

```{literalinclude} projectile.py
:language: python
:caption: projectile.py
:pyobject: ProjectileAcceleration.forward
```

The first line reads the parameter:
`mu = self._get_param("mu", nl_params, Scalar)`. Always read a
*parameter* this way inside `forward` — never as `self.mu`. A bare
`self.mu` is rejected by a runtime guard, because it bypasses
`_get_param`'s static-or-promoted dispatch: the moment `mu` is
promoted to a runtime input (`neml2-compile -p`) the static
`nn.Parameter` no longer exists and the attribute read breaks.
`_get_param` works for both static and promoted parameters — it pulls
the value from `self` when static and from the `*nl_params` pack when
promoted — so the leaf stays promotion-compatible. (Buffers like
`self.g` are *not* parameters, so reading them directly is fine.) That
is also why the signature is `def forward(self, v_in, *nl_params,
v=None)`: the `*nl_params` pack is where promoted parameters arrive.

The next line is the physics: `Vec - Scalar * Vec` gives back a `Vec`,
batched or not. If `v is None` (the usual case) the method returns and
you're done.

The `else` branch is the chain-rule hook. `actions` maps each input
variable to a small function that takes an incoming tangent
(something the same shape as that input) and returns its
contribution to the output's tangent. For this model the math is
simple: $\partial \boldsymbol{a}/\partial \boldsymbol{v} = -\mu I$,
so the closure is `lambda V: -mu * V`, capturing the local `mu` we
read at the top. `apply_chain_rule` then sums the contribution against
any tangents the caller seeded on `v`, without ever building the full
Jacobian matrix in memory.

## Evaluation

That's the whole model. Load it the same way you'd load any built-in
type — `neml2.load_model` finds it through the factory as long as the
module that registers it has been imported.

The input file from the previous tutorial wires the model into HIT:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

Import the module so the class registers, then load and call:

```{code-cell} ipython3
import sys, os
sys.path.insert(0, os.getcwd())

import projectile  # registers ProjectileAcceleration with the native factory
import neml2

model = neml2.load_model("input.i", "accel")
model
```

```{code-cell} ipython3
import torch
from neml2.types import Vec

vel = Vec.fill(10.0, 2.0, 0.0)
accel = model(vel)
accel
```

The result is $\boldsymbol{g} - \mu \boldsymbol{v} = (0, -9.81, 0) -
0.001 \cdot (10, 2, 0) = (-0.01, -9.812, 0)$ — the typed wrapper
preserves the `Vec` shape on the way out.

## First derivatives

Because `forward()` implements the `v` branch, the same model can hand
back directional derivatives with no extra wiring. Seed a tangent on
the velocity input and the JVP comes back through `v_out`:

```{code-cell} ipython3
# Seed the identity on Vec to read off the full Jacobian column ∂a/∂v.
seed = Vec(torch.eye(3, dtype=torch.float64))
accel, v_out = model(vel, v={"v": {"velocity_leaf": seed}})
v_out["a"]["velocity_leaf"].data
```

This matches the analytical $\partial \boldsymbol{a}/\partial
\boldsymbol{v} = -\mu I = -0.001\, I_3$ exactly.

## Auto-deriving the chain rule with `request_AD`

Hand-writing `actions` is the right call when the local Jacobian is simple, like
the projectile's $-\mu I$. But sometimes it isn't — a constitutive law with many
coupled terms, or a **machine-learning surrogate** (a trained `torch.nn.Module`)
whose derivative you would never want to differentiate by hand. For those, a leaf
can declare `request_AD()` and write *only* the value forward; the framework
supplies the first-order chain rule for you by reverse-mode automatic
differentiation:

```python
class MySurrogate(Model):
    hit = HitSchema(...)            # same input/output declarations as usual

    def __post_init__(self):
        self.request_AD()           # auto-derive d(output)/d(input) for all pairs

    def forward(self, x, *nl_params):
        return self.net(x.data)     # value only -- no `v=` branch, no `actions`
```

`request_AD()` with no arguments covers every `(output, input)` pair; pass
`outputs=[...]` / `inputs=[...]` to auto-derive only a subset and hand-write the
rest. The result is indistinguishable from a hand-written chain rule and behaves
identically on **every route** — eager (`py-eager` / `cpp-eager`) and
AOT-compiled (`py-aoti` / `cpp-aoti` / `cpp-dispatch`). It slots into the *same*
forward-mode chain-rule graph the framework already uses: neighbouring analytic
leaves keep their hand-written `actions`, and only the request_AD leaf's
reverse-mode local Jacobian is traced inline (and lowered through AOTInductor on
the compiled routes). The `tests/regression/_fixtures/SurrogateFlowRate.py` fixture
is a worked example wrapping a Python surrogate as a NEML2 flow rate.

A few things to know:

- **First-order only.** `request_AD` supplies the `v=` channel
  ($\partial\,\text{out}/\partial\,\text{in}$). A leaf that must provide the
  second-order chain rule (i.e. one used inside a `Normality` wrap) still
  hand-writes it.
- **Reverse-mode under the hood.** It is the one autodiff that survives
  `torch.export` → AOTInductor. If your differentiated path uses a saved-output
  op (`exp` / `sqrt` / `tanh` / reciprocal), route it through the AOTI-safe
  variants in `neml2.types.functions` (e.g. `exp_ad`) so the compiled routes
  lower (see the upstream-bug note in that module); eager is unaffected.
- **AOTI compile.** Pass `-d` to `neml2-compile` to bake the derivative graph,
  exactly as for an analytic model — `request_AD` changes *how* the Jacobian is
  computed, not *whether* it is compiled. This holds even for a request_AD leaf
  inside an `ImplicitUpdate` residual: the Newton-step / implicit-function-theorem
  graphs differentiate the residual (which contains the leaf) and lower the same
  way.

## Driving the model from a unit-test input

`ModelUnitTest` is the usual way to pin a custom model's behavior in
CI. It loads the model, runs it on inputs you supply, checks the
outputs, and cross-checks the derivatives against PyTorch's autograd:

```{literalinclude} unit_test.i
:language: ini
:caption: unit_test.i
```

```{code-cell} ipython3
from neml2.drivers.ModelUnitTest import ModelUnitTest

report = ModelUnitTest.from_file("unit_test.i").run()
print(f"value checks: {report.value_checks}, JVP checks: {report.jvp_checks}")
```

If both counters are positive (and the cell didn't raise), every
value and every JVP matched. A zero on either side means that check
was skipped, not that it failed silently.

## Where to go next

- The next tutorial, [](tutorials-extension-composition), shows how to
  glue several models together so a dependency resolver wires their
  inputs and outputs and threads the chain rule for you.
