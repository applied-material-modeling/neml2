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

(tutorials-extension-composition)=
# Composing your model with others

The previous three tutorials walked through writing a fresh `Model`:
[](tutorials-extension-arguments) declared its inputs, outputs,
parameters, and a buffer; [](tutorials-extension-input-files) wired
the schema up to the HIT factory; [](tutorials-extension-forward)
implemented the math. The payoff for all that scaffolding is that the
custom `ProjectileAcceleration` now behaves like every other entry in
the catalog — it can be dropped into a
[`ComposedModel`](models-ComposedModel) alongside built-ins, fed into
an `ImplicitUpdate`, and driven across time by a `TransientDriver`,
with NEML2 resolving the wiring automatically by variable name.

## The full trajectory problem

A complete projectile trajectory model needs more than the
acceleration formula. Let $\boldsymbol{x}$ be the position and
$\boldsymbol{v}$ the velocity; the implicit time-integrated system is

$$
\begin{aligned}
  \dot{\boldsymbol{x}} & = \boldsymbol{v}, \\
  \dot{\boldsymbol{v}} & = \boldsymbol{a}
                       = \boldsymbol{g} - \mu \boldsymbol{v}, \\
  \mathbf{r}_{\boldsymbol{x}} &= \boldsymbol{x}
                       - \boldsymbol{x}_n
                       - (t - t_n)\,\dot{\boldsymbol{x}}, \\
  \mathbf{r}_{\boldsymbol{v}} &= \boldsymbol{v}
                       - \boldsymbol{v}_n
                       - (t - t_n)\,\dot{\boldsymbol{v}}, \\
  (\boldsymbol{x}, \boldsymbol{v}) &= \mathop{\mathrm{root}}_{\boldsymbol{x}, \boldsymbol{v}}\,\mathbf{r}.
\end{aligned}
$$

The four pieces map onto NEML2 building blocks as follows:

| Equation | Building block |
|----------|----------------|
| $\dot{\boldsymbol{v}} = \boldsymbol{g} - \mu \boldsymbol{v}$ | The custom `ProjectileAcceleration` from the previous tutorials. |
| $\dot{\boldsymbol{x}} = \boldsymbol{v}$ + the two backward-Euler residuals | Two [`VecBackwardEulerTimeIntegration`](models-VecBackwardEulerTimeIntegration) instances, one per state variable. |
| `root` over $(\boldsymbol{x}, \boldsymbol{v})$ | An [`ImplicitUpdate`](models-ImplicitUpdate) wrapping the residual system. |
| Recursion through time | A [`TransientDriver`](drivers-TransientDriver) sweeping a prescribed time array. |

The point of this tutorial: each piece lives in `[Models]`, named and
typed independently; the dependency resolver glues them together by
matching producer-output names against consumer-input names. The
custom `ProjectileAcceleration` is no different from a built-in.

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

A few things to notice in this file:

1. **`ProjectileAcceleration` slots in next to the built-ins.** The
   `eq2` block uses our custom type just like `eq3a` / `eq3b` use the
   built-in `VecBackwardEulerTimeIntegration`. The registry doesn't
   distinguish them.

2. **Wiring is implicit, through variable names.** `eq2` produces
   `acceleration = 'a'`; the velocity-update integrator `eq3b` consumes
   `rate = 'a'`. Similarly `eq3a` consumes `rate = 'v'` from the
   driver and `eq3b` writes `variable = 'v'`. The dependency resolver
   sees those matches and threads the values through internally —
   `a` and `v` are *not* free inputs of the composed `system`.

3. **`ComposedModel` is itself a `Model`.** `eq3` glues the two
   integrators together; `system` then glues `eq3` to `eq2`. Either
   can be loaded from Python and called like any leaf.

4. **`ImplicitUpdate` wraps the system in a Newton solve.** The four
   residual equations (two from each integrator's `*_residual`
   output) form the nonlinear system the `Newton` solver drives to
   zero at each step.

5. **`TransientDriver` recurses through time.** It calls
   `eq4 = ImplicitUpdate(...)` once per timestep, threading the
   previous step's state forward as the `~1` inputs the integrators
   expect.

## Inspecting the wiring

Before evaluating anything, ask `neml2-inspect` to print the
resolved input/output graph of the composed `system`. Wiring bugs —
a typo in a variable name, a forgotten rename — show up here as
extra unbound inputs or missing outputs, in a tiny fraction of the
time it takes to read a runtime traceback:

```{code-cell} ipython3
import sys, os
sys.path.insert(0, os.getcwd())

import projectile  # registers ProjectileAcceleration with the native factory

!neml2-inspect input.i system
```

Read the output top to bottom:

- **Inputs.** The state-at-previous-step inputs (`x~1`, `v~1`, `t~1`)
  and the new time `t`. `v`, `a`, `x` are *not* free — they're all
  resolved inside the composed system.
- **Outputs.** The two residuals (`x_residual`, `v_residual`) the
  `ImplicitUpdate` will drive to zero.

## Running the trajectory

The `TransientDriver` loads the entire model graph and recurses the
implicit update across the prescribed time array. The example below
launches five projectiles with different viscosities (broadcasting
the `mu` parameter to shape `(5,)`) — the same custom leaf handles
the full batch in one call:

```{code-cell} ipython3
import neml2

factory = neml2.load_input("input.i")
driver = factory.get_driver("driver")
driver.run()
```

The driver writes the full state trajectory to `result.pt` (declared
via `save_as` on the driver block). Load it back to see the shape:

```{code-cell} ipython3
import torch

result = torch.load("result.pt", weights_only=True)
{k: tuple(v.shape) for k, v in result.items() if hasattr(v, "shape")}
```

Each per-step entry carries a leading time axis and the trailing
batch of 5 viscosities — the same composed `system` evaluated 100
times across all five projectiles simultaneously.

## Where to go next

- [](tutorials-models-composition) walks through composition more
  broadly — the producer/consumer rules the dependency resolver
  follows, multi-step chains, and parameter binding via output names.
  This page covered the same machinery from the custom-model angle;
  that page covers the wider story.
- Composed models are themselves `Model`s, so they can be exported,
  compiled to AOT-Inductor, or recursively re-composed inside larger
  graphs without any changes on the consumer side.
