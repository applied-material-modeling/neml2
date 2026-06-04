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
\begin{align}
  \dot{\boldsymbol{v}} & = \boldsymbol{a}
                       = \boldsymbol{g} - \mu \boldsymbol{v}, \\
  \mathbf{r}_{\boldsymbol{x}} &= \boldsymbol{x}
                       - \boldsymbol{x}_n
                       - (t - t_n)\,\dot{\boldsymbol{x}}, \\
  \mathbf{r}_{\boldsymbol{v}} &= \boldsymbol{v}
                       - \boldsymbol{v}_n
                       - (t - t_n)\,\dot{\boldsymbol{v}}, \\
  (\boldsymbol{x}, \boldsymbol{v}) &= \mathop{\mathrm{root}}_{\boldsymbol{x}, \boldsymbol{v}}\,\mathbf{r}.
\end{align}
$$

The four pieces map onto NEML2 building blocks as follows:

| Equation                                                                   | Building block                                                                                                     |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| $\dot{\boldsymbol{v}} = \boldsymbol{g} - \mu \boldsymbol{v}$               | The custom `ProjectileAcceleration` from the previous tutorials.                                                   |
| $\dot{\boldsymbol{x}} = \boldsymbol{v}$ + the two backward-Euler residuals | Two [`VecBackwardEulerTimeIntegration`](models-VecBackwardEulerTimeIntegration) instances, one per state variable. |
| `root` over $(\boldsymbol{x}, \boldsymbol{v})$                             | An [`ImplicitUpdate`](models-ImplicitUpdate) wrapping the residual system.                                         |
| Recursion through time                                                     | A [`TransientDriver`](drivers-TransientDriver) sweeping a prescribed time array.                                   |

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
   `eq1` block uses our custom type just like `eq2` / `eq3` use the
   built-in `VecBackwardEulerTimeIntegration`. The registry doesn't
   distinguish them.

2. **Wiring is implicit, through variable names.** `eq1` produces
   `acceleration = 'a'`; the velocity-update integrator `eq3`
   consumes `rate = 'a'`. Similarly `eq2` consumes `rate = 'v'` and
   `eq3` writes `variable = 'v'`. The dependency resolver sees those
   matches and threads the values through internally — `a`, the
   intra-step `v` hand-off between `eq1` and `eq3`, and the
   integrator chain are *not* free inputs of the composed
   `residual`.

3. **`ComposedModel` glues the three leaves into one residual
   block.** The `residual` block lists `'eq1 eq2 eq3'` and exposes
   `x_residual` and `v_residual` as its outputs — these are the two
   equations the nonlinear solve will drive to zero.

4. **`NonlinearSystem` declares the unknowns / residuals pairing.**
   The `system` block names `x` and `v` as the unknowns and
   `x_residual` / `v_residual` as the corresponding residuals. The
   solver doesn't see the model directly at this layer — only the
   variable roles.

5. **`ImplicitUpdate` wraps the system in a Newton solve.** `eq4`
   binds the equation system to a solver and exposes itself as
   another `Model` — its outputs are the converged `x` and `v` at
   the current step.

6. **`TransientDriver` recurses through time.** It calls
   `eq4 = ImplicitUpdate(...)` once per timestep, threading the
   previous step's state forward as the `~1` inputs the integrators
   expect.

## Inspecting the wiring

Before evaluating anything, ask `neml2-inspect` to print the
resolved input/output graph of the composed `residual` block.
Wiring bugs — a typo in a variable name, a forgotten rename — show
up here as extra unbound inputs or missing outputs, in a tiny
fraction of the time it takes to read a runtime traceback:

```{code-cell} ipython3
import sys, os
sys.path.insert(0, os.getcwd())

import projectile  # registers ProjectileAcceleration with the native factory

# `neml2-inspect` is the same tool you'd call from the shell, but a shell
# subprocess wouldn't inherit our `import projectile` and the registry
# lookup would fail. Calling the CLI's `main` in-process keeps the registry
# warm and gives the identical output.
from neml2.cli.inspect import main as _inspect_main
_inspect_main(["input.i", "residual"])
```

Read the output top to bottom:

- **Inputs.** The trial state `x` / `v`, the state-at-previous-step
  inputs (`x~1`, `v~1`, `t~1`), and the new time `t`. `a` is *not*
  free — it's resolved internally by `eq1`.
- **Outputs.** The two residuals (`x_residual`, `v_residual`) the
  `ImplicitUpdate` will drive to zero.

## Running the trajectory

The `TransientDriver` loads the entire model graph and recurses the
implicit update across the prescribed time array. The `v0` block in
the input file stacks five `Vec.fill(...)` launches into a single
`(5, 3)` initial condition via `dynamic_batch.stack`, so one driver
call covers all five projectiles:

```{code-cell} ipython3
import neml2

factory = neml2.load_input("input.i")
driver = factory.get_driver("driver")
driver.run()
```

`driver.result()` returns a flat dict keyed by `input.<step>.<var>`
/ `output.<step>.<var>`. The leading `5` on each step's position
and velocity is the batched projectile axis:

```{code-cell} ipython3
result = driver.result()
{k: tuple(v.shape) for k, v in result.items()
 if k.startswith("output.0.") or k.startswith("output.1.")}
```

## Plotting the trajectories

Stack the per-step `x` outputs along a new time axis and project
onto the $x$–$y$ plane to see all five arcs:

```{code-cell} ipython3
import torch
import matplotlib.pyplot as plt

nsteps = sum(1 for k in result if k.startswith("output.") and k.endswith(".x"))
positions = torch.stack(
    [result[f"output.{i}.x"] for i in range(1, nsteps + 1)]
).detach()

fig, ax = plt.subplots(figsize=(6, 4))
for j in range(positions.shape[1]):
    ax.plot(positions[:, j, 0], positions[:, j, 1], "-o", markersize=2,
            label=f"projectile {j}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axhline(0.0, color="black", linewidth=0.5)
ax.grid(True)
ax.legend(loc="upper right", fontsize="small")
fig.tight_layout()
plt.show()
```

Each trajectory bends under gravity ($g$) and decays under linear
drag ($\mu$) — faster launches climb higher and travel farther
before drag pulls them down. The composed graph runs the whole
batch in one Newton solve per step; the custom
`ProjectileAcceleration` leaf does its share without knowing
anything about the integrator sitting next to it or the batch
dimension threading through.

## Where to go next

- [](tutorials-models-composition) walks through composition more
  broadly — the producer/consumer rules the dependency resolver
  follows, multi-step chains, and parameter binding via output names.
  This page covered the same machinery from the custom-model angle;
  that page covers the wider story.
- Composed models are themselves `Model`s, so they can be exported,
  compiled to AOT-Inductor, or recursively re-composed inside larger
  graphs without any changes on the consumer side.
