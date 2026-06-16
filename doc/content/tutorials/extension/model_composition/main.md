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
# Composing with existing models

You'll plug the `ProjectileAcceleration` model from the previous
tutorials into a full trajectory simulation — alongside built-in time
integrators, a Newton solve, and a transient driver — and let NEML2
wire everything up by matching variable names.

## The full trajectory problem

A complete projectile trajectory model needs more than the
acceleration formula. Let $\boldsymbol{x}$ be the position and
$\boldsymbol{v}$ the velocity; the implicit time-integrated system is

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

The four pieces map onto NEML2 building blocks as follows:

| Equation                                                                   | Building block                                                                                                     |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| $\dot{\boldsymbol{v}} = \boldsymbol{g} - \mu \boldsymbol{v}$               | The custom `ProjectileAcceleration` from the previous tutorials.                                                   |
| $\dot{\boldsymbol{x}} = \boldsymbol{v}$ + the two backward-Euler residuals | Two [`VecBackwardEulerTimeIntegration`](models-VecBackwardEulerTimeIntegration) instances, one per state variable. |
| `root` over $(\boldsymbol{x}, \boldsymbol{v})$                             | An [`ImplicitUpdate`](models-ImplicitUpdate) wrapping the residual system.                                         |
| Recursion through time                                                     | A [`TransientDriver`](drivers-TransientDriver) sweeping a prescribed time array.                                   |

Each piece is its own block in `[Models]`; NEML2 connects them by
matching output names to input names. The custom
`ProjectileAcceleration` plugs in the same way a built-in would.

## The input file

```{literalinclude} input.i
:language: ini
:caption: input.i
```

A few things to notice in this file:

- **`ProjectileAcceleration` sits next to the built-ins.** The `eq1`
  block uses our custom type just like `eq2` / `eq3` use the built-in
  `VecBackwardEulerTimeIntegration`.
- **Wiring is implicit, through variable names.** `eq1` writes
  `acceleration = 'a'`, and the velocity integrator `eq3` reads
  `rate = 'a'`. Same story for `v`. NEML2 sees the matches and
  threads the values through, so `a` is *not* a free input of the
  composed `residual`.
- **The rest of the blocks (`residual`, `system`, `eq4`, `driver`)**
  bundle the three leaves into a single residual, declare which
  variables are unknowns, wrap the system in a Newton solve, and
  recurse it through time. See
  [`ComposedModel`](models-ComposedModel),
  [`ImplicitUpdate`](models-ImplicitUpdate), and
  [`TransientDriver`](drivers-TransientDriver) for the full option
  surface.

## Inspecting the wiring

Before running anything, use `neml2-inspect` to print the resolved
input/output graph of the composed `residual` block. Typos and
forgotten renames show up here as unbound inputs or missing outputs,
which is much easier to debug than a deep runtime traceback:

```{code-cell} ipython3
import sys, os
sys.path.insert(0, os.getcwd())

import projectile  # registers ProjectileAcceleration with the native factory

# `neml2-inspect` is the same CLI you'd call from the shell (with
# `--load projectile.py` to register the extension). We call the CLI's
# `main` in-process here so the same `import projectile` above does
# double duty and the output prints inline in the notebook.
from neml2.cli.inspect import main as _inspect_main
_inspect_main(["input.i", "residual"])
```

The inputs are the trial state (`x`, `v`), the previous step's state
(`x~1`, `v~1`, `t~1`), and the new time `t`. Note that `a` does not
appear — it's resolved internally by `eq1`. The outputs are the two
residuals `x_residual` and `v_residual` that the `ImplicitUpdate`
will drive to zero.

## Running the trajectory

The input file sets up a *bag-of-balls* scenario: three bags with
different drag coefficients $\mu$, each holding five balls thrown at
different launch velocities. That's 15 trajectories, all evaluated in
a single batched Newton solve per step.

The batching falls out of broadcasting in `[Tensors]`: the launch
velocity is a `Vec` whose dynamic-batch shape is `(5, 1)` (five
launches, one placeholder viscosity slot), and `mu` is a `Scalar`
with dynamic-batch shape `(3,)`. When they meet inside `eq1`, the
size-1 slot broadcasts against the three viscosities for a `(5, 3)`
evaluation — one composed graph, one Newton iterate per step.

```{code-cell} ipython3
import neml2

factory = neml2.load_input("input.i")
driver = factory.get_driver("driver")
driver.run()
```

`driver.result()` returns a flat dict keyed by `input.<step>.<var>`
/ `output.<step>.<var>`. The leading `(5, 3)` on each step's
position and velocity is the batched (ball, bag) grid:

```{code-cell} ipython3
result = driver.result()
{k: tuple(v.shape) for k, v in result.items()
 if k.startswith("output.0.") or k.startswith("output.1.")}
```

## Plotting the trajectories

Stack the per-step `x` outputs along a new leading time axis and
project onto the $x$–$y$ plane. One subplot per bag (viscosity),
five trajectories each:

```{code-cell} ipython3
import torch
import matplotlib.pyplot as plt

nsteps = sum(1 for k in result if k.startswith("output.") and k.endswith(".x"))
positions = torch.stack(
    [result[f"output.{i}.x"] for i in range(1, nsteps + 1)]
).detach()  # shape (nsteps, 5 launches, 3 viscosities, 3 components)

mu_values = factory.get_tensor("mu").data.tolist()
n_launches = positions.shape[1]
n_visc = positions.shape[2]

fig, axes = plt.subplots(1, n_visc, figsize=(12, 4), sharey=True)
for k, ax in enumerate(axes):
    for j in range(n_launches):
        ax.plot(positions[:, j, k, 0], positions[:, j, k, 1], "-o", markersize=2,
                label=f"launch {j}")
    ax.set_xlabel("x")
    ax.set_title(rf"$\mu = {mu_values[k]:g}$")
    ax.set_xlim(-1, 26)
    ax.set_ylim(-12, 4)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.grid(True)
axes[0].set_ylabel("y")
axes[-1].legend(loc="upper right", fontsize="small")
fig.tight_layout()
plt.show()
```

The lightly-damped bag ($\mu = 0.1$) sees the balls fly furthest;
the heavily-damped bag ($\mu = 1$) drags them down within a few
meters. All 15 trajectories ran in one Newton solve per step: the
`ProjectileAcceleration` leaf is written in terms of a single
`(v, mu)` pair, and the `(5, 3)` batch grid threads through every
piece without any per-(launch, viscosity) bookkeeping in the leaf
code.

## Where to go next

- [](tutorials-models-composition) covers composition more broadly —
  the producer/consumer matching rules, multi-step chains, and
  parameter binding via output names.
- A composed model is itself a `Model`, so the same export,
  AOT-Inductor compilation, and outer composition paths apply to it
  — see [](tutorials-models-compiled) for the round trip.
