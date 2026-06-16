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

(tutorials-models-implicit-model)=
# Implicit models

You'll build a Perzyna viscoplasticity update whose plastic strain is
defined as the root of a nonlinear residual, wrap it in an
`ImplicitUpdate` so a Newton solver finds that root, and then
differentiate through the solve.

Stiff viscoplastic flow is the canonical case: the plastic-strain
increment depends on the stress *after* the increment, which in turn
depends on the plastic strain — so the update has no closed form.
The standard move is to write a residual that vanishes at the right
answer and let a solver find it; in this tutorial the residual is
assembled from leaves already in NEML2's catalog.

## The physics

The Perzyna viscoplastic update we will solve is, in continuous form,

\begin{align}
  \boldsymbol{\varepsilon}^e &= \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \\
  \boldsymbol{\sigma} &= 3K\,\operatorname{vol}\boldsymbol{\varepsilon}^e
                       + 2G\,\operatorname{dev}\boldsymbol{\varepsilon}^e, \\
  \bar{\sigma} &= J_2(\boldsymbol{\sigma}), \\
  f^p &= \bar{\sigma} - \sigma_y, \\
  \boldsymbol{N} &= \partial f^p / \partial \boldsymbol{\sigma}, \\
  \dot{\gamma} &= \left(\frac{\langle f^p \rangle}{\eta}\right)^{n}, \\
  \dot{\boldsymbol{\varepsilon}}^p &= \dot{\gamma}\,\boldsymbol{N}.
\end{align}

Backward-Euler time integration converts the rate equation
$\dot{\boldsymbol{\varepsilon}}^p = \dot{\gamma} \boldsymbol{N}$ into
the residual

$$
  \mathbf{r}(\boldsymbol{\varepsilon}^p)
  = \boldsymbol{\varepsilon}^p
    - \boldsymbol{\varepsilon}^p_n
    - (t - t_n)\,\dot{\boldsymbol{\varepsilon}}^p,
$$

whose root is the updated plastic strain $\boldsymbol{\varepsilon}^p$.
Everything that feeds $\dot{\boldsymbol{\varepsilon}}^p$ — elasticity,
yield function, normality, the flow rate — depends on the unknown
$\boldsymbol{\varepsilon}^p$, which is what makes the problem
implicit. Solving this residual is the **return-mapping algorithm**
familiar from the solid-mechanics literature.

## Building the residual

Each line in the system above maps onto a model in NEML2's catalog,
and the residual is the `ComposedModel` that wires them together. The
full input file:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The `[system]` block is the residual model. Loading it gives back a
regular model whose only output is `plastic_strain_residual`:

```{code-cell} ipython3
import torch
import neml2
from neml2.types import SR2, Scalar

torch.set_default_dtype(torch.float64)

system = neml2.load_model("input.i", "system")
system
```

```{code-cell} ipython3
list(system.input_spec), list(system.output_spec)
```

The inputs are a candidate `plastic_strain` (the unknown we'll solve
for), the prescribed `strain`, the previous-step `plastic_strain~1`,
and the current and previous times `t`, `t~1`. Evaluating `system`
plugs those into the algebra above and returns the residual at that
guess:

```{code-cell} ipython3
strain          = SR2.fill(0.01, 0.005, -0.001)   # prescribed total strain
guess           = SR2.zeros()                     # initial guess for ε^p
plastic_strain_n = SR2.zeros()
t               = Scalar(1.0)
t_n             = Scalar(0.0)

(residual_at_guess,) = system(strain, guess, plastic_strain_n, t, t_n)
residual_at_guess
```

The residual is far from zero — the guess `ε^p = 0` is not a root.
The next section wraps the same residual in an `ImplicitUpdate` so a
Newton solver finds the root for us.

## Wrapping the residual in an `ImplicitUpdate`

The input file adds three pieces around the residual: an
`[EquationSystems]` block that names the unknowns, a nonlinear solver
(here [`Newton`](solvers-Newton); see the solver catalog for the other
choices), and a `[model]` block with `type = ImplicitUpdate` that
points at the two. Re-reading the same input file but asking for
`model` returns the wrapped object:

```{code-cell} ipython3
model = neml2.load_model("input.i", "model")
model
```

The inputs are the same as before — `strain`, `plastic_strain`, `t`,
… — but `plastic_strain` is now also an *output*: the solver produces
it. You still pass one in on the call so the solver has an initial
guess to start from.

```{code-cell} ipython3
list(model.input_spec), list(model.output_spec)
```

Calling `model` runs the Newton solve and returns the converged
plastic strain:

```{code-cell} ipython3
(plastic_strain,) = model(strain, guess, plastic_strain_n, t, t_n)
plastic_strain
```

A useful sanity check: evaluate the *residual* model at the
*solver's* answer and confirm it is at solver tolerance:

```{code-cell} ipython3
(residual_at_soln,) = system(strain, plastic_strain, plastic_strain_n, t, t_n)
residual_at_soln.data.norm().item()
```

That is within the `abs_tol = 1e-10` set on the `Newton` block, as
expected.

## Vectorized solves

`ImplicitUpdate` batches the same way every other NEML2 model does.
Stack `N` prescribed strains into a leading batch dimension, pass the
batched inputs in once, and the Newton solver advances every state
together — one step of all `N` problems, one linear solve of an
`(N, 6, 6)` Jacobian, until every problem in the batch is below
tolerance.

```{code-cell} ipython3
N = 5
strain_batch = torch.zeros(N, 6, dtype=torch.float64)
strain_batch[:, 0] = torch.linspace(0.005, 0.025, N)   # ramp ε_xx
strain_b  = SR2(strain_batch)
guess_b   = SR2(torch.zeros(N, 6, dtype=torch.float64))
psn_b     = SR2(torch.zeros(N, 6, dtype=torch.float64))
t_b       = Scalar.ones(N)
tn_b      = Scalar.zeros(N)

(plastic_strain_b,) = model(strain_b, guess_b, psn_b, t_b, tn_b)
plastic_strain_b.data
```

No Python loop — see [](tutorials-models-vectorization) for the
general pattern.

## Differentiating through the solve

The wrapped model is differentiable end-to-end through the Newton
solve. Once the residual is zero, the implicit function theorem gives

$$
  \frac{\partial \mathbf{s}}{\partial \mathbf{g}}
  = -\left(\frac{\partial \mathbf{r}}{\partial \mathbf{s}}\right)^{-1}
     \frac{\partial \mathbf{r}}{\partial \mathbf{g}},
$$

and `ImplicitUpdate` reuses the converged Jacobian factorization to
apply it in the backward pass — no unrolling of the Newton iterations.
We can drive the prescribed strain with a scalar load multiplier
`alpha`, push it through the solver, read off a scalar functional of
the plastic-strain answer, and call `.backward()`:

```{code-cell} ipython3
alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
strain_base = torch.tensor([0.01, 0.005, -0.001, 0.0, 0.0, 0.0], dtype=torch.float64)
strain_v = SR2(alpha * strain_base)

(ps,) = model(strain_v, guess, plastic_strain_n, t, t_n)
eqv_plastic_strain = torch.sqrt(torch.tensor(2.0 / 3.0)) * ps.data.norm()
eqv_plastic_strain.backward()

float(eqv_plastic_strain), float(alpha.grad)
```

The same backward pass works for parameters (e.g. the yield stress,
the Perzyna reference stress) if you mark them with
`requires_grad=True` — gradient-based training and inverse problems
plug into this hook without changing the forward model.

:::{note}
`ImplicitUpdate` applies the implicit function theorem in backward:
it re-assembles the system Jacobian at the converged state and runs
a single adjoint linear solve, rather than unrolling the Newton
iterations. Because the IFT formula is exact, the returned gradients
agree with finite differences to roughly machine precision regardless
of how many Newton iterations the forward solve took. Picking a good
predictor (the catalog currently offers
`ConstantExtrapolationPredictor` and `LinearExtrapolationPredictor`)
shortens the forward solve without changing the backward result.
:::

## Where to go next

- [](tutorials-models-composition) — `ImplicitUpdate` is just one
  more model. It can sit inside a bigger `ComposedModel`, feed the
  next stage of a pipeline, or be composed with other implicit
  updates.
- [](tutorials-models-transient-driver) — for time-stepping the
  Perzyna update across a load history, `TransientDriver` handles the
  outer loop and reuses the `ImplicitUpdate` at each step.
- [](tutorials-models-vectorization) — the leading batch dimension we
  used above is the same one every NEML2 model accepts.
