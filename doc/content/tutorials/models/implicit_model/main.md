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

A constitutive model often updates its internal state by solving a
**nonlinear algebraic system** rather than evaluating a closed-form
expression. Stiff viscoplastic flow is the canonical case: the
plastic-strain increment over a time step depends on the stress
*after* the increment, which in turn depends on the plastic strain, so
the update equation only has a closed form in special cases. NEML2
expresses this pattern as

$$
  \mathbf{r}(\tilde{\mathbf{s}}; \mathbf{g}, \mathbf{p}) = \mathbf{0},
  \qquad
  \mathbf{s} = \operatorname*{root}_{\tilde{\mathbf{s}}} \mathbf{r}. \notag
$$

Here $\mathbf{r}$ is a residual that vanishes at the converged state
$\mathbf{s}$, $\mathbf{g}$ is whatever is given (e.g. the new total
strain and time), and $\mathbf{p}$ collects the parameters.

The model that writes the residual is just another `Model` — usually a
`ComposedModel` glued together from a stack of small ingredients. To
turn it into an updater that *solves* for $\mathbf{s}$, wrap it in an
`ImplicitUpdate`. The wrapper

1. Declares $\mathbf{s}$ as an output instead of an input.
2. Validates that the residual and the unknowns have the same shape
   (the system is square).
3. Invokes a vectorized Newton solver to drive $\mathbf{r}$ to zero
   over the whole batch in parallel.
4. Applies the **implicit function theorem** so that the wrapped
   model is end-to-end differentiable through the solve, to machine
   precision.

This tutorial walks through that pattern on a Perzyna-style
viscoplasticity update.

## The physics

The Perzyna viscoplastic update we will solve is, in continuous form,

$$
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
$$

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

Each line in the system above maps onto a registered model in NEML2's
catalog, and the residual is the `ComposedModel` that wires them
together. The full input file:

```{literalinclude} input.i
:language: ini
:caption: input.i
```

The first block we care about is `[system]` — it is the
`ComposedModel` that *computes the residual*. Loading it gives a
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

Note the input list: a candidate `plastic_strain` (the unknown), the
prescribed `strain`, the previous-step `plastic_strain~1`, and the
current and previous times `t`, `t~1`. Evaluating `system` just plugs
all of those into the algebra above and returns the residual at that
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
Finding the root is what `ImplicitUpdate` exists for.

## Wrapping the residual in an `ImplicitUpdate`

Three pieces are needed in addition to the residual model:

1. A **nonlinear system** description that names the unknowns.
   `[EquationSystems]` does this — `type = NonlinearSystem`,
   `model = 'system'`, `unknowns = 'plastic_strain'`.
2. A **nonlinear solver**. NEML2 ships two fully vectorized choices:
   [`Newton`](solvers-Newton) (vanilla full-step Newton-Raphson) and
   [`NewtonWithLineSearch`](solvers-NewtonWithLineSearch) (the same,
   with several common line-search strategies). Both reference a
   **linear solver** for the inner system Jacobian; here we use
   [`DenseLU`](solvers-DenseLU).
3. The wrapper itself: `[model]` with `type = ImplicitUpdate`,
   pointing at the equation system and the solver.

Re-reading the same input file but asking for `model` returns the
wrapped object:

```{code-cell} ipython3
model = neml2.load_model("input.i", "model")
model
```

The input signature is identical to the residual model's input
signature — same `strain`, `plastic_strain`, `t`, … — but
`plastic_strain` is now declared as an *output*: it's the unknown the
solver produces, not something the caller supplies. The slot is still
present on the call so the solver can be seeded with an initial
guess.

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

`ImplicitUpdate` is batched the same way every other NEML2 model is.
Stack `N` prescribed strains into a leading batch dimension, hand the
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
t_b       = Scalar(torch.ones(N, dtype=torch.float64))
tn_b      = Scalar(torch.zeros(N, dtype=torch.float64))

(plastic_strain_b,) = model(strain_b, guess_b, psn_b, t_b, tn_b)
plastic_strain_b.data
```

There is no Python loop — that is the entire point of
[](tutorials-models-vectorization), and the implicit solver respects
it.

## Differentiating through the solve

This is the property that makes `ImplicitUpdate` more than a thin
shim around a Newton solver. Once $\mathbf{r}(\mathbf{s}; \mathbf{g}) = 0$,
the implicit function theorem gives

$$
  \frac{\partial \mathbf{s}}{\partial \mathbf{g}}
  = -\left(\frac{\partial \mathbf{r}}{\partial \mathbf{s}}\right)^{-1}
     \frac{\partial \mathbf{r}}{\partial \mathbf{g}}.
$$

The wrapped model reuses the converged Jacobian factorization to
apply this formula in its backward pass, so any quantity downstream
of the solved state is differentiable end-to-end without unrolling
the Newton iterations. Concretely, we can drive the prescribed strain
with a scalar load multiplier `alpha`, push it through the solver,
read off a scalar functional of the plastic-strain answer, and call
`.backward()`:

```{code-cell} ipython3
alpha = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
strain_base = torch.tensor([0.01, 0.005, -0.001, 0.0, 0.0, 0.0], dtype=torch.float64)
strain_v = SR2(alpha * strain_base)

(ps,) = model(strain_v, guess, plastic_strain_n, t, t_n)
eqv_plastic_strain = torch.sqrt(torch.tensor(2.0 / 3.0)) * ps.data.norm()
eqv_plastic_strain.backward()

float(eqv_plastic_strain), float(alpha.grad)
```

The same backward pass would work for parameters (e.g. the yield
stress, the Perzyna reference stress) if any of them were declared
with `requires_grad=True` — gradient-based training and inverse
problems sit on top of this hook without any change to the forward
model.

:::{note}
`ImplicitUpdate` reuses the LU factorization of the system Jacobian
at the converged state to apply the implicit function theorem. The
backward pass is therefore $O(N_s^2)$ per item (one back-solve), not
$O(N_s^3)$ (a full re-factor) — which matters once $N_s$ grows or the
batch size does. Because the implicit-function formula is exact, the
returned gradients agree with finite differences to roughly machine
precision, regardless of how many Newton iterations the forward solve
took. Picking a good predictor (e.g. an elastic-trial state for
return mapping) shortens the forward solve without changing the
backward result.
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
