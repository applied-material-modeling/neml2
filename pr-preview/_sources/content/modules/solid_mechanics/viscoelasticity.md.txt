(modules-solid-mechanics-viscoelasticity)=
# Viscoelasticity

## Overview

Viscoelastic models describe time-dependent stress-strain behavior
that is partly elastic (recoverable) and partly viscous
(rate-dependent). A viscoelastic material carries memory of past
loading through internal strain variables, but the constitutive
response stays linear — there is no yield surface and no
consistency parameter.

NEML2 builds viscoelastic models from two primitive rheological elements — a
linear *spring* with modulus $E$ and a Newtonian *dashpot* with viscosity
$\eta$ — combined in series and parallel. Standard textbook configurations are
shipped as pre-assembled element models for convenience
([](models-KelvinVoigtElement), [](models-ZenerElement),
[](models-WiechertElement), [](models-BurgersElement)), while
[](models-LinearDashpot) is exposed as a leaf for assembling arbitrary
networks from primitives. Each model treats the spring and dashpot relations
component-wise on $\boldsymbol{\sigma}$ and $\boldsymbol{\varepsilon}$, so the
same parameters describe both the deviatoric and volumetric response — a common
simplification when only one effective modulus is available from experiment.

## Governing equations

The two primitives are

$$
\boldsymbol{\sigma} = E \boldsymbol{\varepsilon}
\quad\text{(spring)},
\qquad
\boldsymbol{\sigma} = \eta \dot{\boldsymbol{\varepsilon}}
\quad\text{(dashpot)}.
$$

A series connection adds strains under a shared stress; a parallel connection
adds stresses under a shared strain. The four named topologies in the catalog
follow from these two rules. Taking
$\boldsymbol{\varepsilon}_v$ as the dashpot (viscous) strain in each branch:

For [](models-KelvinVoigtElement) — spring and dashpot in parallel — the total
stress is the sum of the two branches and there is no separate viscous-strain
unknown:

$$
\boldsymbol{\sigma} = E \boldsymbol{\varepsilon} + \eta \dot{\boldsymbol{\varepsilon}}.
$$

For [](models-ZenerElement) — equilibrium spring in parallel with a Maxwell
branch (Standard Linear Solid):

\begin{align}
\boldsymbol{\sigma} &= (E_\infty + E_M) \boldsymbol{\varepsilon} - E_M \boldsymbol{\varepsilon}_v, \\
\dot{\boldsymbol{\varepsilon}}_v &= \frac{E_M}{\eta_M} (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_v).
\end{align}

For [](models-WiechertElement) — an equilibrium spring in parallel with $N$
Maxwell branches (generalized Maxwell), one viscous strain
$\boldsymbol{\varepsilon}_v^{(i)}$ per branch:

\begin{align}
\boldsymbol{\sigma} &= E_\infty \boldsymbol{\varepsilon} + \sum_{i=1}^N E_M^{(i)} \left( \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_v^{(i)} \right), \\
\dot{\boldsymbol{\varepsilon}}_v^{(i)} &= \frac{E_M^{(i)}}{\eta_M^{(i)}} \left( \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_v^{(i)} \right), \quad i = 1, \dots, N.
\end{align}

For [](models-BurgersElement) — a Maxwell branch in series with a Kelvin-Voigt
block, with $\boldsymbol{\varepsilon}_v^M$ the Maxwell dashpot strain and
$\boldsymbol{\varepsilon}^K$ the Kelvin-Voigt block strain:

\begin{align}
\boldsymbol{\sigma} &= E_M \left( \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_v^M - \boldsymbol{\varepsilon}^K \right), \\
\dot{\boldsymbol{\varepsilon}}_v^M &= \boldsymbol{\sigma} / \eta_M, \\
\dot{\boldsymbol{\varepsilon}}^K &= \left( \boldsymbol{\sigma} - E_K \boldsymbol{\varepsilon}^K \right) / \eta_K.
\end{align}

In every case the constitutive model exposes the rate
$\dot{\boldsymbol{\varepsilon}}_v$ of each internal strain variable as a
derived output. [](models-SR2BackwardEulerTimeIntegration) turns each rate
into an implicit residual, and an [](models-ImplicitUpdate) wrapping a
[](solvers-Newton) solver advances the unknowns one time step at a time.

## Example: Zener element (Standard Linear Solid)

The simplest non-trivial case is the Zener element. The single viscous strain
$\boldsymbol{\varepsilon}_v$ is the only Newton unknown, and the canned
[](models-ZenerElement) closure provides both the stress and the
viscous-strain rate.

```{literalinclude} ../../../../tests/regression/solid_mechanics/viscoelasticity/zener/model.i
:language: ini
```

Walking through the wiring:

1. **`[zener]`** is the constitutive closure. It consumes `strain` (the
   forcing) and `viscous_strain` (the unknown) and produces `stress` and
   `viscous_strain_rate`. The three parameters — `equilibrium_modulus`
   ($E_\infty$), `maxwell_modulus` ($E_M$), and `maxwell_viscosity`
   ($\eta_M$) — are the only material constants in the model.
2. **`[integrate_Ev]`** is a [](models-SR2BackwardEulerTimeIntegration)
   instance. It consumes `viscous_strain` (current iterate),
   `viscous_strain_rate` (from `zener`), and the old-state pair, and emits a
   residual `viscous_strain_residual` that is zero when the backward-Euler
   identity $\boldsymbol{\varepsilon}_v - \boldsymbol{\varepsilon}_v^{n-1} =
   \Delta t \dot{\boldsymbol{\varepsilon}}_v$ holds.
3. **`[implicit_rate]`** is a [](models-ComposedModel) that glues `zener` and
   `integrate_Ev` together by variable name — there is no explicit edge list,
   just shared input/output names.
4. **`[eq_sys]`** declares the `NonlinearSystem`: the unknown is
   `viscous_strain`, the residual is `viscous_strain_residual`, and the
   model that computes the residual is `implicit_rate`.
5. **`[update]`** wraps the system in an [](models-ImplicitUpdate). On each
   forward pass, the [](models-ConstantExtrapolationPredictor) seeds
   `viscous_strain` from the previous step, the [](solvers-Newton) solver
   drives the residual to zero, and the converged `viscous_strain` is fed
   forward.
6. **`[model]`** composes the implicit update with a fresh evaluation of
   `zener` so that the post-solve `stress` is computed from the converged
   internal state. The `additional_outputs = 'viscous_strain'` line keeps the
   internal variable visible to the driver for time integration of the next
   step.

Swapping in any other pre-assembled element is a one-line change to the
`[zener]` block — the rest of the wiring (time integration, equation system,
implicit update) is identical for any single-unknown viscoelastic model. The
[](models-WiechertElement) and [](models-BurgersElement) classes extend this
pattern to multi-branch generalized-Maxwell and Burgers networks without
requiring you to assemble springs and dashpots yourself.

## Composing custom networks

When a topology doesn't match a named element — for example a 5-element
generalized Maxwell or a Burgers-plus-extra-dashpot — the network can be
assembled from primitives directly. The recipe is purely topological:

- Use [](models-LinearDashpot) for each dashpot leaf. It maps
  $\boldsymbol{\sigma} \mapsto \dot{\boldsymbol{\varepsilon}}_v =
  \boldsymbol{\sigma} / \eta$.
- Use [](models-SR2LinearCombination) (or a full
  [](models-LinearIsotropicElasticity)) as the constitutive law for each
  spring, and again for the strain decompositions and stress sums that
  encode series/parallel topology.
- Wire it all up with [](models-ComposedModel). The graph is by *variable
  name*: series-chain elements share a stress variable, parallel branches
  share a strain variable, and each dashpot's strain is one unknown in the
  `NonlinearSystem`.

Worked examples for every named topology, plus a 5-element generalized
Maxwell, live under
`tests/regression/solid_mechanics/viscoelasticity/composed_*/`. The Burgers
composition is the most instructive — two viscous strains share the chain
stress, so it exercises both series and parallel assembly in one input file:

```{literalinclude} ../../../../tests/regression/solid_mechanics/viscoelasticity/composed_burgers/model.i
:language: ini
```

The unknowns are now `maxwell_viscous_strain` and `kelvin_voigt_strain` — two
[](models-SR2BackwardEulerTimeIntegration) instances, two residuals, one
Newton solve. The hand-composed input is mathematically identical to
`tests/regression/.../burgers/model.i` (which uses
[](models-BurgersElement) directly) and produces the same gold
result — the difference is that the composed form lets you change
the topology by editing the input file, without authoring a new
Model class.

## See also

- [](tutorials-models-composition) — the general
  [](models-ComposedModel)/`ComposedModel` mechanics that the
  custom-network recipe above relies on.
- [](tutorials-models-implicit-model) — a step-by-step walk-through of an
  [](models-ImplicitUpdate)/Newton solve, with the same general shape as the
  Zener example above.
- [](tutorials-models-transient-driver) — how the `TransientDriver` advances
  any model (including these) over a load history.
- [](modules-solid-mechanics-elasticity) — the elastic primitives
  ([](models-LinearIsotropicElasticity), [](models-SR2LinearCombination))
  reused inside hand-assembled viscoelastic networks.
- [](modules-solid-mechanics-plasticity) — the next step up in complexity:
  add a yield surface and a consistency parameter on top of a viscoelastic
  backbone.
- [syntax catalog](syntax-catalog) — auto-generated, per-type option lists
  for every viscoelastic model.
