(modules-phase-field-fracture)=
# Phase-field fracture

## Overview

The **phase-field fracture** module collects the building blocks needed to
compose regularized brittle-fracture models in NEML2. Sharp cracks are
replaced by a smeared scalar phase field $d \in [0, 1]$ whose evolution
follows a variational principle: the material loses load-bearing capacity
gradually as $d$ grows from $0$ (intact) toward $1$ (fully cracked). The
catalog factors that variational problem into three small, composable
pieces — a **crack geometric function** $\alpha(d)$ that controls how the
field is regularized, a **degradation function** $g(d)$ that controls how
stiffness is lost, and a **strain energy density** $\psi$ that drives the
phase field and supplies the post-fracture stress. Once these three
ingredients are wired up, the standard NEML2 primitives
([](models-Normality), [](models-FBComplementarity),
[](models-ImplicitUpdate), [](models-ComposedModel)) carry out the
implicit update at each material point.

This module is designed to be coupled with other physics — most commonly
the [](modules-solid-mechanics) module to add (visco)plastic dissipation
to the fracture driving force.

## Math

Let $d$ be the phase-field variable, $\alpha(d)$ the crack geometric
function, $g(d)$ the degradation function, and $\psi_\text{active}(\mathbf{E})$
the active part of the elastic strain energy density. Letting
$G_c$, $l$, and $c_0$ denote the fracture energy, regularization length,
and normalization constant of the chosen $\alpha$, the regularized free
energy density per unit volume reads

$$
\psi(d, \mathbf{E}) = \frac{G_c}{c_0\, l}\, \alpha(d) + g(d)\, \psi_\text{active}(\mathbf{E}) + \psi_\text{inactive}(\mathbf{E}).
$$

The Karush–Kuhn–Tucker (KKT) conditions for irreversible phase-field
evolution read

```{math}
\begin{align}
  f &= \eta\, \dot{d} + \frac{\partial \psi}{\partial d} \ge 0, \\
  \dot{d} &\ge 0, \\
  f\, \dot{d} &= 0,
\end{align}
```

with viscous regularization $\eta \ge 0$. Following the standard
phase-field treatment, the inequality complementarity is recast as a
single smooth residual via the Fischer–Burmeister function

$$
f + \dot{d} - \sqrt{f^2 + \dot{d}^2} = 0,
$$

which can be solved for $d$ with a standard Newton iteration. The stress
is obtained from the same potential by differentiation,

$$
\mathbf{S} = \frac{\partial \psi}{\partial \mathbf{E}}.
$$

NEML2 evaluates both derivatives ($\partial\psi/\partial d$ and
$\partial\psi/\partial\mathbf{E}$) symbolically through the
[](models-Normality) operator, so users never write them by hand — only the
energy $\psi$ itself.

### Crack geometric functions

NEML2 ships the two canonical AT-family functionals:

| Type                                        | $\alpha(d)$ | $c_0$  | Behavior                    |
| ------------------------------------------- | ----------- | ------ | --------------------------- |
| [](models-CrackGeometricFunctionAT1)        | $d$         | $8/3$  | Elastic limit before damage |
| [](models-CrackGeometricFunctionAT2)        | $d^2$       | $2$    | Damage from the onset       |

### Degradation functions

The degradation function multiplies the active strain energy to model
loss of stiffness. The catalog provides two parameterized families:

- [](models-PowerDegradationFunction): $g(d) = (1-d)^p (1-\eta) + \eta$
- [](models-RationalDegradationFunction): $g(d) = \dfrac{(1-d)^p}{(1-d)^p + Q(d)}(1-\eta) + \eta$
  with $Q(d) = b_1 d (1 + b_2 d + b_2 b_3 d^2)$

The residual stiffness floor $\eta \ge 0$ keeps the algebraic system
well-conditioned at full damage.

### Strain energy density

The supplied concrete energy is
[](models-LinearIsotropicStrainEnergyDensity), which evaluates the
linear-elastic isotropic strain energy and supports an optional
volumetric–deviatoric (`VOLDEV`) split so that only the tensile/deviatoric
part drives fracture. User-supplied energies / geometric functions /
degradation functions plug into the same wiring as long as they expose
the same input/output variable names as the concrete leaves above.

## Example: linear-elastic brittle fracture

The following input file composes an AT-2, power-degradation, VOLDEV-split
elastic-brittle fracture model and integrates it under uniaxial tension
with the Fischer–Burmeister complementarity solver:

```{literalinclude} ../../../tests/regression/phase_field_fracture/elastic_brittle_fracture/small_deformation.i
:language: ini
```

## Explanation

Reading the `[Models]` block top-to-bottom:

- **`degrade`** ([](models-PowerDegradationFunction)) maps the phase field
  `d` to the degradation factor `g` using the power law with exponent
  $p = 2$.
- **`sed0`** ([](models-LinearIsotropicStrainEnergyDensity)) takes the
  total small strain `E` and splits the strain energy into an `active`
  and `inactive` part using the volumetric–deviatoric decomposition.
  Elastic constants are supplied as Young's modulus and Poisson's ratio.
- **`sed1`** ([](models-ScalarMultiplication)) multiplies the active
  strain energy by the degradation factor:
  $\psi_\text{degraded} = g\, \psi_\text{active}$.
- **`sed`** ([](models-ScalarLinearCombination)) sums the degraded
  active part and the undegraded inactive part to give the total elastic
  strain energy $\psi_e$.
- **`cracked`** ([](models-CrackGeometricFunctionAT2)) maps `d` to
  $\alpha = d^2$.
- **`sum`** combines $\alpha$ and $\psi_e$ into the total regularized
  free energy $\psi$, weighting $\alpha$ by the precomputed scalar
  $G_c / (c_0 l)$ that the input calls `GcbylbyCo`.
- **`energy`** ([](models-ComposedModel)) bundles all six models above
  into a single forward operator $\psi(d, \mathbf{E})$.

The next four models build the KKT residual:

- **`dpsidd`** ([](models-Normality)) symbolically differentiates the
  composed energy with respect to `d`, producing $\partial \psi / \partial d$.
- **`drate`** ([](models-ScalarVariableRate)) produces $\dot{d}$ from the
  current and previous values of `d`.
- **`functional`** sums the two into the KKT functional $f$
  (with $\eta = 1$ here, controlling the viscous regularization).
- **`Fish_Burm`** ([](models-FBComplementarity)) wraps $f$ and $\dot{d}$
  into the smoothed Fischer–Burmeister residual `d_residual`.
- **`eq`** ([](models-ComposedModel)) bundles the four into a single
  residual model `eq` that is fed to the nonlinear system.

The `[EquationSystems]` and `[Solvers]` blocks declare a `NonlinearSystem`
with `d` as the unknown and a standard [](solvers-Newton) solver with a
dense LU linear backend.

The second `[Models]` block then ties everything together:

- **`predictor`** ([](models-LinearExtrapolationPredictor)) supplies a
  linear-extrapolation initial guess for the Newton solve at each step.
- **`solve_d`** ([](models-ImplicitUpdate)) wraps the equation system
  into a forward operator that solves for `d`.
- **`stress`** ([](models-Normality)) reuses the same energy potential
  to compute the stress $\mathbf{S} = \partial \psi / \partial \mathbf{E}$
  — automatically picking up the damaged elastic moduli through $g(d)$.
- **`model`** is the top-level [](models-ComposedModel) exposing
  `solve_d` and `stress` so that the transient driver integrates the
  phase field and reports the degraded stress over the loading history.

:::{note}
The example uses the symbol `d` for the phase-field variable rather than
$\phi$ to keep the input file compact. The catalog types use the
variable name `phase` on input.
:::

## See also

- [](tutorials-models-composition) — general patterns for composing
  NEML2 models with `ComposedModel`.
- [](tutorials-models-implicit-model) — background on `ImplicitUpdate`
  and Newton-solved residual models.
- [](modules-solid-mechanics) — the elasticity and plasticity catalog
  that supplies the stress driver coupled to phase-field damage in
  multi-physics setups.
- [](syntax-catalog) — per-type option reference for every model
  registered in this module.
