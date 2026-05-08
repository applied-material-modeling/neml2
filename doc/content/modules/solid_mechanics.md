# Solid Mechanics {#solid-mechanics}

[TOC]

The solid mechanics physics module is a collection of objects serving as building blocks for composing material models for solids. Each category of the material model is explained below, with both the mathematical formulations and example input files.

## Elasticity

Elasticity models describe the relationship between stress \f$ \boldsymbol{\sigma} \f$ and strain \f$ \boldsymbol{\varepsilon} \f$ without any history-dependent (internal) state variables. In general, the stress-strain relation can be written as

\f[
  \boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon}
\f]

where \f$ \mathbb{C} \f$ is the fourth-order elasticity tensor. For linear isotropic elasticity, this relation can be simplified as

\f[
  \boldsymbol{\sigma} = 3 K \operatorname{vol} \boldsymbol{\varepsilon} + 2 G \operatorname{dev} \boldsymbol{\varepsilon}
\f]

where \f$ K \f$ is the bulk modulus, and \f$ G \f$ is the shear modulus.

Below is an example input file that defines a linear elasticity model.

@list-input:tests/unit/models/solid_mechanics/elasticity/LinearIsotropicElasticity.i:Models

## Viscoelasticity

Viscoelastic models describe time-dependent stress-strain behavior in which the response is partially elastic (recoverable) and partially viscous (rate-dependent). Unlike elasticity, viscoelastic models retain a memory of past loading through internal strain variables, but unlike plasticity the response is linear in the constitutive sense — there is no yield surface and no consistency parameter.

NEML2 builds viscoelastic models from two primitive rheological elements — a linear *spring* with modulus \f$ E \f$, and a Newtonian *dashpot* with viscosity \f$ \eta \f$ — combined in series and parallel. Each model treats the spring and dashpot relations component-wise on \f$ \boldsymbol{\sigma} \f$ and \f$ \boldsymbol{\varepsilon} \f$, so the same parameters describe both the deviatoric and volumetric response (a common simplification when only one effective modulus is available from experiment). Internal strain variables evolve as rates and are time-integrated implicitly with `SR2BackwardEulerTimeIntegration` inside an `ImplicitUpdate` Newton solve.

Standard textbook configurations are pre-assembled as named element models for convenience: `KelvinVoigtElement` (spring + dashpot in parallel), `ZenerElement` (Standard Linear Solid), `WiechertElement` (generalized Maxwell), and `BurgersElement` (Maxwell in series with Kelvin-Voigt). The example input file below uses the Zener element directly — the constitutive object provides both the stress and the viscous-strain rate; `SR2BackwardEulerTimeIntegration` turns the rate into an implicit residual; and the resulting nonlinear system is solved at each time step by a Newton solver.

@list-input:tests/regression/solid_mechanics/viscoelasticity/zener/model.i:Models,EquationSystems,Solvers

For topologies that don't match a pre-assembled element — for example, a 5-element generalized Maxwell or a custom series-parallel network — assemble the network from primitives directly: `LinearDashpot` for each dashpot leaf (mapping stress to viscous strain rate via \f$ \dot{\boldsymbol{\varepsilon}} = \boldsymbol{\sigma}/\eta \f$), `LinearIsotropicElasticity` (or a scalar-modulus `SR2LinearCombination`) for each spring, `SR2LinearCombination` for the strain decompositions and stress sums that encode series and parallel topology, and one `ComposedModel` to glue it all together. The wiring is purely by variable naming — series-chain elements share a stress variable, parallel branches share a strain variable, and each dashpot's strain is one unknown in the `NonlinearSystem`. Worked examples for every named topology, plus a 5-element generalized Maxwell, live under `tests/regression/solid_mechanics/viscoelasticity/composed_*/`. The Burgers composition (Maxwell branch in series with a Kelvin-Voigt block, two unknowns sharing the chain stress) is the most instructive of the set:

@list-input:tests/regression/solid_mechanics/viscoelasticity/composed_burgers/model.i:Models,EquationSystems

Refer to [Syntax Documentation](@ref syntax-models) for the complete catalog of viscoelastic objects, including each one's parameters, inputs, and outputs.

## Plasticity (macroscale)

Generally speaking, plasticity models describe (oftentimes irreversible and dissipative) history-dependent deformation of solid materials. The plastic deformation is governed by the plastic loading/unloading conditions (or more generally the Karush-Kuhn-Tucker conditions):

\f{align*}
  f^p \leq 0, \quad \dot{\gamma} \geq 0, \quad \dot{\gamma}f^p = 0, \\
\f}

where \f$ f^p \f$ is the yield function, and \f$ \gamma \f$ is the consistency parameter.

### Consistent plasticity

Consistent plasticity refers to the family of (macroscale) plasticity models that solve the plastic loading/unloading conditions (or the KKT conditions) exactly (up to machine precision).

> Consistent plasticity models are sometimes considered rate-independent. But that is a misnomer as rate sensitivity can be baked into the yield function definition in terms of the rates of the internal variables.

Residual associated with the KKT conditions can be written as the Fischer-Burmeister complementarity condition

\f{align*}
  r = \dot{\gamma} - f^p - \sqrt{{\dot{\gamma}}^2 + {f^p}^2}.
\f}

This complementarity condition is implemented by `FBComplementarity` with `a_inequality = 'LE'`. A complete example input file for consistent plasticity is shown below, and the composition and possible modifications are explained in the following subsections.

@list-input:tests/regression/solid_mechanics/rate_independent_plasticity/perfect/model.i:Models,EquationSystems,Solvers

### Viscoplasticity

Viscoplasticity models regularize the KKT conditions by introducing approximations to the constraints. A widely adopted approximation is the Perzyna model where rate sensitivity is baked into the approximation following a power-law relation:

\f{align*}
  \dot{\gamma} = \left( \dfrac{\left< f^p \right>}{\eta} \right)^n,
\f}

where \f$ \eta \f$ is the reference stress and \f$ n \f$ is the power-law exponent.

The Perzyna model is implemented by the object `PerzynaPlasticFlowRate`. A complete example input file for viscoplasticity is shown below, and the composition and possible modifications are explained in the following subsections.

@list-input:tests/regression/solid_mechanics/viscoplasticity/perfect/model.i:Models,EquationSystems,Solvers

### Effective stress

The effective stress is a measure of stress describing how the plastic deformation "flows". For example, the widely-used \f$ J_2 \f$ plasticity uses the von Mises stress as the stress measure, i.e.,

\f{align*}
  \bar{\sigma} &= \sqrt{3 J_2}, \\
  J_2 &= \frac{1}{2} \operatorname{dev} \boldsymbol{\sigma} : \operatorname{dev} \boldsymbol{\sigma}.
\f}

Commonly used stress measures are defined using `SR2Invariant`.

@list-input:tests/unit/models/common/SR2Invariant_VONMISES.i:Models

### Perfectly Plastic Yield function

For perfectly plastic materials, the yield function only depends on the effective stress and a constant yield stress, i.e., the envelope does not shrink or expand depending on the loading history.

\f{align*}
  f^p &= \bar{\sigma} - \sigma_y.
\f}

Below is an example input file defining a perfectly plastic yield function with \f$ J_2 \f$ flow.

@list-input:tests/regression/solid_mechanics/rate_independent_plasticity/perfect/model.i:Models/vonmises,Models/yield

### Isotropic hardening

The equivalent plastic strain \f$ \bar{\varepsilon}^p \f$ is a scalar-valued internal variable that can be introduced to control the shape of the yield function. The isotropic strain hardening \f$ k \f$ is controlled by the accumulated equivalent plastic strain, and enters the yield function as

\f{align*}
  f^p &= \bar{\sigma} - \sigma_y - k(\bar{\varepsilon}^p).
\f}

Below is an example input file defining a yield function with \f$ J_2 \f$ flow and linear isotropic hardening.

```python
[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'S'
    invariant = 'effective_stress'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
[]
```

### Kinematic hardening

The kinematic plastic strain \f$ \boldsymbol{K}^p \f$ is a tensor-valued internal variable that can be introduced to control the shape of the yield function. The kinematic hardening \f$ X \f$ is controlled by the accumulated kinematic plastic strain, and the effective stress is defined in terms of the over stress. In case of \f$ J_2 \f$, the effective stress can be rewritten as

\f{align*}
  \bar{\sigma} &= \sqrt{3 J_2}, \\
  J_2 &= \frac{1}{2} \operatorname{dev} \boldsymbol{\Xi} : \operatorname{dev} \boldsymbol{\Xi}, \\
  \boldsymbol{\Xi} &= \boldsymbol{\sigma} - \boldsymbol{X}.
\f}

Below is an example input file defining a yield function with \f$ J_2 \f$ flow and linear kinematic hardening.

```python
[Models]
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
  [overstress]
    type = SR2LinearCombination
    from = 'S X'
    to = 'O'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
  []
[]
```

### Back stress

An alternative way of introducing hardening is through back stresses. Instead of modeling the accumulation of kinematic plastic strain, back stress models directly describe the evolution of back stress. An example input file defining a yield function with \f$ J_2 \f$ flow and two back stresses is shown below.

```python
[Models]
  [overstress]
    type = SR2LinearCombination
    from = 'S X1 X2'
    to = 'O'
    weights = '1 -1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
  []
[]
```

### Mixed hardening

Isotropic hardening, kinematic hardening, and back stresses are all optional and can be "mixed" in the definition of a yield function. The example input file below shows a yield function with \f$ J_2 \f$ flow, isotropic hardening, kinematic hardening, and two back stresses.

```python
[Models]
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [kinharden]
    type = LinearKinematicHardening
    hardening_modulus = 1000
    back_stress = 'X0'
  []
  [overstress]
    type = SR2LinearCombination
    from = 'S X0 X1 X2'
    to = 'O'
    weights = '1 -1 -1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 'effective_stress'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'isotropic_hardening'
  []
[]
```

### Flow rules

Flow rules are required to map from the consistency parameter \f$ \gamma \f$ to various internal variables describing the state of hardening, such as the equivalent plastic strain \f$ \bar{\varepsilon}^p \f$, the kinematic plastic strain \f$ \boldsymbol{K}^p \f$, and the back stress \f$ \boldsymbol{X} \f$.

Associative flow rules define flow directions variationally according to the principle of maximum dissipation, i.e.,

\f{align*}
  \dot{\boldsymbol{\varepsilon}}^p &= \dot{\gamma} \dfrac{\partial f^p}{\partial \boldsymbol{\sigma}}, \\
  \dot{\bar{\varepsilon}}^p &= -\dot{\gamma} \dfrac{\partial f^p}{\partial k}, \\
  \dot{\boldsymbol{K}}^p &= \dot{\gamma} \dfrac{\partial f^p}{\partial \boldsymbol{X}}.
\f}

The example input file below defines associative \f$ J_2 \f$ flow rules

```python
  [flow]
    ...
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'yield_function'
    from = 'mandel_stress isotropic_hardening X'
    to = 'flow_direction isotropic_hardening_direction kinematic_hardening_direction'
  []
  [eprate]
    type = AssociativeIsotropicPlasticHardening
  []
  [Kprate]
    type = AssociativeKinematicPlasticHardening
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
[]
```

In the above example, a model named "normality" is used to compute the associative flow directions, and the rates of the internal variables are mapped using the rate of the consistency parameter and each of the associative flow direction. The cross-referenced model named "flow" (omitted in the example) is the composition of models defining the yield function \f$ f^p \f$ in terms of the variational arguments \f$ \boldsymbol{\sigma} \f$, \f$ k \f$, and \f$ \boldsymbol{X} \f$.


## Mixed stress/strain control

NEML2 models can be setup to take as input strain and provide stress or take as input stress and provide strain.  With some additional work, a model that "natively" works in either strain or stress control can be modified to work under mixed stress/strain control.

Mixed control here means that some components of the strain tensor and some components of the stress tensor are provided as input and the model must solve for the conjugate, missing components of stress or strain.

Mathematically, at each time step consider a tensor of applied, mixed strain or stress conditions \f$ f_{ij}  \f$ and a control signal \f$ c_{ij} \f$.  When \f$ c_{ij} < h \f$ for some threshold \f$ h \f$ we consider the corresponding component of the input \f$ f_{ij} \f$ to be a strain value and the model must solve for the corresponding value of stress.  If \f$ c_{ij} \ge h \f$ we consider the corresponding component of the input \f$ f_{ij} \f$ to be a stress value and the model must solve for the corresponding value of strain.

Modifying a model for mixed control only requires one additional object, `MixedControlSetup`, which maps the mixed input and conjugate state vector to the model's stress and strain variables:

```python
  [mixed]
    type = MixedControlSetup
    x_above = 'fixed_values'
    x_below = 'mixed_state'
    y = 'stress'
    z = 'strain'
  []
```

Here `fixed_values` is the prescribed mixed input (force), `mixed_state` is the unknown conjugate variable that the solver determines, `y` is the stress variable name, and `z` is the strain variable name. The components of `mixed_state` that are active (i.e., the unknowns being solved) are determined by the `control` signal tensor provided to the driver.

The `mixed_state` unknown must be listed in the `NonlinearSystem`, with its residual mapped to the stress residual:

```python
[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'mixed_state ...'
    residuals = 'stress_residual ...'
  []
[]
```

## Crystal plasticity

NEML2 adopts an incremental rate-form view of crystal plasticity.  The fundemental kinematics derive from the rate expansion of the elastic-plastic multiplactive split:

\f{align*}
  F = F^e F^p
\f}

where the spatial velocity gradient is then

\f{align*}
  l = \dot{F} F^{-1} = \dot{F}^e F^{e-1} + F^e \dot{F}^{p} {F}^{p-1} F^{e-1}
\f}

The plastic deformation \f$ \bar{l}^p = \dot{F}^{p} {F}^{p-1} \f$ defines the crystal plasticity kinemtics and NEML2 assumes that the elastic stretch is small (\f$ F^e = \left(I + \varepsilon \right) R^e \f$) so that spatial velocity gradient becomes

\f{align*}
  l =  \dot{\varepsilon} +  \Omega^e - \Omega^e \varepsilon + \varepsilon \Omega^e + l^p + \varepsilon l^p -  l^p \varepsilon
\f}

defining \f$ l^p = R^e \bar{l}^p R^{eT} \f$ as the constitutive plastic velocity gradient rotated into the current configuration and \f$ \Omega^e = \dot{R}^e R^{eT} \f$ as the elastic spin and assuming that

1. Terms quadratic in the elastic stretch (\f$ \varepsilon\f$) are small.
2. Terms quadratic in the rate of elastic stretch (\f$ \dot{\varepsilon} \f$) are also small.

The first assumption is accurate for metal plasticity, the second assumption is more questionable if the material deforms at a fast strain rate.

Define the current orientation of a crystal as the composition of its initial rotation from the crystal system to the lab frame and the elastic rotation, i.e.

\f{align*}
  Q = R^e Q_0
\f}

and note with this defintion we can rewrite the spin equation:

\f{align*}
  \Omega^e = \dot{R}^e R^{eT} = \dot{Q} Q_0^T Q_0 Q^T = \dot{Q} Q^T
\f}

With this definition and the choice of kinematics above we can derive evolution equations for our fundemental constitutive quantities, the elastic stretch \f$ \varepsilon \f$ and the orientation $Q$ by splitting the spatial velocity gradient into symmetric and skew parts and rearranging the resulting equations:

\f{align*}
  \dot{\varepsilon}= d -d^p-\varepsilon w +  w \varepsilon \\
  \dot{Q} = \left(w - w^p - \varepsilon d^p + d^p \varepsilon\right) Q .
\f}

where \f$l = d + w\f$ and \f$l^p = d^p + w^p\f$

For most (or all) choices of crystal plasticity constitutive models we also need to define the Cauchy stress as:
\f{align*}
  \sigma = C : \varepsilon
\f}

with \f$ C \f$ a (generally) anisotropic crystal elasticity tensor rotated into the current configuration.

The crystal plasticity examples in NEML2 integrate the elastic strain (and the constitutive internal variables) using a backward Euler integration rule and integrate the crystal orientation using either an implicit or explicit exponential rule.  These integrate rules can either be coupled or decoupled, i.e. integrated together in a fully implicit manner or first integrate the strain and internal variables and then sequentially integrating the rotations.

A full constitutive model must then define the plastic deformation $l^p$ and whatever internal variables are used in this definition. A wide variety of choices are possible, but the examples use the basic assumption of Asaro:

\f{align*}
  l^p = \sum_{i=1}^{n_{slip}} \dot{\gamma}_i Q \left(d_i \otimes n_i \right) Q^T
\f}

where now \f$ \dot{\gamma}_i \f$, the slip rate on each system, is the constitutive chioce.  NEML provides a variety of options for defining these slip rates in terms of internal hardening variables and the results shear stress

\f{align*}
  \tau_i = \sigma : Q \operatorname{sym}\left(d_i \otimes n_i \right) Q^T
\f}

Ancillary classes automatically generate lists of slip and twin systems from the crystal sytem, so the user does not need to manually provide these themselves.

NEML2 uses *modified* Rodrigues parameters to define orientations internally.  These can be converted to Euler angles, quaternions, etc. for output.
