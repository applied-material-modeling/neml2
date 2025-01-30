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

```python
[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
```

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

This complementarity condition is implemented by the object `RateIndependentPlasticFlowConstraint`. A complete example input file for consistent plasticity is shown below, and the composition and possible modifications are explained in the following subsections.

```
[Models]
  [elastic_strain]
    type = ElasticStrain
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/S'
    invariant = 'state/internal/s'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 1000
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_function'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/S'
    to = 'state/internal/NM'
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [consistency]
    type = RateIndependentPlasticFlowConstraint
  []
  [surface]
    type = ComposedModel
    models = "elastic_strain elasticity
              vonmises yield_function normality Eprate
              consistency integrate_Ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'surface'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep'
  []
[]
```

### Viscoplasticity

Viscoplasticity models regularize the KKT conditions by introducing approximations to the constraints. A widely adopted approximation is the Perzyna model where rate sensitivity is baked into the approximation following a power-law relation:

\f{align*}
  \dot{\gamma} = \left( \dfrac{\left< f^p \right>}{\eta} \right)^n,
\f}

where \f$ \eta \f$ is the reference stress and \f$ n \f$ is the power-law exponent.

The Perzyna model is implemented by the object `PerzynaPlasticFlowRate`. A complete example input file for viscoplasticity is shown below, and the composition and possible modifications are explained in the following subsections.

```
[Models]
  [elastic_strain]
    type = SR2LinearCombination
    from_var = 'forces/E state/internal/Ep'
    to_var = 'state/internal/Ee'
    coefficients = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/S'
    invariant = 'state/internal/s'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
  []
  [flow]
    type = ComposedModel
    models = 'vonmises yield_function'
  []
  [normality]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/S'
    to = 'state/internal/NM'
  []
  [flow_rate]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
  []
  [Eprate]
    type = AssociativePlasticFlow
  []
  [integrate_Ep]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/internal/Ep'
  []
  [implicit_rate]
    type = ComposedModel
    models = "isoharden elastic_strain elasticity
              vonmises yield_function flow_rate
              normality Eprate integrate_Ep"
  []
  [return_map]
    type = ImplicitUpdate
    implicit_model = 'implicit_rate'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'return_map elastic_strain elasticity'
    additional_outputs = 'state/internal/Ep'
  []
[]
```

### Effective stress

The effective stress is a measure of stress describing how the plastic deformation "flows". For example, the widely-used \f$ J_2 \f$ plasticity uses the von Mises stress as the stress measure, i.e.,

\f{align*}
  \bar{\sigma} &= \sqrt{3 J_2}, \\
  J_2 &= \frac{1}{2} \operatorname{dev} \boldsymbol{\sigma} : \operatorname{dev} \boldsymbol{\sigma}.
\f}

Commonly used stress measures are defined using `SR2Invariant`.

```python
[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/S'
    invariant = 'state/internal/s'
  []
[]
```

### Perfectly Plastic Yield function

For perfectly plastic materials, the yield function only depends on the effective stress and a constant yield stress, i.e., the envelope does not shrink or expand depending on the loading history.

\f{align*}
  f^p &= \bar{\sigma} - \sigma_y.
\f}

Below is an example input file defining a perfectly plastic yield function with \f$ J_2 \f$ flow.

```python
[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/S'
    invariant = 'state/internal/s'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
  []
[]
```

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
    tensor = 'state/internal/S'
    invariant = 'state/internal/s'
  []
  [isoharden]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
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
    from_var = 'state/internal/S state/internal/X'
    to_var = 'state/internal/O'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
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
    from_var = 'state/internal/S state/internal/X1 state/internal/X2'
    to_var = 'state/internal/O'
    coefficients = '1 -1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
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
    back_stress = 'state/internal/X0'
  []
  [overstress]
    type = SR2LinearCombination
    from_var = 'state/internal/S state/internal/X0 state/internal/X1 state/internal/X2'
    to_var = 'state/internal/O'
    coefficients = '1 -1 -1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield_function]
    type = YieldFunction
    yield_stress = 5
    isotropic_hardening = 'state/internal/k'
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
    function = 'state/internal/fp'
    from = 'state/internal/S state/internal/k state/internal/X'
    to = 'state/internal/NM state/internal/Nk state/internal/NX'
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

Modifying a model for mixed control only requires a few additional objects.  The first maps from the mixed input and a conjugate mixed state vector into the actual model stress and strain input axes:

```python
  [mixed]
    type = MixedControlSetup
  []
  [mixed_old]
    type = MixedControlSetup
    control = "old_forces/control"
    mixed_state = "old_state/mixed_state"
    fixed_values = "old_forces/fixed_values"
    cauchy_stress = "old_state/S"
    strain = "old_forces/E"
  []
```

The second modification renames the resulting residual to map to the correct name for the mixed state:

```python
  [rename]
    type = CopySR2
    from = "residual/S"
    to = "residual/mixed_state"
  []
```

In this example the "base" model is setup for strain control, so that the residual is formed on stress.  For "base" stress control the only change would be the name of the `from` parameter in this object.

These two modifications will allow the model to be run in mixed control.  One additional modification clarifies the output by mapping the mixed input and state back to stress and strain tensors

```python
  [mixed_output]
    type = MixedControlSetup
    cauchy_stress = 'output/stress'
    strain = 'output/strain'
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
