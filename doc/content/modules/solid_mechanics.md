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

## Cohesive Zone Models

Cohesive zone models (CZMs) describe the constitutive response of an interface between two material regions. All NEML2 CZMs derive from `TractionSeparationModel`, which declares the standard interface: a displacement jump vector

\f[
  \boldsymbol{\delta} = [\delta_n,\; \delta_{s1},\; \delta_{s2}]
\f]

expressed in the local interface frame (normal \f$ n \f$ plus two tangential directions \f$ s_1, s_2 \f$), and an interface traction vector

\f[
  \mathbf{T} = [T_n,\; T_{s1},\; T_{s2}].
\f]

The displacement jump is read from `forces/displacement_jump` and the traction is written to `state/traction`. Positive \f$ \delta_n \f$ denotes interface opening; negative values correspond to compression.

### PureElasticTractionSeparation

The simplest traction-separation law assumes a linear elastic relationship with independent normal and tangential stiffnesses:

\f[
  T_n    = K_n\, \delta_n, \qquad
  T_{s1} = K_t\, \delta_{s1}, \qquad
  T_{s2} = K_t\, \delta_{s2}.
\f]

In matrix form, \f$ \mathbf{T} = \operatorname{diag}(K_n, K_t, K_t)\, \boldsymbol{\delta} \f$. The model is stateless (no history variables) and provides an exact analytic Jacobian.

| Parameter | Description | Units |
|-----------|-------------|-------|
| `normal_stiffness` \f$ K_n \f$ | Penalty stiffness in the opening direction | stress / length |
| `tangent_stiffness` \f$ K_t \f$ | Penalty stiffness in both sliding directions | stress / length |

@list-input:tests/unit/models/solid_mechanics/cohesive/PureElasticTractionSeparation.i:Models

### SalehaniIrani3DCTraction

This model implements the 3D exponential cohesive law of Salehani & Irani (2018). The traction components decay exponentially with interface opening:

\f[
  T_i = a_i \frac{\delta_i}{\delta_{i,0}} \exp(-x),
\f]

where the coupling exponent is

\f[
  x = \frac{\delta_n}{\delta_{n,0}} + \left(\frac{\delta_{s1}}{\delta_{t,0}}\right)^2 + \left(\frac{\delta_{s2}}{\delta_{t,0}}\right)^2,
\f]

and the prefactors are

\f[
  a_0 = e\, T_{n,\max}, \qquad a_{12} = \sqrt{2e}\, T_{s,\max}.
\f]

The characteristic tangential gap used internally is \f$ \delta_{t,0} = \sqrt{2}\, \delta_{t,\text{in}} \f$, where \f$ \delta_{t,\text{in}} \f$ is the user-supplied input. The model provides a full \f$ 3 \times 3 \f$ analytic Jacobian (not diagonal because \f$ x \f$ couples all three components).

| Parameter | Description | Units |
|-----------|-------------|-------|
| `normal_gap_at_maximum_normal_traction` \f$ \delta_{n,0} \f$ | Characteristic normal gap at peak normal traction | length |
| `tangential_gap_at_maximum_shear_traction` \f$ \delta_{t,\text{in}} \f$ | Characteristic tangential gap input (stored as \f$ \delta_{t,0} = \sqrt{2}\,\delta_{t,\text{in}} \f$) | length |
| `maximum_normal_traction` \f$ T_{n,\max} \f$ | Peak normal traction | stress |
| `maximum_shear_traction` \f$ T_{s,\max} \f$ | Peak shear traction | stress |

@list-input:tests/unit/models/solid_mechanics/cohesive/SalehaniIrani3DCTraction.i:Models

### BiLinearMixedModeTraction

This model implements the bilinear mixed-mode damage law of Camanho & Davila (NASA/TM-2002-211737). It tracks a scalar damage variable \f$ d \in [0, 1] \f$ that is stored in `state/damage` and evolves irreversibly (it can only increase) with optional viscous regularization.

#### Bilinear damage law

The effective mixed-mode displacement jump is

\f[
  \delta_m = \sqrt{\langle \delta_n \rangle^2 + \delta_{s1}^2 + \delta_{s2}^2},
\f]

where \f$ \langle \cdot \rangle \f$ denotes the Macaulay bracket (zero for compression). Two characteristic values depend on the mode-mixity ratio \f$ \beta = \delta_s / \delta_n \f$:

- **Damage-initiation jump** \f$ \delta_{\text{init}} \f$ (Benzeggagh-Kenane mixed-mode onset criterion):
\f[
  \delta_{\text{init}} = \frac{\delta_{n,0}\,\delta_{s,0}\,\sqrt{1+\beta^2}}{\sqrt{\delta_{s,0}^2 + (\beta\,\delta_{n,0})^2}},
\f]
  where \f$ \delta_{n,0} = N/K \f$ and \f$ \delta_{s,0} = S/K \f$.

- **Full-degradation jump** \f$ \delta_{\text{final}} \f$ from the selected fracture criterion (see below).

The bilinear damage variable is

\f[
  d_{\text{bilinear}} = \frac{\delta_{\text{final}}\,(\delta_m - \delta_{\text{init}})}{\delta_m\,(\delta_{\text{final}} - \delta_{\text{init}})}.
\f]

#### Irreversibility and viscous regularization

The irreversibility constraint \f$ d \geq d_{\text{old}} \f$ is enforced by taking the maximum of the trial damage and the previous-step damage. Viscous regularization with coefficient \f$ \mu \f$ then gives the updated damage:

\f[
  d = \frac{d_{\text{irreversible}} + (\mu / \Delta t)\, d_{\text{old}}}{\mu / \Delta t + 1}.
\f]

Setting \f$ \mu = 0 \f$ disables regularization.

#### Traction

The traction follows a standard continuum damage form with compression cutoff (the normal traction is not degraded under compression):

\f[
  T_n    = (1-d)\,K\,\langle\delta_n\rangle + K\,(\delta_n - \langle\delta_n\rangle), \qquad
  T_{si} = (1-d)\,K\,\delta_{si}.
\f]

#### Mixed-mode fracture criteria

Two criteria are available via the `criterion` option:

- **BK** (Benzeggagh-Kenane, default):
\f[
  \delta_{\text{final}} = \frac{2}{K\,\delta_{\text{init}}} \left[ G_{\mathrm{Ic}} + (G_{\mathrm{IIc}} - G_{\mathrm{Ic}}) \left(\frac{\beta^2}{1+\beta^2}\right)^\eta \right].
\f]

- **POWER_LAW**:
\f[
  \delta_{\text{final}} = \frac{2(1+\beta^2)}{K\,\delta_{\text{init}}} \left[ G_{\mathrm{Ic}}^{-\eta} + \left(\frac{\beta^2}{G_{\mathrm{IIc}}}\right)^\eta \right]^{-1/\eta}.
\f]

#### Variables

| Variable | Axis path | Description |
|----------|-----------|-------------|
| `displacement_jump` (input) | `forces/displacement_jump` | Current displacement jump \f$ \boldsymbol{\delta} \f$ |
| `traction` (output) | `state/traction` | Interface traction \f$ \mathbf{T} \f$ |
| `damage` (output) | `state/damage` | Damage variable \f$ d \f$ after irreversibility and regularization |
| `damage_old` (input) | `old_state/damage` | Damage variable from the previous step |
| `displacement_jump_old` (input) | `old_forces/displacement_jump` | Displacement jump from the previous step |
| `time` (input) | `forces/t` | Current time |
| `time_old` (input) | `old_forces/t` | Time at start of the step |

#### Parameters

| Parameter | Symbol | Description | Units |
|-----------|--------|-------------|-------|
| `penalty_stiffness` | \f$ K \f$ | Elastic penalty stiffness | stress / length |
| `mode_I_critical_fracture_energy` | \f$ G_{\mathrm{Ic}} \f$ | Mode I critical energy release rate | energy / area |
| `mode_II_critical_fracture_energy` | \f$ G_{\mathrm{IIc}} \f$ | Mode II critical energy release rate | energy / area |
| `normal_strength` | \f$ N \f$ | Tensile interface strength | stress |
| `shear_strength` | \f$ S \f$ | Shear interface strength | stress |
| `mixed_mode_exponent` | \f$ \eta \f$ | Exponent for BK or power-law criterion | dimensionless |
| `viscosity` | \f$ \mu \f$ | Viscous regularization coefficient (0 = off) | time |
| `criterion` | — | `"BK"` or `"POWER_LAW"` | — |
| `lag_mode_mixity` | — | Use previous-step \f$ \boldsymbol{\delta} \f$ when computing \f$ \beta \f$, \f$ \delta_{\text{init}} \f$, \f$ \delta_{\text{final}} \f$ (default `true`) | — |
| `lag_displacement_jump` | — | Use previous-step \f$ \boldsymbol{\delta} \f$ when computing \f$ \delta_m \f$ (default `false`) | — |

@list-input:tests/unit/models/solid_mechanics/cohesive/BiLinearMixedModeTraction.i:Models
