# Phase-Field Fracture {#phase-field-fracture}

[TOC]

The **phase-field fracture** physics module is a collection of objects serving as building blocks for composing models describing the evolution of regularized fracture, and the accompanying loss of materials' load-bearing capacity. The models defined in this module offer a set of phase-field fracture constitutive choices that can be utilized to set up a fracture simulation using both staggered and monolithic solution schemes.

This module can be coupled with other physics modules to describe energy dissipation coupled with fracture. For example, the **solid mechanics** module can be coupled to simulate a variety of (visco)plastic dissipation. This documentation offers a short description of each object under this module followed by a comprehensive example showing how each of them can be used to compose a bigger model describing fracture evolution locally at independent material points.

## Crack geometric functions

The crack geometric function, \f$ \alpha(\phi) \f$ determines the distribution of the crack phase field that governs the "shape" of the smeared crack. Given the phase field, \f$ \phi \f$, it calculates the corresponding scalar-valued crack geometric function. Below are some example crack geometric functions.

- [CrackGeometricFunctionAT1](#crackgeometricfunctionat1)
- [CrackGeometricFunctionAT2](#crackgeometricfunctionat2)

## Degradation functions

Generally, as the fracture evolves, the material gradually loses its load-carrying capacity. Given the phase field, \f$ \phi \f$, the degradation function, \f$ g(\phi) \f$ calculates the corresponding scalar-valued function to degrade the materials' stiffness. Some example degradation functions are listed below.

- [PowerDegradationFunction](#powerdegradationfunction)
- [RationalDegradationFunction](#rationaldegradationfunction)

## Strain Energy Density

This module also offers definitions of the strain energy density to facilitate the setup of variational constitutive update. Given the strain as input, the strain energy density objects calculate the active part of the strain energy density, \f$ \psi_\mathrm{active} \f$ that drives the fracture propagation and the inactive part, \f$ \psi_\mathrm{inactive} \f$.


## Example

In this section, we'll describe how everything discussed above fits together to compose a bigger model that can be used to simulate fracture evolution at material points. Consider the following governing equation for crack propagation

\f{align}
 f & = \eta \dot{\phi} + \frac{\partial}{\partial \phi} \left( \alpha \frac{G_c}{c_0 l} + g(\phi) \psi_\mathrm{active} \right) \ge 0, \quad \dot{\phi} \ge 0, \quad f \dot{\phi} = 0,

\f}
which can be equivalently formulated using the Fischer-Burmeister complementarity condition as

\f{align}
 f + \dot{\phi} - \sqrt{ f^2 + \dot{\phi}^2 } = 0.
\f}

The complementarity condition can be implicitly solved for the phase field, \f$ \phi \f$ using the [ImplicitUpdate](#implicitupdate) model. The implicit equation can also be coupled with other physics. The following example input file walks through the model composition where \f$ d \f$ is used instead of \f$ \phi \f$ as the phase-field variable for the sake of brevity.

```
[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'forces/E'
    force_SR2_values = 'strains'
    predictor = LINEAR_EXTRAPOLATION
    save_as = 'fb_pff_result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
  []
[]

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 3
    nstep = 1000
  []
  [exx]
    type = FullScalar
    value = 0.016
  []
  [eyy]
    type = FullScalar
    value = -0.008
  []
  [ezz]
    type = FullScalar
    value = -0.008
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 1000
  []
  [p]
    type = Scalar
    values = 2
  []
  [GcbylbyCo]
    type = Scalar
    values = 0.0152 # Gc/l/Co with Gc = 95 N/m, l = 3.125 mm, Co = 2
  []
[]

[Models]
  # strain energy density: g * psie0
  [degrade]
    type = PowerDegradationFunction
    phase = 'state/d'
    degradation = 'state/g'
    power = 'p'
  []
  [sed0]
    type = LinearIsotropicStrainEnergyDensity
    strain = 'forces/E'
    strain_energy_density_active = 'state/psie_active'
    strain_energy_density_inactive = 'state/psie_inactive'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients = '25.84e3 0.18'
    # decomposition = 'NONE'
    decomposition = 'VOLDEV'
  []
  [sed1]
    type = ScalarMultiplication
    from_var = 'state/g state/psie_active'
    to_var = 'state/psie_degraded'
  []
  [sed]
    type = ScalarLinearCombination
    from_var = 'state/psie_degraded state/psie_inactive'
    to_var = 'state/psie'
    coefficients = '1 1'
  []
  # crack geometric function: alpha
  [cracked]
    type = CrackGeometricFunctionAT2
    phase = 'state/d'
    crack = 'state/alpha'
  []
  # total energy
  [sum]
    type = ScalarLinearCombination
    from_var = 'state/alpha state/psie'
    to_var = 'state/psi'
    coefficients = 'GcbylbyCo 1'
  []
  [energy] # this guy maps from (strain, d) -> energy
    type = ComposedModel
    models = 'degrade sed0 sed1 sed cracked sum'
  []
  # phase rate, follows from the variation of total energy w.r.t. phase field
  [dpsidd]
    type = Normality
    model = 'energy'
    function = 'state/psi'
    from = 'state/d'
    to = 'state/dpsi_dd'
  []
  # obtain d_rate
  [drate]
    type = ScalarVariableRate
    variable = 'state/d'
    rate = 'state/d_rate'
  []
  # define functional
  [functional]
    type = ScalarLinearCombination
    from_var = 'state/dpsi_dd state/d_rate'
    to_var = 'state/F'
    coefficients = '1 1'
  []
  # Fischer Burmeister Complementary Condition
  [Fisch_Burm]
    type = FischerBurmeister
    first_var = 'state/F'
    second_var = 'state/d_rate'
    fischer_burmeister = 'residual/d'

  []
  # system of equations
  [eq]
    type = ComposedModel
    models = 'Fisch_Burm functional drate dpsidd'
  []
  # solve for d
  [solve_d]
    type = ImplicitUpdate
    implicit_model = 'eq'
    solver = 'newton'
  []
  # After the solve take the derivative of the total energy w.r.t. strain to get stress
  [stress]
    type = Normality
    model = 'energy'
    function = 'state/psi'
    from = 'forces/E'
    to = 'state/S'
  []
  [model]
    type = ComposedModel
    models = 'solve_d stress'
    additional_outputs = 'state/d'
  []
[]
```

The stress evaluated at the end of the input file acts only as a post-processor to verify our results.

\anchor singlephase

![Evolution of the phase field][singlephase]{html: width=95%}
[singlephase]: asset/phase.png

\anchor singlestress

![Corresponding degradation of the uniaxial stress over time][singlestress]{html: width=95%}
[singlestress]: asset/stress.png




