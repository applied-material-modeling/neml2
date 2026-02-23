# Thermosetting Pyrolysis {#thermosetting-pyrolysis}

[TOC]

This example demonstrates the use of the chemical reactions physics module to compose models for pyrolysis kinetics of a thermoset material, where a binder is burnt into char. The framework and usage of the model is explained below, with both mathematical formulations and example input files.

## Framework

The thermosetting pyroslysis processs of a binder-particle composite is shown schematically in the [figure below](@ref controlmass). This is a temperature-controlled process in which the precursor (the binder) gets decomposed into gas and a residue solid. A portion of the gas is trapped inside the binder and/or the solid. It is assumed that the particle phase does not participate in the reaction and its mass remains constant. It is also assumed that the residue does not further decompose or react.

\anchor controlmass

![Mass conservative system of a binder-particle composite under the pyrolysis process at the (a) initial, (b) intermediate, and (c) final state.][controlmass]{html: width=95%}
[controlmass]: asset/pyrolysis_controlmass.svg

Let \f$ \omega_i \f$ with \f$i=p,b,g,c\f$ denote the current weight fraction of the particle, binder, gas, and char, respectively. When the reaction is complete, all of the binder gets converted to char with final yield (often experimentally measured using thermogravimetric analysis, or TGA).

## Reaction kinetics

In this example, the reaction kinetics is defined by the [contracting geometry model](#contractinggeometry), i.e.
\f[
    \dot{\alpha} = k \left( 1-\alpha \right)^n,
\f]
where the reaction coefficient \f$k\f$ is temperature dependent following an Arrhenius relation.

## Control volume

To couple with other physics based on control volume, we consider the control volume shown in [figure below](@ref pyrolysis_rve),

\anchor pyrolysis_rve

![Schematics of the Representative Volume Element (RVE) (solid line) at a given state during pyrolysis, depicting the non-reactive particles, the binder, char, close pores, and open pore (with no gas). The dashed line depicts the control mass of the reaction system.][pyrolysis_rve]{html: width=50%}
[pyrolysis_rve]: asset/pyrolysis_rve.svg

Within the RVE, we track four independent state variables: \f$ \omega_b, \omega_c, \omega_g, \varphi_{o} \f$, where \f$\omega_b\f$, \f$\omega_c\f$, and \f$\omega_g\f$ are the mass fraction of the binder, char, and gas, respectively; \f$\varphi_{o}\f$ is the volume fraction of the open pore.

Thus, the effective volume of the RVE can be calculated as
\f[
    V = \dfrac{M_\mathrm{ref}}{1-\varphi_o} \left( \dfrac{\omega_b}{\rho_b} + \dfrac{\omega_p}{\rho_p} + \dfrac{\omega_c}{\rho_s} + \dfrac{\omega_g}{\rho_g} \right),
\f]
with \f$\rho\f$ as the mass density.

We define \f$\mu \in [0, 1]\f$ as the instantaneous ratio between the amount of gas trapped in closed pores and the total amount of gas produced, i.e.
\f[
    \dot{\omega}_g = \mu \left( \dot{\omega}_g + \dot{\omega}_{g, o} \right).
\f]
Following conservation of mass, the gas production rate can be expressed as
\f[
    \dot{\omega}_g = -\mu \left( \dot{\omega}_b + \dot{\omega}_c \right).
\f]

Finally, the production rate of open pore is given as
\f[
    \dot{\varphi}_{o} = \zeta \dot{\alpha},
\f]
where \f$ \zeta \f$ is total volume fraction of the open pore upon reaction completion.

## Implementation details

The following table summarizes the relationship between the mathematical expressions and the NEML2 models.
| Expression                                                                                                                                                              | Syntax                                      |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------ |
| \f$ \dot{\alpha} = k \left( 1-\alpha \right)^n \f$                                                                                                                      | [ContractingGeometry](#contractinggeometry) |
| \f$k = k_0 \exp{\dfrac{-Q}{RT}} \f$                                                                                                                                     | [ArrheniusParameter](#arrheniusparameter)   |
| \f$V = \dfrac{M_\mathrm{ref}}{1-\varphi_o} \left( \dfrac{\omega_b}{\rho_b} + \dfrac{\omega_p}{\rho_p} + \dfrac{\omega_c}{\rho_s} + \dfrac{\omega_g}{\rho_g} \right) \f$ | [EffectiveVolume](#effectivevolume)         |

The state variables are integrated in time using the backward-Euler scheme.

## Example Input File

```
nbatch = '(1)'
nstep = 100

# reaction mechanism
Y = 0.5835777126099713 # yield
n = 1.0 # reaction order
k0 = 0.04210147513030456 # reaction rate coefficient
Q = 21191.61425138572 # J/mol
R = 8.31446261815324 # J/K/mol

# initial mass fraction
wc0 = 0 # char (residue)
wb0 = 1 # binder (precursor)

# control volume
mu = 0.2
zeta = 0.05

[Tensors]
  [yield]
    type = Scalar
    values = '${Y}'
  []
  [endtime]
    type = Scalar
    values = '2700'
    batch_shape = '${nbatch}'
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = endtime
    nstep = '${nstep}'
  []
  [T]
    type = LinspaceScalar
    start = '300'
    end = '1500'
    nstep = '${nstep}'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    time = 'forces/t'

    force_Scalar_names = 'forces/T'
    force_Scalar_values = 'T'

    ic_Scalar_names = 'state/wb state/wc'
    ic_Scalar_values = '${wb0} ${wc0}'
    save_as = 'result.pt'

    verbose = false
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Solvers]
  [newton]
    type = Newton
    verbose = false
  []
[]

[Models]
  [reaction_coef]
    type = ArrheniusParameter
    reference_value = '${k0}'
    activation_energy = '${Q}'
    ideal_gas_constant = '${R}'
    temperature = 'forces/T'
    parameter = 'state/k'
  []
  [reaction_rate]
    type = ContractingGeometry
    reaction_coef = 'reaction_coef'
    reaction_order = '${n}'
    conversion_degree = 'state/alpha'
    reaction_rate = 'state/alpha_rate'
  []
  [reaction_ode]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/alpha'
  []
  [reaction]
    type = ComposedModel
    models = 'reaction_rate reaction_ode'
  []
[]

[EquationSystems]
  [solve_reaction_sys]
    type = NonlinearSystem
    model = 'reaction'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [solve_reaction]
    type = ImplicitUpdate
    equation_system = 'solve_reaction_sys'
    solver = 'newton'
  []
  [binder_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '-1'
    to_var = 'state/wb_rate'
  []
  [char_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '${Y}'
    to_var = 'state/wc_rate'
  []
  [gas_rate]
    type = ScalarLinearCombination
    from_var = 'state/wb_rate state/wc_rate'
    coefficients = '-${mu} -${mu}'
    to_var = 'state/wg_rate'
  []
  [open_pore_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '${zeta}'
    to_var = 'state/phio_rate'
  []
  [binder]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wb'
  []
  [char]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wc'
  []
  [gas]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wg'
  []
  [open_pore]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/phio'
  []
  [model]
    type = ComposedModel
    models = "solve_reaction reaction_rate
              binder_rate char_rate gas_rate open_pore_rate
              binder char gas open_pore"
    additional_outputs = 'state/alpha'
  []
[]
```
