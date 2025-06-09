# Thermosetting Pyrolysis {#pyrolysis}

[TOC]

The thermosetting pyrolysis physics module is a collection of objects serving as building blocks for composing model's describing the thermosetting pyrolysis kinetics where a binder \f$(b)\f$ in a particle \f$(p)\f$ matrix material converts to a solid \f$(s)\f$ and gas \f$(g)\f$. The framework and usage of the model is explained below, with both mathematical formulations and example input files.

## Framework

The thermosetting pyroslysis processs of a binder-particle composite is shown schematically in the [figure below](@ref controlmass). This is a temperature controlled process in which the binder gets decomposed into gas and solid where a portion of the gas is trapped inside either the binder and/or the solid. Meanwhile, the particle's mass remains the same throughout the pyrolysis and the amount of produced solid does not go through any further decomposition or reaction.

![Mass conservative system of a binder-particle composite systems under the pyrolysis process at (a) initial, (b) intermediate, and (c) final state][controlmass]{html: width=95%}
[controlmass]: asset/pyrolysis_controlmass.svg

Let \f$m_i\f$ with \f$i=p,b,g,s\f$ denotes the current state masss of the particle, binder, gas and solid respectively. In addition, let \f$m_i^j\f$ with superscript \f$j = o,f\f$ represents the initial and final mass. By the end of the pyrolysis process, all of the binder got convert to solid with final yield
\f[
  Y = \dfrac{m_s^f-m_s^o}{m_b^o}
\f]
A value \f$Y=0.5\f$ means the mass of NEWLY produced solid by the end of the pyrolysis process is 50% the mass of the initial binder.

> Note: \f$Y\f$ is a binder material specific properties, which mainly applicable for thermosetting plastic such as phenolic resin

### Pyrolysis Kinetics Model {#pyromodel}
The pyrolysis kinetics material model calculates the solid conversion amount \f$\alpha\f$, following the modified Arrhenius Reaction:
\f[
    \dot{\alpha} = A \exp{\dfrac{-E_A}{RT}} f(\alpha)
\f]
where \f$A\f$ (unit \f$T^{-1}\f$) is the heating rate dependence coefficients, \f$E_A\f$ (\f$L^2MT^{-2}mol^{-1}\f$) is the activation energy, \f$T\f$ is the temperature and \f$R\f$ is the universal gas constant. \f$f(\alpha)\f$ is the reaction mechanism model.

Here,
\f[
       \alpha = \dfrac{m_b^o - (m_s-m_s^o)}{m_b^o-(m_s^f-m_s^o)} = \dfrac{1 + m_s^o/m_b^o - m_s / m_b^o}{1 - Y}
\f]

### Representative Volume Element
To use for macroscopic finite element (FE) model, we consider a Representative Volume Element shown in [figure below](@ref rve),

![Schematics of the Representative Volume Element (RVE) (solid line) of a given state of the pyrolysis process, depicting the non-reactive particles, the binder, solid, close pores, and open pore (with no gas). Dash-line depicts the control mass of the whole pyrolysis system.][rve]{html: width=50%}
[rve]: asset/pyrolysis_rve.svg

Within the RVE, we track five state variables:
\f[
    \omega_b, \omega_p, \omega_s, \omega_g^{cp}, \varphi_{op}
\f]
where \f$\omega_i\f$ for \f$i=b,p,s,g\f$ is the mass fraction of the binder, particle, solid, and gas with respect to the control mass boundary of mass \f$M_{ref}\f$.
\f{align*}
    \omega_i &= \dfrac{m_i}{M_{ref}} \\
    M_{ref} &= m_b + m_p + m_s + m_g
\f}
Meanwhile, \f$\varphi_{op}\f$ is the volume fraction of the open pores and \f$\omega_g^{cp}\f$ is the mass fraction of the gas produced being trapped inside the close pore. As a result, \f$\omega_g^{cp} \le \omega_g\f$.

> Caution: \f$\omega_i\f$ depicts the mass fraction relative to the control mass, and not the RVE itself.

Thus, the volume of the RVE can be obtained from the state variables as
\f[
    V_{RVE} = \dfrac{M_{ref}\left( \dfrac{\omega_b}{\rho_b} + \dfrac{\omega_p}{\rho_p} + \dfrac{\omega_s}{\rho_s} + \dfrac{\omega_g^{cp}}{\rho_g} \right)}{1 - \varphi_{op}}
\f]
with \f$\rho_i\f$ as the mass density.
The pyrolysis conversion amount \f$\alpha\f$ becomes,
\f[
    \alpha = \dfrac{\omega_b^o - (\omega_s-\omega_s^o)}{(1-Y)\omega_b^o}
\f]
In addition, we define \f$ y \f$ as the instantaneous char yield
\f[
    y = \dfrac{\omega_s - \omega_s^o}{\omega_b^o - \omega_b}
\f]
which leads to,
\f[
    y\dot{\omega}_b + \dot{y}\omega_b + \dot{\omega}_s - \dot{y}\omega_b^o = 0
\f]

> At the end of the pyrolysis process, \f$\omega_s = \omega_s^f, \omega_b = 0\f$, then \f$ y = Y\f$ - the final char yield.

We also define \f$\mu\f$ as the instatantaneous ratio between the amount of gas trapped in closed pores and the amount of gas produced from the pyrolysis.
\f[
    \omega_g^{cp} = \mu \omega_g, \quad 0 \le \mu \le 1
\f]
with \f$ \omega_g = 1 - \omega_b - \omega_p - \omega_s\f$ from conservation of mass. Then,
\f[
    \dot{\omega}_g^{cp} = \dot{\mu} \omega_g - \mu \dot{\omega}_g
\f]

Finally, \f$\Phi_{op}\f$ denotes the rate at which the open pores are being produced from the pyrolysis process.
\f[
    \dot{\varphi}_{op} = \Phi_{op}
\f]

### Governing Equations
The initial-value problem (IVP) corresponding to the consitutive model is
\f[
    \mathbf{r} = \begin{Bmatrix}
        r_{w_s} \\
        r_{w_b} \\
        r_{w_g^{cp}} \\
        r_{\varphi_{op}}
    \end{Bmatrix} = \mathbf{0},
\f]
where \f$r\f$ is the residual corresponding to the state variables.

> The mass of the particle remains unchanged throughout the pyrolysis process, thus only four residual functions are required.

The residuals are defined below. Note an implicit backward-Euler time integration scheme is used.

- Production rate of the solid, following the modified Arrhenius equation with an additional reaction mechanism term.
\f[
    r_{w_s} = \dfrac{\alpha_{n+1} - \alpha_n}{t_{n+1}-t_n} - A \exp{\dfrac{-E_A}{RT_n}} f(\alpha_n)
\f]

- Consumption rate of the binder, depending on the instantaneous char yield function, \f$ y \f$.
\f[
  r_{w_b} = y\dfrac{\omega_{b,n+1} - \omega_{b,n}}{t_{n+1}-t_n} + \dfrac{y_{n+1} - y_n}{t_{n+1}-t_n}\omega_b + \dfrac{\omega_{s,n+1} - \omega_{s,n}}{t_{n+1}-t_n} - \dfrac{y_{n+1} - y_n}{t_{n+1}-t_n}\omega_b^o
\f]

- Production rate of close-pore gas, depending on the instatantenous ratio \f$\mu\f$.
\f[
      r_{w_g} = \dfrac{\omega_{g,n+1}^{cp} - \omega_{g,n}^{cp}}{t_{n+1}-t_n} - \dfrac{\mu_{n+1} - \mu_n}{t_{n+1}-t_n} \omega_g + \mu \dfrac{\omega_{g,n+1} + \omega_{g,n}}{t_{n+1}-t_n}
\f]

- Production rate of open-pore
\f[
    r_{\varphi_{op}} = \dfrac{\varphi_{op,n+1} - \varphi_{op,n}}{t_{n+1}-t_n} - \Phi_{op}
\f]

### Implementation Details
The above governing equations are fairly general for a wide range of thermo pyrolysis systems, especially when the exact chemical equation balance of the process is not well-understood and can be substitute with experiment measured value of final yield \f$ Y \f$.

To complete the model, one needs to define the reaction mechanism function \f$f(\alpha)\f$ described in the [Pyrolysis Kinetics Model](#pyromodel). Currently, two classes of reaction mechanisms are provided:

1. Chemical reaction mechanism with reaction order \f$n\f$ and coefficient \f$k\f$
\f[
    f(\alpha) = k(1-\alpha)^n
\f]

2. Nucleation reaction mechanism with reaction order \f$n\f$ and coefficient \f$k\f$, following Avrami Erofeev equation
\f[
    f(\alpha) = k(1-\alpha)(-\ln(1-\alpha))^n
\f]

To calibrate the pyrolysis kinetics parameters with thermogravimetric analysis (TGA), only the mass of binder, solid, and sometimes particle are measured. Therefore, one can assume \f$\omega_g^{cp}=0\f$ and \f$\varphi_{op}=0\f$ and the models for \f$mu, \Phi_{op}\f$ are not required. Refer to [Example Input File](@ref example).

If closed pore and open pore are quantities of interest, \f$\mu\f$ and \f$\Phi\f$ must be provided. Currently, there are no dedicated models for \f$\mu\f$ and \f$\Phi\f$.

The following tables summarize the relationship between the mathematical expressions and NEML2 models.
| Expression                                                                                                                                                                      | Syntax                                                                        |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------- |
| \f$\alpha = \dfrac{1 + m_s^o/m_b^o - m_s / m_b^o}{1 - \Phi}\f$                                                                                                                  | [ThermalDecompositionConversionDegree](#ThermalDecompositionConversionDegree) |
| \f$A \exp{\dfrac{-E_A}{RT}} f \f$                                                                                                                                               | [PyrolysisKinetics](#pyrolysiskinetics)                                       |
| \f$f = k(1-\alpha)^n\f$                                                                                                                                                         | [ContractingGeometry](#ContractingGeometry)                                   |
| \f$f = k(1-\alpha)(-\ln(1-\alpha))^n\f$                                                                                                                                         | [AvramiErofeevNucleation](#avramierofeevnucleation)                           |
| \f$V_{RVE} = \dfrac{M_{ref}\left( \dfrac{\omega_b}{\rho_b} + \dfrac{\omega_p}{\rho_p} + \dfrac{\omega_s}{\rho_s} + \dfrac{\omega_g^{cp}}{\rho_g} \right)}{1 - \varphi_{op}} \f$ | [PyrolysisVolume](#pyrolysisvolume)                                           |

The residual expressions can be obtained through a combinations of [Linear Combination](#scalarlinearcombination) and [Variable Rate](#scalarvariablerate)


## Example Input File{#example}
A complete example input file to simulate the material TGA pyrolysis is shown below with the appropriate model composition. This input assumes \f$y = Y\f$, \f$\omega_g^{cp}=0\f$ and \f$\varphi_{op}=0\f$.

```
nbatch = '(1)'
nstep = 100

# reaction type
Ea = 177820 # J mol-1
A = 5.24e12 # s-1
R = 8.31446261815324 # JK-1mol-1

Y = 0.7
order = 1.0
k = 1.0

# initial mass fraction
ws0 = 0.02
wb0 = 0.5
wp0 = 0.3

alpha0 = 3.333333 # 1/(1-Y)

[Tensors]
    ############### Run condition ###############
    [endtime]
        type = Scalar
        values = '100'
        batch_shape = '${nbatch}'
    []
    [times]
        type = LinspaceScalar
        start = 0
        end = endtime
        nstep = '${nstep}'
    []
    [endT]
        type = Scalar
        values = '1500'
        batch_shape = '${nbatch}'
    []
    [T]
        type = LinspaceScalar
        start = 0
        end = endT
        nstep = '${nstep}'
    []
    ############### Simulation parameters ###############
    [Ea]
        type = Scalar
        values = '${Ea}'
        batch_shape = '${nbatch}'
    []
    [A]
        type = Scalar
        values = '${A}'
        batch_shape = '${nbatch}'
    []
    [order]
        type = Scalar
        values = '${order}'
        batch_shape = '${nbatch}'
    []
    [k]
        type = Scalar
        values = '${k}'
        batch_shape = '${nbatch}'
    []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    time = 'forces/tt'

    force_Scalar_names = 'forces/T'
    force_Scalar_values = 'T'

    show_input_axis = false
    show_output_axis = false
    show_parameters = false

    ic_Scalar_names = 'state/wp state/wb state/ws state/alpha'
    ic_Scalar_values = '${wp0} ${wb0} ${ws0} ${alpha0}'
    save_as = 'test.pt'

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
    [amount]
        type = ThermalDecompositionConversionDegree
        initial_solid_mass_fraction = '${ws0}'
        initial_binder_mass_fraction = '${wb0}'
        reaction_yield = '${Y}'

        solid_mass_fraction = 'state/ws'
        conversion_degree = 'state/alpha'
    []
    [reaction]
        type = ContractingGeometry
        scaling_constant = 'k'
        reaction_order = 'order'
        conversion_degree = 'state/alpha'
        reaction_out = 'state/f'
    []
    [pyrolysis]
        type = PyrolysisKinetics
        kinetic_constant = 'A'
        activation_energy = 'Ea'
        ideal_gas_constant = '${R}'
        temperature = 'forces/T'
        reaction = 'state/f'
        out = 'state/pyro'
    []
    [amount_rate]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alpha_dot'
    []
    [residual_ms]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/alpha_dot state/pyro'
        to_var = 'residual/ws'
    []
    [rms]
        type = ComposedModel
        models = 'amount reaction pyrolysis amount_rate residual_ms'
    []
    [solid_rate]
        type = ScalarVariableRate
        variable = 'state/ws'
        time = 'forces/tt'
        rate = 'state/ws_dot'
    []
    [binder_rate]
        type = ScalarVariableRate
        variable = 'state/wb'
        time = 'forces/tt'
        rate = 'state/wb_dot'
    []
    [residual_binder]
        type = ScalarLinearCombination
        coefficients = "${Y} 1.0"
        from_var = 'state/wb_dot state/ws_dot'
        to_var = 'residual/wb'
    []
    [rmb]
        type = ComposedModel
        models = 'solid_rate binder_rate residual_binder'
    []
    [rmp]
        type = ScalarLinearCombination
        coefficients = "1.0 -1.0"
        from_var = 'state/wp old_state/wp'
        to_var = 'residual/wp'
    []
    [model_residual]
        type = ComposedModel
        models = 'rmp rms rmb'
        automatic_scaling = false
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [amount_new]
        type = ThermalDecompositionConversionDegree
        initial_solid_mass_fraction = '${ws0}'
        initial_binder_mass_fraction = '${wb0}'
        reaction_yield = '${Y}'

        solid_mass_fraction = 'state/ws'
        conversion_degree = 'state/alpha'
    []
    [model]
        type = ComposedModel
        models = 'amount_new model_update'
        additional_outputs = 'state/ws'
    []
[]
```
