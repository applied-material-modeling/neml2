# Phase Field Fracture {#phase-field-fracture}

[TOC]

The **Phase Field Fracture** physics module is a collection of objects (smaller models) serving as building blocks for composing models describing the evoluion of regularized fracture that converges to sharp crack according to \f$ \boldsymbol{\Gamma} \f$ convergence theorem, and the accompanying loss of materials' load bearing cappacity. The models defined in this module offer a set of phase field fracture constitutive choices that can be utilized to set up a fracture simulation using both staggered and monolithic solution schemes. Currently, the module supports only linear isotropic response. However, they can be coupled with appropriate **Solid Mechanics** objects to simulate a variety of energy dissipation coupled with fracture. They can potentially be coupled with other physics as well. This documentation offers a short description of each object under this module followed by a comprehensive example showing how each of them can be used to compose a bigger model describing fracture locally at a single material point.

## Crack Geometric Functions

The crack geometric function, \f$ \alpha(\phi) \f$ determines the distribution of the crack phase field that governs the shape of the smeared crack. Given the phase field, \f$ \phi \f$, it calculates the corresponding scalar value of the crack geometric function. At present, the module offers the following crack geometric functions.

#### [CrackGeometricFunctionAT1](https://applied-material-modeling.github.io/neml2/syntax-models.html)
Providing linear distribution of the phase field with, \f$ \alpha(\phi) = \phi \f$.

#### [CrackGeometricFunctionAT2](https://applied-material-modeling.github.io/neml2/syntax-models.html)
Providing quadratic distribution of the phase field with, \f$ \alpha(\phi) = \phi^2 \f$.

## Degradation Functions

Generally, as the fracture evolves, the material gradually loses its strength to support tensile loads. Given the phase field, \f$ \phi \f$, degradation functions, \f$ g(\phi) \f$ calculate the corresponding scalar value of the function to degrade the material stiffness. Currently, the module offers the following degradation functions.

#### [PowerDegradationFunction](https://applied-material-modeling.github.io/neml2/syntax-models.html)
Calculates the degradation according to, \f$ g(\phi) = (1-\phi)^p \f$.

#### [RationalDegradationFunction](https://applied-material-modeling.github.io/neml2/syntax-models.html)
Calculates the degradation according to, \f$ g(\phi) = \frac{(1-\phi)^p}{(1-\phi)^p+Q(\phi)} \f$ where \f$ Q(\phi) = b_1\phi (1+b_2 \phi + b_2 b_3 \phi^2) \f$.

## Strain Energy Density

As mentioned in the begining, given a strain tensor as its input, [**LinearIsotropicStrainEnergyDensity**](https://applied-material-modeling.github.io/neml2/syntax-models.html) model calculates the active part of the strain energy density, \f$ \psi_{active} \f$ that drives the fracture propagation and the inactive part, \f$ \psi_{inactive} \f$ according to the linear isotropic response. The following decomposition schemes are supported.

#### NONE
No decomposition of the strain energy density. Therefore,
\f[
 \psi_{active} = \frac{1}{2} \boldsymbol{\sigma} : \boldsymbol{\varepsilon} \,\,\, and \,\,\, \psi_{inactive} = 0
\f]

#### VOLDEV
Strain energy density is decomposed into active and inactive parts following the decomposition of the linearized strain tensor into volumetric and deviatoric components. Therefore,

\f[
 \boldsymbol{\varepsilon} = \boldsymbol{\varepsilon_{vol}} + \boldsymbol{\varepsilon_{dev}}, \,\,\, \boldsymbol{\varepsilon_{vol}} = \frac{1}{3}tr(\boldsymbol{\varepsilon}), \,\,\, \boldsymbol{\varepsilon_{dev}} = \boldsymbol{\varepsilon} - \frac{1}{3}tr(\boldsymbol{\varepsilon})
\f]

\f[
 \psi_{active} = \frac{1}{2} \kappa \langle tr(\boldsymbol{\varepsilon})^2 \rangle_+ + \mu \boldsymbol{\varepsilon_{dev}} : \boldsymbol{\varepsilon_{dev}}
\f]

\f[
 \psi_{inactive} = \frac{1}{2} \kappa \langle tr(\boldsymbol{\varepsilon})^2 \rangle_- 
\f]

## Example

In this section, we'll describe how everything discussed above fits together to compose a bigger model that can be used to simulate fracture evolution at a material point. In the classical phase field fracture, the governing equations that pop out from the minimization of the total energy of the domain - expressed as the sum of the strain energy and fracture energy - are as follows.

\f{align}
 - \div \boldsymbol{\sigma} & = 0 \\ 
 \mathcal{E}_{,\phi} & = - \div \frac{2G_cl}{c_0} \grad \phi + \frac{\partial}{\partial \phi} \left( \alpha \frac{G_c}{c_0 l} + g(\phi) \psi_{active} \right) \ge 0, \,\,\,\, \dot{\phi} \ge 0, \,\,\,\, \mathcal{E}_{,\phi}\dot{\phi}=0 

\f}

While Equation (1) describes the kinematics response, Equation (2) drives the evolution of the phase field describing the fracture of the body. It must be noted that the material stress \f$ \boldsymbol{\sigma} \f$ going in to the Equation (1) is degraded over time as the phase field depicting the crack evolves, i.e., \f$ \boldsymbol{\sigma} = \frac{\partial}{\partial \boldsymbol{\varepsilon}} \left(g(\phi) \psi_{active} \right) \f$. However, for a single material point, by ignoring the non-local terms of the governing equations, we only have the following equation with the KKT (Karush-Kuhn-Tucker) condition for ensuring the crack irreversibility constraint.

\f{align}
 \mathcal{E}_{,\phi}  =  \frac{\partial}{\partial \phi} \left( \alpha \frac{G_c}{c_0 l} + g(\phi) \psi_{active} \right) \ge 0, \,\,\,\, \dot{\phi} \ge 0, \,\,\,\, \mathcal{E}_{,\phi}\dot{\phi}=0 

\f}

The KKT condition associated with Equation (3) can be boiled down to a single equation using the Fischer-Burmeister complementary condition whose residual, \f$ r \f$ is defined below.

\f{align}
 & r =  \mathcal{E}_{,\phi} + \dot{\phi} - \sqrt{\left( \mathcal{E}_{,\phi}^2 + \dot{\phi}^2 \right)} = 0\\
 & \text{where}, \,\,\,\, \mathcal{E}_{,\phi}  =  \frac{\partial}{\partial \phi} \left( \alpha \frac{G_c}{c_0 l} + g(\phi) \psi_{active} \right) \notag
\f}

Now, Equation (4) can be easily implicitly solved for the phase field, \f$ \phi \f$ using the [ImplicitUpdate](https://applied-material-modeling.github.io/neml2/syntax-models.html) with a [Newton](https://applied-material-modeling.github.io/neml2/syntax-solvers.html) solver. The following example input file walks through the model composition and solution of Equation (4) where \f$ d \f$ is used instead of \f$ \phi \f$ as the phase field variable for the sake of brevity. Additionally, \f$ \dot{\phi} \f$ is added to the functional derivative, \f$ \mathcal{E}_{,\phi} \f$ to incorporate some viscous effect that regularizes the phase evolution over the time domain. Rerunning the input file without it will result in a faster phase evolution.

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

High resemblance can be observed between the input file describing model composition step by step and the theory described above. However, it must be noted that the stress evaluated at the end of the input file using [Normality](https://applied-material-modeling.github.io/neml2/syntax-models.html) acts as a postprocessor to verify our results. Consequently, one can easily visualize the phase evolution and resulting stress degradation presented below using the visualization techniques described in the previous tutorials.

\anchor singlephase

![Evolution of the phase field describing fracture from 0 to 1][singlephase]{html: width=95%}
[singlephase]: asset/phase.png

\anchor singlestress

![Corresponding degradation of the uniaxial stress over time][singlestress]{html: width=95%}
[singlestress]: asset/stress.png

Similarly, use of these phase field constitutive models under this module to calculate the material updates at each timestep for a full scale fracture simulation using PDEs involving non-local terms over the whole domain in [MOOSE](https://mooseframework.inl.gov), [RACCOON](https://hugary1995.github.io/raccoon/index.html) or other [MOOSE](https://mooseframework.inl.gov) based app is fairly straight forward.



