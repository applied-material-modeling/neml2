# Reactive Infiltration {#reactive-infiltration}

[TOC]

This example demonstrates the use of the chemical reactions physics module to compose models for reactive infiltration kinetics, as a solid \f$(S)\f$ material is infiltrated by a liquid \f$(L)\f$ to create a product \f$(P)\f$. The framework and usage of the model is explained below, with both mathematical formulations and example input files.

## Representative Volume Element

The infiltration process for a material considers a cylindrical Representative Volume Element (RVE) with radius \f$R\f$ and span \f$H\f$, depicting a solid capillary of radius \f$r_o\f$, corresponding to the initial porosity of \f$\varphi_0\f$. The mathematical description of the model uses cylindrical coordinates and assumes axisymmetry around \f$r = 0\f$.

As the liquid enters the cylinder, it reacts with a solid to form a product with thickness \f$\delta_P\f$, as shown schematically in the [figure below](@ref infiltration_rve).

\anchor infiltration_rve

![The (a) top and (b) cross-section view of the Representative Volume Element (RVE) of a given state of the reactive infiltration process, depicting the outer solid walls, the infiltrated liquid and the product from the chemical reaction ][infiltration_rve]{html: width=80%}
[infiltration_rve]: asset/lsi_rve_simplify.svg

Throughout this process, the solid radius, \f$r_o\f$, decreases while the product thickness, \f$\delta_P\f$, increases. Let \f$r_i\f$ denote the inner radius of the product. In addition, define \f$\alpha_i\f$ (in units of mole per volume) as the amount of substance in the RVE, and \f$\Omega_i = \dfrac{M_i}{\rho_i}\f$ as the molar volume of a material with molar mass \f$M_i\f$ (in units of amu, or gram per mole) and density \f$\rho_i\f$ (mass per volume), with subscripts taking \f$L\f$, \f$S\f$, and \f$P\f$, respectively.

The volume fraction, \f$\varphi_i\f$  of each material is then
\f[
    \varphi_i = \alpha_i \Omega_i
\f]
and the RVE porosity (void) is
\f[
    \varphi_v = 1 - \varphi_L - \varphi_P - \varphi_S
\f]

\anchor assumptions

**Key assumptions** made throughout the derivation are:
-  Liquid remains liquid over the entire course of reaction, i.e., no phase change.
-  The reaction is irreversible.
-  Formation of the initial product layer is immediate once liquid comes into contact with solid.
-  Once product is formed, reaction rate is primarily limited by the diffusion of liquid through product to the product-solid interface, with a diffusion coefficient \f$D\f$.
-  The product wall thickness remains uniform during the infiltration.
-  The only reaction is between the liquid and the solid.

The following nondimensionalization is applied to the constitutive model for the reactive infiltration process.
\f{align*}
    &\ \bar{\delta}_P = \frac{\delta_P}{R}, \quad \bar{r}_o = \dfrac{r_o}{R} = \sqrt{1-\varphi_S}, \quad \bar{r}_i = \dfrac{r_i}{R} = \sqrt{1-\varphi_S-\varphi_P}.
\f}

The complete state of the RVE is denoted by the tuple \f$\left( \varphi_L, \varphi_S, \varphi_P\right)\f$, with \f$\alpha_L\f$ as the prescribed force.

> Mathematically, it is possible that \f$ \varphi_L + \varphi_P + \varphi_S \ge 1 \f$. Physically, this implies "overflow", aka the prescribed \f$\alpha_L\f$ is larger than the available voids. Care must be taken at the macroscopic model to avoid or resolve this issue.

## Governing equations

The initial-value problem (IVP) corresponding to the constitutive model is
\f[
    \mathbf{r} = \begin{Bmatrix}
        r_{\varphi_P} \\
        r_{\varphi_S}
    \end{Bmatrix} = \mathbf{0},
\f]
where \f$r_{\varphi_i}\f$ is the residual corresponding to the volume fraction of the liquid, solid and product. The residuals are defined below. Note an implicit backward-Euler time intergration scheme is used.

- Production rate of the product, dictates by the diffusion of the liquid through the product to react with the solid at the product-solid interface
\f[
    r_{\varphi_P} = \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n} - \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S
\f]
Here \f$R_L(\varphi_L), R_S(\varphi_S) \in [0, 1]\f$ are the reactivity of liquid and solid, represented by a step function of the void fraction, \f$\varphi_i\f$.

- Reaction rate of the solid, with a (solid, product) reaction coefficient, \f$(k_S, k_P)\f$ respectively
\f[
    r_{\varphi_S} = \dfrac{\alpha_{S, n+1} - \alpha_{S, n}}{t_{n+1}-t_n} + \dfrac{k_P}{k_S} \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n}
\f]

## Implementation details

The above governing equations are fairly general for a wide range of reactive infiltration systems within this framework given the chemical reaction kinetics with appropriate reaction coefficient (\f$k_i\f$) and the corresponding molar volume \f$\Omega_i\f$ of the materials.

Example for determining \f$k_i\f$ and \f$\Omega_i\f$:
Consider a reactive infiltration process of liquid (chemical formula \f$L_x\f$) and solid (\f$S_y\f$), with the molar mass \f$M_L\f$ and \f$M_S\f$. The chemical reaction creates a product with chemical formula \f$L_m S_n\f$, with reaction coefficients \f$k_S\f$ and \f$k_P\f$.
\f[
    L_x + k_S S_y \rightarrow k_P L_mS_n
\f]
For stoichiometric balance, the reaction coefficients are:
\f{align*}
    k_S = \dfrac{xn}{ym}, \quad k_P = \dfrac{x}{m}
\f}
The densities of the liquid, solid, and product are respectively \f$\rho_L, \rho_S, \rho_P\f$.

Then, the molar volume is:
\f[
    \Omega_L = \dfrac{x M_L}{\rho_L}. \quad \Omega_S = \dfrac{y M_S}{\rho_S}, \quad \Omega_P = \dfrac{mM_L+nM_S}{\rho_P}.
\f]


The instantaneous balance also implies that
\f[
    \dot{\alpha}_S = \dfrac{k_P}{k_S} \dot{\alpha}_P,
\f]
Since this constitutive model considers \f$\alpha_L\f$ as the forcing function for the IVP problem, \f$\dot{\alpha}_L \ne k_P \dot{\alpha}_P\f$.

The following table summarizes the relationship between the mathematical expressions and the NEML2 models.

| Expression                                                                                                                                                                | Syntax                                                                               |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------- |
| \f$ \bar{r}_i = \sqrt{1-\varphi_S-\varphi_P}; \quad \bar{r}_o = \sqrt{1-\varphi_S}\f$                                                                                     | [CylindricalChannelGeometry](#cylindricalchannelgeometry)                            |
| \f$ \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S\f$                                                                        | [DiffusionLimitedReaction](#diffusionlimitedreaction)                                |
| \f$ R_L; \quad R_S \f$                                                                                                                                                    | [HermiteSmoothStep](#hermitesmoothstep)                                              |
| \f$  \varphi_i = \alpha_i \Omega_i \f$                                                                                                                                    | [Linear Combination](#scalarlinearcombination)                                       |
| \f$ r_{\varphi_P} = \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n} - \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S\f$ | [Linear Combination](#scalarlinearcombination), [Variable Rate](#scalarvariablerate) |
| \f$r_{\varphi_S} = \dfrac{\alpha_{S, n+1} - \alpha_{S, n}}{t_{n+1}-t_n} + \dfrac{k_P}{k_S} \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n}\f$                        | [Linear Combination](#scalarlinearcombination), [Variable Rate](#scalarvariablerate) |

## Complete input file
A complete example input file for the reactive infiltration is shown below with the appropriate model composition.

```
ntime = 200
D = 1e-6
omega_Si = 12.0
omega_C = 5.3
omega_SiC = 12.5
chem_ratio = 1.0

oCm1 = 0.18867924528 # 1/Omega_C
oSiCm1 = 0.08 # 1/Omega_SiC

[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1e4
    nstep = ${ntime}
  []
  [alpha]
    type = FullScalar
    batch_shape = '(${ntime})'
    value = 0.01
  []
[]

[Drivers]
  [driver]
    type = InfiltrationDriver
    model = 'model'
    prescribed_time = 'times'
    time = 'forces/t'
    prescribed_liquid_species_concentration = 'alpha'
    liquid_species_concentration = 'forces/alpha'
    ic_Scalar_names = 'state/phi_P state/phi_S state/alpha_P state/alpha_S'
    ic_Scalar_values = '0 0.3 0 0.05660377358'
    save_as = 'result.pt'
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
  []
[]

[Models]
  [liquid_volume_fraction]
    type = ScalarLinearCombination
    from_var = 'forces/alpha'
    to_var = 'state/phi_L'
    coefficients = '${omega_Si}'
  []
  [outer_radius]
    type = CylindricalChannelGeometry
    solid_fraction = 'state/phi_S'
    product_fraction = 'state/phi_P'
    inner_radius = 'state/ri'
    outer_radius = 'state/ro'
  []
  [liquid_reactivity]
    type = HermiteSmoothStep
    argument = 'state/phi_L'
    value = 'state/R_L'
    lower_bound = 0
    upper_bound = 0.1
  []
  [solid_reactivity]
    type = HermiteSmoothStep
    argument = 'state/phi_S'
    value = 'state/R_S'
    lower_bound = 0
    upper_bound = 0.1
  []
  [reaction_rate]
    type = DiffusionLimitedReaction
    diffusion_coefficient = '${D}'
    molar_volume = '${omega_Si}'
    product_inner_radius = 'state/ri'
    solid_inner_radius = 'state/ro'
    liquid_reactivity = 'state/R_L'
    solid_reactivity = 'state/R_S'
    reaction_rate = 'state/react'
  []
  [substance_product]
    type = ScalarLinearCombination
    from_var = 'state/phi_P'
    to_var = 'state/alpha_P'
    coefficients = '${oSiCm1}'
  []
  [product_rate]
    type = ScalarVariableRate
    variable = 'state/alpha_P'
    rate = 'state/adot_P'
    time = 'forces/t'
  []
  [substance_solid]
    type = ScalarLinearCombination
    from_var = 'state/phi_S'
    to_var = 'state/alpha_S'
    coefficients = '${oCm1}'
  []
  [solid_rate]
    type = ScalarVariableRate
    variable = 'state/alpha_S'
    rate = 'state/adot_S'
    time = 'forces/t'
  []
  ##############################################
  ### IVP
  ##############################################
  [residual_phiP]
    type = ScalarLinearCombination
    from_var = 'state/adot_P state/react'
    to_var = 'residual/phi_P'
    coefficients = '1.0 -1.0'
  []
  [residual_phiL]
    type = ScalarLinearCombination
    from_var = 'state/adot_P state/adot_S'
    to_var = 'residual/phi_S'
    coefficients = '1.0 ${chem_ratio}'
  []
  [model_residual]
    type = ComposedModel
    models = "residual_phiP residual_phiL
              liquid_volume_fraction outer_radius liquid_reactivity solid_reactivity
              reaction_rate substance_product product_rate substance_solid  solid_rate"
  []
  [model_update]
    type = ImplicitUpdate
    implicit_model = 'model_residual'
    solver = 'newton'
  []
  [substance_product_new]
    type = ScalarLinearCombination
    from_var = 'state/phi_P'
    to_var = 'state/alpha_P'
    coefficients = '${oSiCm1}'
  []
  [substance_solid_new]
    type = ScalarLinearCombination
    from_var = 'state/phi_S'
    to_var = 'state/alpha_S'
    coefficients = '${oCm1}'
  []
  [model]
    type = ComposedModel
    models = 'model_update liquid_volume_fraction substance_solid_new substance_product_new'
    additional_outputs = 'state/phi_P state/phi_S'
  []
[]
```
