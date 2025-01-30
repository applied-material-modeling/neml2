# Reactive Infiltration {#reactive-infiltration}

[TOC]

The reactive infiltration physics module is a collection of objects serving as building blocks for composing model's describing reactive infiltration kinects as a solid \f$(S)\f$ material is infiltrated by a liquid \f$(L)\f$ to create a product \f$(P)\f$. The framework and usage of the model is explained below, with both mathematical formulations and example input files.

## Framework

### Representative Volume Element

The infiltration process for a material considers a cylindrical Representative Volume Element (RVE) with radius \f$R\f$ and span \f$H\f$, depicting a solid capillary of radius \f$r_o\f$, corresponding to the initial porosity of \f$\varphi_0\f$. The mathematical description of the model uses cylindrical coordinates and assumes axisymmetry around \f$r = 0\f$. 

As the liquid enters the cylinder, it reacts with a solid to form a product with thickness \f$\delta_P\f$, as shown schematically in the [figure below](@ref rve). 

\anchor rve

![The (a) top and (b) cross-section view of the Representative Volume Element (RVE) of a given state of the reactive infiltration process, depicting the outer solid walls, the infiltrated liquid and the product from the chemical reaction ][rve]{html: width=80%}
[rve]: asset/lsi_rve_simplify.svg

Throughout this process, the solid radius, \f$r_o\f$, decreases while the product thickness, \f$\delta_P\f$, increases. Let \f$r_i\f$ denotes the innter radius of the product. In addition, define \f$\alpha_i\f$ (units of molar per volume) as the amount of substance in the RVE, and \f$\Omega_i = \dfrac{M_i}{\rho_i}\f$ as the molar volume of a material with molar mass \f$M_i\f$ (amu) and density \f$\rho_i\f$ (mass per volume), with subscripts taking \f$L\f$, \f$S\f$, and \f$P\f$, respectively.

The volume fraction, \f$\varphi_i\f$  of each material is then
\f[
    \varphi_i = \alpha_i \Omega_i
\f]
and the RVE porosity (void) is
\f[
    \varphi_v = 1 - \varphi_L - \varphi_P - \varphi_S  
\f] 

\anchor assumptions

<span style="color:red">Key assumptions</span> made throughout the derivation are:
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

### Governing Equations

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

### Implementation Details

The above governing equations are fairly general for a wide range of reactive infiltraion systems within this framework as long as the chemical reactions kinetics with the appropriate reaction coefficient (\f$k_i\f$), and the correct corresponding molar volume, \f$\Omega_i\f$, of the involving materials.

<span style="color:red">Example</span> for determining \f$k_i\f$ and \f$\Omega_i\f$:
- Considered a reactive infiltraion process of liquid (chemical formula \f$L_x\f$) and solid (\f$S_y\f$), with the molar mass \f$M_L\f$ and \f$M_S\f$.
- The chemical reactions create a product with chemical formula \f$L_m S_n\f$, with a reaction coefficient \f$k_S, k_P\f$.
\f[
    L_x + k_S S_y \rightarrow k_P L_mS_n 
\f]
For chemical balance, the reaction coefficient is:
\f{align*}
    &\ k_S = \dfrac{xn}{ym} \quad k_P = \dfrac{x}{m} 
\f} 
- The material density of the liquid, solid, product is \f$\rho_L, \rho_S, \rho_P\f$.
    
    Then, the molar volume is:
\f{align*}
    &\ \Omega_L = \dfrac{x M_L}{\rho_L} \quad \Omega_S = \dfrac{y M_S}{\rho_S} \quad \Omega_P = \dfrac{mn M_LM_S}{\rho_P}
\f}


The chemical equation also implies that
\f[
    \dot{\alpha}_S = \dfrac{k_P}{k_S} \dot{\alpha}_P 
\f]
> Since this consitutive model considered \f$\alpha_L\f$ as the forcing function for the IVP problem. \f$\dot{\alpha}_L \ne k_P \dot{\alpha}_P\f$.

The following tables summarize the relationship between the mathematical expressions and NEML2 models.

| Expression                                                                 | Model building block  | Syntax            |
| :------------------------------------------------------------------------- | :-------------------- | :---------------- |
| \f$ \bar{r}_i = \sqrt{1-\varphi_S-\varphi_P}; \quad \bar{r}_o = \sqrt{1-\varphi_S}\f$ | [Product and solid's inner radius](#ProductGeometry) | [ProductGeometry](#productgeometry) |
| \f$ \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S\f$ | [Product reaction rate](#DiffusionLimitedReaction) | [DiffusionLimitedReaction](#diffusionlimitedreaction) |
| \f$ R_L; \quad R_S \f$ | [Step Function](#HermiteSmoothStep) | [HermiteSmoothStep](#hermitesmoothstep) |
| \f$  \varphi_i = \alpha_i \Omega_i \f$ | --- | [Linear Combination](#scalarlinearcombination) |

| Residual Expression                                                        | Syntax            |
| :------------------------------------------------------------------------- | :---------------- |
| \f$ r_{\varphi_P} = \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n} - \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S\f$  | [Linear Combination](#scalarlinearcombination), [Variable Rate](#scalarvariablerate)|
| \f$r_{\varphi_S} = \dfrac{\alpha_{S, n+1} - \alpha_{S, n}}{t_{n+1}-t_n} + \dfrac{k_P}{k_S} \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n}\f$  | [Linear Combination](#scalarlinearcombination), [Variable Rate](#scalarvariablerate)|


## Model Building Blocks

The RVE keeps track of the volume fraction of the liquid, solid and product, \f$\varphi_L, \varphi_S, \varphi_P\f$. \f$\varphi_L\f$ directly corresponds to the prescribed liquid substance \f$\alpha_L\f$ where \f$\varphi_S, \varphi_P\f$ is obtained from solving the IVP problem with [Implicit Update](#implicitupdate) syntax model.

### Product and solid's inner radius {#ProductGeometry}
Model object [ProductGeometry](#productgeometry)

This modelcalculates the product and solid's inner radius, \f$r_i, r_o\f$. Note that, when there are presence of the product, the solid' inner radius is the outer radius of the product.
\f[
    \bar{r}_o = \sqrt{1-\varphi_S} \quad \bar{r}_i = \sqrt{1-\varphi_S-\varphi_P}    
\f]

Example input file that defines the product and solid's inner radius
```
[Models]
  [model]
    type = ProductGeometry
    solid_fraction = 'state/phi_s'
    product_fraction = 'state/phi_p'
    inner_radius = 'state/ri'
    outer_radius = 'state/ro'
  []
[]
```

### Product reaction rate {#DiffusionLimitedReaction}
Model object [DiffusionLimitedReaction](#diffusionlimitedreaction)

The reaction rates dictates the creation of the product. Within our assumptions, the growth of the product is controlled by the diffusion of the liquid through the product to react with the solid at the product-solid interface. Considered the same RVE as in [figure above](@ref rve), with \f$c_P\f$ (units of moles) as the concentration of the product in the RVE. Diffusion equation yields
\f[
    \dot{c}_P = \nabla \cdot D\nabla c_P
\f]
\f$D\f$ is the diffusion coeficient (untis of area per time) of liquid through the product. Integrating through the product's volume and applying Green's theorem,
\f[
    \dot{C} = D \dfrac{C_{liquid-product} - C_{product-solid}}{\delta_P}A_{enclosed}
\f]
Here,
\f{align*}
     &\ \dot{C} = \dot{\alpha}_P V_{RVE} \quad  C_{product-solid} = 0 \quad C_{liquid-product} = \dfrac{\dot{\alpha}_L V_{RVE}}{\varphi_L V_{RVE}} = \dfrac{1}{\Omega_L} \\
     &\ A_{enclosed} = 2\pi(r_o + r_i)H \quad \delta_P = r_o - r_i
\f}
The step function \f$R_L(\alpha_L)\f$, \f$R_S(\alpha_S)\f$ as reactivity is introduced to supressed the production rate when either the solid or the liquid is fully consumed. Finally, in normalized variables, and let the \f$R = l_c\f$ as the characterisitc length scales of the pores features. The product reaction rate is then,
\f[
    \dot{\alpha}_P = \dfrac{2}{\Omega_L}\dfrac{D}{l_c^2}\dfrac{\bar{r}_o+\bar{r}_i}{\bar{r}_o-\bar{r}_i} R_L R_S
\f]

Example input file that defines the product reaction rate
```
[Models]
  [model]
    type = DiffusionLimitedReaction
    diffusion_coefficient = 1e-3
    molar_volume = 1
    product_inner_radius = 'state/ri'
    solid_inner_radius = 'state/ro'
    liquid_reactivity = 'state/R_l'
    solid_reactivity = 'state/R_s'
    reaction_rate = 'state/alpha_rate'
  []
[]
```

### Hermite Smooth Step Function {#HermiteSmoothStep}
Model object [HermiteSmoothStep](#hermitesmoothstep)

A cubic Hermite interpolation after clambing is used as a smooth step function:
\f{align*}
    &0 & x_r \le 0 \\
    R(x) = & 3x_r^2 - 2x_r^3 & 0 \ge x_r \le 1 \\
    &0 & x_r \ge 1
\f}
where
\f[
    x_r = \dfrac{x-x_l}{x_u-x_l}
\f]
Here, \f$x_l,x_u\f$ denote the lower and upper bound of the smoothen transition regime.

> The Hermite Function is used in the reactive infiltration for the liquid and solid reactivity \f$R_i\f$

Example input file that use the smooth step function

```
[Models]
  [model]
    type = HermiteSmoothStep
    argument = 'state/foo'
    value = 'state/bar'
    lower_bound = '0'
    upper_bound = '0.05'
  []
[]
```

## Complete Input File
A complete example input file for the reactive infiltration is shown below with the appropriate model composition. The parameters are chosen to reflect a liquid silicon infiltration (LSI) on carbon foam.

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
    verbose = true
    show_input_axis = true
    show_output_axis = true
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
    verbose = true
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
    type = ProductGeometry
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