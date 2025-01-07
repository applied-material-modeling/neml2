# Liquid Infiltration {#liquid-infiltration}

[TOC]

The liquid infiltration physics module is a collection of objects serving as building blocks for composing models describing infiltration kinetics as a solid \f$(S)\f$ material is infiltrated by a liquid \f$(L)\f$ to create a product \f$(P)\f$. The framework and usage of the model is explained below, with both mathematical formulations and example input files.

\note
The particular liquid filtration model composed in NEML2 is derived based on a set of [assumptions](@ref assumptions). However, the building blocks are general enough in the hope that they can be swapped out for different chemical interactions in a variety of systems.

## Framework

### Representative Volume Element

The infiltration process for a material considers a cylindrical Representative Volume Element (RVE) with radius \f$R\f$ and height \f$H\f$, depicting a solid capillary of radius \f$r_0\f$, corresponding to the initial porosity of \f$\varphi_0\f$. The mathematical description of the model uses cylindrical coordinates and assumes axisymmetry around \f$r = 0\f$. A schematic illustration of the RVE is shown in the [figure below](@ref rve).

\anchor rve

![Fig. 1: RVE -- need to be updated][rve]{html: width=75%}
[rve]: asset/lsi_rve.png

<span style="color:red">Add some description of the figure, introduce geometrical quantities.</span>

The span of liquid \f$(L)\f$ and product \f$(P)\f$ are denoted by \f$h_L\f$ and \f$h_P\f$, respectively. In addition, define \f$\alpha_i\f$ as the amount of substance in the RVE and \f$\Omega_i\f$ as the molar volume (substance per volume), with subscripts taking \f$L\f$, \f$S\f$, and \f$P\f$, respectively.

\anchor assumptions

Key assumptions made throughout the derivation are:
-  Liquid remains liquid over the entire course of reaction, i.e., no phase change.
-  The reaction is irreversible.
-  Formation of the initial product layer is immediate once liquid comes into contact with solid.
-  Once product is formed, reaction rate is primarily limited by the diffusion of Liquid through product to the product-solid interface.
-  The product wall thickness remains uniform during the infiltration.
-  The liquid front inside a capillary wall is flat.
-  In the abundance of liquid, the height of the product layer follows the liquid front.

The following nondimensionalization is applied to the constitutive model for the reactive infiltration process.
\f{align*}
    &\ \bar{\delta}_P = \frac{\delta_P}{R}, \quad \bar{r}_0 = \dfrac{r_0}{R} = \sqrt{\varphi_0}, \quad \bar{r}_1 = \dfrac{r_1}{R} = \sqrt{\varphi_0} -M{\bar{\delta}_P^*}^2 \\
    &\ \bar{h}_P = \frac{h_P}{H}, \quad \bar{h}_L = \frac{h_L}{H}=\frac{\alpha_L\Omega_L}{\bar{r}_1^2}.
\f}
The complete state of the RVE is denoted by the tuple \f$\left( \alpha_L, \bar{\delta}_P^*, \bar{h}_P \right)\f$, where \f$\bar{\delta}_P^* = \sqrt{\bar{\delta}_P}\f$.

\note
Explain the positivity-preserving lifting transformation.

The effective porosity can be written in terms of the nondimensional variables as
\f[
    \varphi = \bar{r}_0^2 \left( 1 - \bar{h}_P \right) + \bar{r}_1^2 \left( \bar{h}_P - \bar{h}_L \right), \quad \bar{h}_L = \dfrac{\Omega_L \alpha_L}{\bar{r}_1^2}
\f]
and, the amount of solid can be expressed as
\f[
    \alpha_S = \frac{1}{\Omega_{S}}(1-\varphi - \alpha_P\Omega_P - \alpha_L\Omega_L)
\f]

### Constraints

The following constraints apply to the reaction constitutive model:
- The liquid flow path remains open, i.e.,
    \f[
        \bar{r}_1 \ge 0
    \f]
    When \f$\bar{r}_1=0\f$, no liquid can enter the RVE, i.e. \f$\dot{\alpha}_L^I=0\f$. Therefore, the simulation is halted.
- The reaction stops when there is no solid remaining in the RVE,
    \f[
        \bar{\delta}_P \le \min\left(\dfrac{1-\sqrt{\varphi_0}}{1-M },\dfrac{\sqrt{\varphi_0}}{M}\right)=\bar{\delta}_P^c.
    \f]
    Here, within our assumption that the product layer front remains uniform during the infiltration, product layer only grows in the \f$+z\f$ direction. The reaction equation becomes
    \f[
        \dot{\bar{\delta}}_P = 0
    \f]
- Since the product layer cannot span beyond the RVE, the outflow rate should satisfy
    \f{align*}
        \dot{\alpha}_L^O = \begin{cases}
            0, \text{ if } \bar{h}_L < 1, \\
            \dot{\alpha}_L^I, \text{ if } \bar{h}_L = 1.
        \end{cases}
    \f}
- <span style="color:red">Try to explain why \f$\bar{h}_p \leq 1\f$ is self-enforced.</span>
    <!-- \f[
        \bar{h}_P \le 1.
    \f]
    Since \f$\bar{h}_P\f$ is irreversible, \f$\bar{h}_P = 1\f$ corresponds to \f$\dot{\bar{h}}_P = 0\f$, which implies
    \f[
        \dot{\alpha}_L^O = \dot{\alpha}_L^I,
    \f]
    where \f$\dot{\alpha}_L^O\f$ represents the outflow rate of liquid. Note \f$\dot{\alpha}_L^O = 0\f$ for \f$h_L<1\f$. -->

### Governing Equations

When including the constraints, the initial-value problem (IVP) corresponding to the constitutive model is
\f[
    \mathbf{r} = \begin{Bmatrix}
        r_\delta \\
        r_h \\
        r_\alpha
    \end{Bmatrix} = \mathbf{0},
\f]
where \f$r_\delta\f$ is the residual corresponding to the product thickness growth, \f$r_h\f$ is the residual corresponding to the product span growth, and \f$r_\alpha\f$ defines the local mass balance for the liquid. The residuals are defined below. Note an implicit backward-Euler time integration scheme is used.

- Product thickness growth
    \f{align*}
        r_\delta &= \bar{\delta}_{P,n+1} - \bar{\delta}_{P,n} - \dot{\bar{\delta}}_P^* (t_{n+1} - t_n) = 0, \\
        \dot{\bar{\delta}}_P^* &= R G \left[1-H\left(\dfrac{\bar{\delta}_{P, n+1}}{\bar{\delta}_P^c}-1 \right)\right]
    \f}
    Here \f$H(x)\f$ is the Heaviside function, and \f$G\f$ denotes the thickeness growth of the product when liquid in abundant. When there is insufficient liquid, the product's thickness growth is reduced by a factor \f$R\f$.

- Product span growth:
    The growth of span in the \f$ z \f$ direction takes the form of the Fischer-Burmeister complementarity condition:
    \f[
        r_h = \phi + \dot{\bar{h}}_P - \sqrt{ \phi^2 + \dot{\bar{h}}_P^2 } = 0, \quad \phi = \bar{h}_{P, n+1} - \bar{h}_{L},
    \f]
    which is equivalent to the KKT conditions,
    \f[
        \phi \geq 0, \quad \dot{\bar{h}}_P \geq 0, \quad \phi \dot{\bar{h}}_P = 0,
    \f]
    where \f$\phi\f$ is the product span growth function.
- Local mass balance for the liquid
    \f{align*}
        r_\alpha &= \dfrac{\alpha_{L, n+1} - \alpha_{L, n}}{t_{n+1}-t_n} - \left( \dot{\alpha}_L^I + \dot{\alpha}_L^R +\dot{\alpha}_L^O \right) = 0, \\
        \dot{\alpha}_L^R &= -\nu \dfrac{\alpha_{P, n+1} - \alpha_{P, n}}{t_{n+1}-t_n}.
    \f}
    where \f$\dot{\alpha}_L^R\f$ is the liquid reaction rate, and \f$\nu\f$ is the stoichiometric coefficient of the reaction.

### Constitutive choices and implementational details

The above governing equations are fairly general for a wide range of liquid infiltration systems within this framework. To complete the definition of the constitutive model, two constitutive choices have to be made for \f$R\f$ and \f$G\f$. Here, we choose
\f{align*}
    G &= \dfrac{D}{l_c^2} \frac{\rho_L}{\rho_P} \frac{\bar{r}_1} {2{\bar{\delta}_P^*}^3\left[\sqrt{\varphi_0}+(1-2M){\bar{\delta}^*_P}^2\right]}, \\
    R &= \left<\dfrac{\bar{h}_L}{\bar{h}_P}\right>^p,
\f}
where \f$\left< \cdot \right>\f$ is the Macaulay bracket.

From geometric relations, we also have
\f[
    \alpha_P = \dfrac{\bar{h}_P}{\Omega_P} \left( 2\bar{r}_1 {\bar{\delta}_P^*}^2 + {\bar{\delta}_P^*}^4 \right).
\f]

With that, the constitutive model is completely defined. The following table summarizes the relationship between mathematical expressions and NEML2 models.
| Expression                                                                 | Model building block  | Syntax            |
| :------------------------------------------------------------------------- | :-------------------- | :---------------- |
| \f$\bar{r}_1 = \dfrac{r_1}{R} = \sqrt{\varphi_0} -M{\bar{\delta}_P^*}^2\f$ | [InletGap](#InletGap) | [link](#inletgap) |


## Model building blocks

These sub-models are built in order to solve the IVP - 3 equations with 3 unknowns \f$\left( \alpha_L, \bar{\delta}_P^*, \bar{h}_P \right)\f$, where \f$\bar{\delta}_P^* = \sqrt{\bar{\delta}_P}\f$.

### Size of the Inlet {#InletGap}
Model object: `InletGap`

This model calculates the inlet size of the RVE, following
\f[
    \bar{r}_1 = \sqrt{\varphi_0} -M{\bar{\delta}_P^*}^2
\f]

Example input file that defines the size of the inlet.
```
[Models]
    [model]
        type = InletGap
        product_thickness_growth_ratio = 0.576
        initial_porosity = 0.5
        product_thickness = 'state/delta'
        inlet_gap = 'state/r1'
    []
[]
```

### Constraint Functions
Model object: `SwitchingFunction`

A smooth heaviside function of \f$x\f$ of the form \f$g\left(\dfrac{x}{x_c} - x_0\right)\f$. The current implementation is Sigmoid,
\f[
    g = \frac{1}{2}+\frac{1}{2}\tanh{\left[n\left(\dfrac{x}{x_c}-x_0\right)\right]}
\f]

> For use in the liquid infiltration, make sure to use the \f$1-g\f$ by settting `one_subtract_condition = True`.

Example input file that defines the constraint functions.
```
[Models]
    [model]
        type = SwitchingFunction
        smooth_degree = 10.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        one_subtract_condition = true
        variable = 'state/var'
        switch_out = 'state/out'
    []
[]
```

### Product height growth with Liquid front
Model object: `ProductGrowthWithLiquid`

When there is sufficient Liquid, the span (height) of the Product and and Liquid is equal. In addition, the flat liquid front is assumed. Based on geometric relation, these imply the relationship through output variable \f$\phi\f$ as,
\f[
    \phi = \bar{r}_1^2 \bar{h}_P - \alpha_L\Omega_L
\f]

Example input file that defines the Product height growth with Liquid front.
```
[Models]
    [model]
        type = ProductGrowthWithLiquid
        liquid_molar_volume = 0.7
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        phi_condition = 'state/pcond'
    []
[]
```

### Fischer-Burmeister condition
Model object: `FischerBurmeister`

If \f$ a \ge 0, b \ge 0, ab = 0 \f$ then Fischer-Burmeister complementary condition is
 \f[
    a+b-\sqrt{a^2+b^2} = 0
\f]

> For use in liquid infiltration, condition_a is `ProductGrowthWithLiquid` and condition_b is `VariableRate` of a state variable corresponding to Product's height. The output of the model is one of the residual for the IVP.

Example input file that defines the Fischer-Burmeister condition.
```
[Models]
    [model]
        type = FischerBurmeister
        condition_a = 'state/a'
        condition_b = 'state/b'
        fischer_burmeister = 'state/fb'
    []
[]
```

### Maximum Product thickness
Model object: `ProductThicknessLimit`

The product's maximum thickness governs by the growth condition \f$M\f$ and the amount of Solid to react with, aka initial porosity, \f$\varphi\f$ (see Fig. 1). Based on geometric relation,
\f[
    \bar{\delta}_P^c = \min\left(\dfrac{1-\sqrt{\varphi_0}}{1-M },\dfrac{\sqrt{\varphi_0}}{M}\right).
\f]

Example input file that defines the maximum Product thickness.
```
[Models]
    [model]
        type = ProductThicknessLimit
        initial_porosity = 0.5
        product_thickness_growth_ratio = 'M'
        product_thickness = 'state/delta'
        limit_ratio = 'state/dratio'
    []
[]
```

### Ideal Product thickness growth
Model object: `LiquidProductDiffusion1D`

Assuming quasi steady state of Liquid infiltration, consider the same RVE as in Fig. 1 with a constant \f$h_P\f$, the mass \f$m_P\f$ of the product is,
\f[
    \begin{align}
        m_P = \rho_PV_P = \rho_P\pi ((r_1+\delta_P)^2-r_1^2)h_P \\
        \dot{m}_P = 2\rho_P\pi h_P(r_0+(1-2M)\delta_P)\dot{\delta}_P
    \end{align}
\f]

Product grows due to the diffusion between the Liquid-Product interface and Product-Solid interface, with a diffusion coefficient \f$D\f$.  Then,
\f[
    \dot{m}_P=D\nabla C \cdot \boldsymbol{n}
\f]

where \f$C\f$ is the mass concentration, \f$\boldsymbol{n}\f$ is the normal vector. Assuming Liquid reacts immediately with Solid at the Product-Solid interface, i.e. the concentration of Liquid (with molar mass \f$M_L\f$) at the Product-Solid interface is zero, then
\f[
    \dot{m}_P = 2D \pi r_1 h_P \frac{\rho_L}{\delta_P}
\f]
Combining, the Product thickness height growth, in abundance of Liquid, is
\f[
        \dot{\delta}_P = D \frac{\rho_L}{\rho_P} \frac{r_1}{\delta_P(r_0+(1-2M)\delta_P)}
\f]
Equivalently in normalized units,
\f[
    {\dot{\bar{\delta}}_P^*} = \dfrac{D}{l_c^2} \frac{\rho_L}{\rho_P} \frac{\bar{r}_1} {2{\bar{\delta}_P^*}^3\left[\sqrt{\varphi_0}+(1-2M){\bar{\delta}^*_P}^2\right]}
\f]

Example input file that defines the ideal Product thickness growth.
```
[Models]
    [model]
        type = LiquidProductDiffusion1D
        liquid_product_density_ratio = 0.8
        initial_porosity = 0.5
        product_thickness_growth_ratio = 0.57
        liquid_product_diffusion_coefficient = 1e-4
        representative_pores_size = 1.0
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        ideal_thickness_growth = 'state/delta_growth'
    []
[]
```

### Reduction Scaling for Product thickness growth when Liquid is deficient
Model object: `LiquidDeficientPowerScale`

When the Liquid in the RVE is insufficient to sustain the Product's thickness growth, the Product's thickness growth rate decreases from the ideal condition by a factor \f$R\f$ as
\f[
    R = \left\{\dfrac{\bar{h}_L}{\bar{h}_P}\right\}^p = \left\{\dfrac{\alpha_L \Omega_L}{\bar{r}_1^2 \bar{h}_P}\right\}^p
\f]
with \f$\{.\}\f$ denotes the Macaulay brackets to ensure \f$\left(\dfrac{\bar{h}_L}{\bar{h}_P}\right)^p\f$ has real values. Note that when Liquid is sufficient, aka \f$\bar{h}_L = \bar{h}_P\f$, then \f$R = 1\f$.

Example input file that defines the power scale reduction scaling for Product thickness growth when Liquid is deficient.
```
[Models]
    [model]
        type = LiquidDeficientPowerScale
        liquid_molar_volume = 1e-5
        power = 2.3
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        scale = 'state/scale'
    []
[]
```

### Effective Product thickness growth
Model object: `ProductThicknessGrowthRate`

The total effective combination of the product thickness growth rate, combining the `ideal_thickness_growth` (\f$R\f$) with `scale`\f$(S)\f$ to account for liquid deficiency and `switch`(\f$H\f$) to limit the growth rate by a critical value.
\f[
    {\dot{\bar{\delta}}_P^*} - R S H = 0
\f]

> The output of the model is one of the residual for the IVP. To use for liquid infiltration, make sure to have `switch` as an appropriate `ComposedModel` between `ProductThicknessLimit` and `SwitchingFunction`. Refer to the Complete Input File under Framework section for example.

Example input file that defines effective Product thickness growth.
```
[Models]
    [model]
        type = ProductThicknessGrowthRate
        thickness_rate = 'state/ddot'
        scale = 'state/scale'
        ideal_thickness_growth = 'state/dideal'
        switch = 'state/switch'
        residual_delta = 'residual/rdelta'
    []
[]
```

### Product geometric relation with Liquid
Model object: `ProductGeometricRelation`

Product's geometry dictates the amount of the Product in the RVE,
\f[
    \alpha_P = \dfrac{\bar{h}_P}{\Omega_P} \left( 2\bar{r}_1 {\bar{\delta}_P^*}^2 + {\bar{\delta}_P^*}^4 \right)
\f]

Example input file that defines the Product geometric relation with Liquid.
```
[Models]
    [model]
        type = ProductGeometricRelation
        product_molar_volume = 1e-5
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
[]
```

### Liquid total mass balance
Model object: `ChemMassBalance`

The total mass balance of the Liquid in the RVE, subjecting to inlet rate \f$\dot{\alpha}_L^I\f$, outlet rate \f$\dot{\alpha}_L^O\f$ and reaction rate \f$\dot{\alpha}_R^I\f$
\f[
    \dot{\alpha}_L - \dot{\alpha}_L^I - \dot{\alpha}_L^O + (-\dot{\alpha}_L^R) = 0
\f]
When \f$\bar{h}_P < 1\f$, \f$\dot{\alpha}_L^O = 0\f$. Therefore,
\f[
    \dot{\alpha}_L - S \dot{\alpha}_L^I + (-\dot{\alpha}_L^R) = 0
\f]
with `switch` \f$(S)\f$ can be written from Heaviside function as
\f[
    S = 1 - H\left( \bar{h}_P -1 \right)
\f]

> The output of the model is one of the residual for the IVP. To use for liquid infiltration, use `SwitchingFunction` model with appropriate input for `switch` \f$(S)\f$. Refer to the Complete Input File under Framework section for example.

Example input file that defines the Liquid total mass balance.
```
[Models]
    [model]
        type = ChemMassBalance
        in = 'state/in'
        switch = 'state/switch'
        minus_reaction = 'state/mreact'
        current = 'state/current'
        total = 'residual/total'
    []
[]
```


## Complete Input File

A complete example input file for the liquid infiltration is shown below with the appropriate model composition. The parameters are chosen to reflect a liquid silicon infiltration (LSI) process.

```
[Tensors]
    ############### Run condition ###############
    [times]
        type = ScalarFromTorchScript
        pytorch_pt_file = 'aLIndot.pt'
        tensor_name = 'time'
    []
    [aLInDot]
        type = ScalarFromTorchScript
        pytorch_pt_file = 'aLIndot.pt'
        tensor_name = 'data'
    []
    ############### Simulation parameters ###############
    [M]
        type = Scalar
        values = '0.1'
        batch_shape = '${nbatch}'
    []
    [phi0]
        type = Scalar
        values = '0.1'
        batch_shape = '${nbatch}'
    []
    [omega_L]
        type = Scalar
        values = 1.2e-5
        batch_shape = '${nbatch}'
    []
    [omega_P]
        type = Scalar
        values = 1.25e-5
        batch_shape = '${nbatch}'
    []
    [p]
        type = Scalar
        values = '0.25 0.5 1.0 2.0 5.0'
        batch_shape = '${nbatch}'
    []
    [lc]
        type = Scalar
        values = '2.236'
        batch_shape = '${nbatch}'
    []
[]

[Drivers]
    [driver]
      type = LiquidInfiltrationDriver
      model = 'model'
      prescribed_time = 'times'
      time = 'forces/tt'

      prescribed_liquid_inlet_rate = 'aLInDot'
      liquid_inlet_rate = 'forces/aLInDot'

      show_input_axis = true
      show_output_axis = true
      show_parameters = false
      ic_Scalar_names = 'state/alpha state/delta state/h state/alphaP'
      ic_Scalar_values = '1e-4 1e-2 1e-3 1e-3'
      save_as = 'test.pt'

      verbose = true
    []
    [regression]
      type = TransientRegression
      driver = 'driver'
      reference = 'gold/result.pt'
    []
  []

[Solvers]
    [newton]
        type = NewtonWithLineSearch
        linesearch_type = STRONG_WOLFE
        rel_tol = 1e-8
        abs_tol = 1e-10
        max_its = 100
        verbose = true
    []
[]

[Models]
    [inlet_gap]
        type = InletGap
        product_thickness_growth_ratio = 'M'
        initial_porosity = 'phi0'
        product_thickness = 'state/delta'
        inlet_gap = 'state/r1'
    []
    [product_growth]
        type = ProductGrowthWithLiquid
        liquid_molar_volume = 'omega_L'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        phi_condition = 'state/pcond'
    []
    [hdot]
        type = ScalarVariableRate
        variable = 'state/h'
        time = 'forces/tt'
        rate = 'state/hdot'
    []
    [fbcond]
        type = FischerBurmeister
        condition_a = 'state/pcond'
        condition_b = 'state/hdot'
        fischer_burmeister = 'residual/h'
    []
    ############### H RESIDUAL ###############
    [residual_h]
        type = ComposedModel
        models = 'fbcond inlet_gap product_growth hdot'
    []
    #############################################
    [product_geo]
        type = ProductGeometricRelation
        product_molar_volume = 'omega_P'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [alpha_transition]
        type = SwitchingFunction
        smooth_degree = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        one_subtract_condition = true
        variable = 'state/h'
        switch_out = 'state/alpha_transition'
    []
    [aR_dot]
        type = ScalarVariableRate
        variable = 'state/alphaP'
        time = 'forces/tt'
        rate = 'state/aRdot'
    []
    [alpha_dot]
        type = ScalarVariableRate
        variable = 'state/alpha'
        time = 'forces/tt'
        rate = 'state/alphadot'
    []
    [mass_balance]
        type = ChemMassBalance
        in = 'forces/aLInDot'
        switch = 'state/alpha_transition'
        minus_reaction = 'state/aRdot'
        current = 'state/alphadot'
        total = 'residual/alpha'
    []
    ############### ALPHA RESIDUAL ###############
    [residual_alpha]
        type = ComposedModel
        models = 'product_geo inlet_gap alpha_transition aR_dot alpha_dot mass_balance'
    []
    #############################################
    [deficient_scale]
        type = LiquidDeficientPowerScale
        liquid_molar_volume = 'omega_L'
        power = 'p'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        scale = 'state/def_scale'
    []
    [perfect_growth]
        type = LiquidProductDiffusion1D
        liquid_product_density_ratio = 0.8
        initial_porosity = 'phi0'
        product_thickness_growth_ratio = 'M'
        liquid_product_diffusion_coefficient = 5e-7
        representative_pores_size = 'lc'

        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        ideal_thickness_growth = 'state/delta_growth'
    []
    [delta_dcrit_ratio]
        type = ProductThicknessLimit
        initial_porosity = 'phi0'
        product_thickness_growth_ratio = 'M'
        product_thickness = 'state/delta'
        limit_ratio = 'state/dratio'
    []
    [delta_limit]
        type = SwitchingFunction
        smooth_degree = 100.0
        smooth_type = 'SIGMOID'
        scale = 1.0
        offset = 1.0
        one_subtract_condition = true
        variable = 'state/dratio'
        switch_out = 'state/dlimit'
    []
    [ddot]
        type = ScalarVariableRate
        variable = 'state/delta'
        time = 'forces/tt'
        rate = 'state/ddot'
    []
    [product_thickness_growth]
        type = ProductThicknessGrowthRate
        thickness_rate = 'state/ddot'
        scale = 'state/def_scale'
        ideal_thickness_growth = 'state/delta_growth'
        switch = 'state/dlimit'
        residual_delta = 'residual/delta'
    []
    ############### DELTA RESIDUAL ###############
    [residual_delta]
        type = ComposedModel
        models = 'deficient_scale inlet_gap perfect_growth delta_dcrit_ratio delta_limit ddot product_thickness_growth'
    []
    #############################################
    [model_residual]
        type = ComposedModel
        models = 'residual_h residual_alpha residual_delta'
        automatic_scaling = true
    []
    [model_update]
        type = ImplicitUpdate
        implicit_model = 'model_residual'
        solver = 'newton'
    []
    [aSiC_new]
        type = ProductGeometricRelation
        product_molar_volume = 'omega_P'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [model]
        type = ComposedModel
        models = 'model_update inlet_gap aSiC_new'
        additional_outputs = 'state/delta state/h'
    []
[]
```
