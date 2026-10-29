(modules-solid-mechanics-kinematics)=
# Kinematics

## Overview

The kinematics catalog supplies the small, composable primitives that
encode *non-mechanical* contributions to a material's deformation —
the strains or volumetric changes that arise from temperature
excursions, phase transformations, density changes during sintering,
or freezing/melting of a contained fluid. Pair them with an elastic
or inelastic constitutive model, which subtracts the non-mechanical
contribution from the total kinematic measure to recover the
mechanical part.

The catalog carves the space along two axes:

- **Output measure.** *Eigenstrain* primitives return a symmetric
  rank-2 tensor `eigenstrain : SR2` suitable for the small-strain
  additive split $\boldsymbol{\varepsilon} =
  \boldsymbol{\varepsilon}^{\text{mech}} + \boldsymbol{\varepsilon}^{*}$.
  *Deformation Jacobian* primitives return a `jacobian : Scalar`
  $J^{*}$ suitable for the finite-strain multiplicative split
  $\boldsymbol{F} = \boldsymbol{F}^{\text{mech}} \boldsymbol{F}^{*}$
  (combined with [](models-VolumeAdjustDeformationGradient) to lift
  the scalar Jacobian into a deformation gradient correction).
- **Driving quantity.** Temperature, volume change, phase fraction,
  or a coupled swelling/phase-change state.

For the small-strain workflow these eigenstrains are typically wired
into an [](models-SR2LinearCombination) that subtracts every
non-mechanical strain from the total strain before the elastic
constitutive call. For the finite-strain workflow the deformation
Jacobian primitives are multiplied together and then fed through
[](models-VolumeAdjustDeformationGradient) to produce the
volume-corrected deformation gradient.

## Math

### Eigenstrain (small-strain additive split)

For an isotropic eigenstrain driven by a scalar field $q$ (temperature,
volume, phase fraction), NEML2 adopts the standard cumulative form

$$
\boldsymbol{\varepsilon}^{*}(q) = \varepsilon^{*}_v(q)\,\boldsymbol{I},
$$

where $\varepsilon^{*}_v$ is the (linear) volumetric eigenstrain and
$\boldsymbol{I}$ is the rank-2 identity. The three concrete leaves
in the catalog differ only in how $\varepsilon^{*}_v$ depends on its
driver:

\begin{align}
\text{Thermal:}\quad &\varepsilon^{*}_v = \alpha\,(T - T_0), \\
\text{Volume change:}\quad &\varepsilon^{*}_v = \left(\dfrac{V}{V_0}\right)^{1/3} - 1, \\
\text{Phase transformation:}\quad &\varepsilon^{*}_v = \Delta V \, f.
\end{align}

Here $\alpha$ is the coefficient of thermal expansion, $T_0$ the
stress-free reference temperature, $V_0$ the reference volume,
$\Delta V$ the volume fraction change between phases A and B, and
$f \in [0,1]$ the current phase fraction. Each variant exposes the
same `eigenstrain` output name, which lets a downstream model treat
them interchangeably.

The mechanical (elastic) strain is then recovered by the additive
split

$$
\boldsymbol{\varepsilon}^{\text{el}} = \boldsymbol{\varepsilon} -
\boldsymbol{\varepsilon}^{p} - \boldsymbol{\varepsilon}^{*},
$$

with any plastic strain $\boldsymbol{\varepsilon}^{p}$ also subtracted
in inelastic flows.

### Deformation Jacobian (finite-strain multiplicative split)

In the finite-strain setting NEML2 represents the same physics through
a scalar deformation Jacobian $J^{*}$ — the determinant of the
volumetric correction to be removed from the total deformation
gradient:

\begin{align}
\text{Thermal:}\quad &J^{*} = 1 + \alpha\,(T - T_0), \\
\text{Swelling + phase change:}\quad &J^{*} = 1 + \alpha\,c\,\phi^{f}
+ (1-c)\,\phi^{f}\,\Delta\Omega.
\end{align}

For the coupled swelling/phase-change case, $\phi^{f}$ is the fluid
volume fraction that participates in the swelling, $c \in [0,1]$ is the
phase fraction (0 for fully solid, 1 for fully liquid), $\alpha$ is the
swelling coefficient, and $\Delta\Omega$ is the relative difference in
reference volume between the two phases.

Once the per-mechanism Jacobians are multiplied together into a single
$J$, [](models-VolumeAdjustDeformationGradient) lifts the scalar back
to a rank-2 correction on the deformation gradient,

$$
\boldsymbol{F}^{\text{mech}} = J^{-1/3}\,\boldsymbol{F},
$$

so that the mechanical part carries only the deviatoric (shape-changing)
piece while the volumetric piece $J$ is removed before the elastic
call.

## Example: thermal eigenstrain in a free-sintering model

The `free_sintering` regression scenario composes
[](models-ThermalEigenstrain) into a full GTN poroplastic
free-sintering model. The eigenstrain enters through the elastic-strain
combination, so the elastic constitutive call sees only the *mechanical*
strain after both plastic and thermal contributions are removed:

```{literalinclude} ../../../../tests/regression/solid_mechanics/viscoplasticity/free_sintering/model.i
:language: ini
:caption: tests/regression/solid_mechanics/viscoplasticity/free_sintering/model.i
```

## Explanation

The kinematic piece of the model is concentrated in two `[Models]`
blocks.

`[eigenstrain]` declares a [](models-ThermalEigenstrain) with
`reference_temperature = 300` and `CTE = 1e-6`. At every batch entry
its `temperature` input is sourced from the `temperature` force
prescribed by the driver, and it produces `eigenstrain : SR2` —
the cumulative thermal strain $\alpha(T - T_0)\,\boldsymbol{I}$
relative to the stress-free 300 K state.

`[elastic_strain]` is an [](models-SR2LinearCombination) that
implements the additive split

$$
\boldsymbol{\varepsilon}^{\text{el}} =
1\cdot\boldsymbol{\varepsilon} - 1\cdot\boldsymbol{\varepsilon}^{p}
- 1\cdot\boldsymbol{\varepsilon}^{*}_T,
$$

with `from = 'E plastic_strain eigenstrain'` and
`weights = '1 -1 -1'`. The output `elastic_strain` is then the strain
that the [](models-LinearIsotropicElasticity) block consumes — so by
the time stress is evaluated the thermal contribution has already
been peeled off. This is the typical wiring pattern for eigenstrains
in NEML2: declare the eigenstrain model, then subtract its
`eigenstrain` output from the total strain inside an
`SR2LinearCombination` whose result is consumed by elasticity.

To extend this composition to additional eigenstrain sources — say
adding a phase-transformation contribution via
[](models-PhaseTransformationEigenstrain) — declare the second
eigenstrain block, give its output a distinct variable name, and add
it to the `from` / `weights` lists of the same `SR2LinearCombination`.
No other part of the model needs to change because every eigenstrain
leaf in the catalog publishes the same `eigenstrain : SR2` surface.

For finite-strain workflows the analogous pattern uses
[](models-ThermalDeformationJacobian) or
[](models-SwellingAndPhaseChangeDeformationJacobian), multiplied
together via a `ScalarMultiplication` block, then handed
to [](models-VolumeAdjustDeformationGradient) to produce the
mechanical deformation gradient that the elastic model consumes.
On the small-strain side, [](models-VolumeChangeEigenstrain) plays
the corresponding role when a volume-change eigenstrain is needed.

:::{note}
All eigenstrain leaves are *cumulative* — they return the total
non-mechanical strain relative to the reference state, not an
increment. This is why subtraction at the elastic-strain step yields
the correct mechanical strain regardless of how the temperature or
phase history evolves through time.
:::

## See also

- [](tutorials-models-composition) — the underlying mechanics of
  wiring small `Model` pieces together into a full constitutive
  composition.
- [](modules-solid-mechanics-elasticity) — the elastic constitutive
  models that consume the mechanical strain produced by the
  eigenstrain subtraction shown above.
- [](modules-solid-mechanics-plasticity) — plastic-strain primitives
  combine with eigenstrains via the same `SR2LinearCombination`
  pattern.
- Syntax catalog entries for the individual types:
  [](models-ThermalEigenstrain),
  [](models-VolumeChangeEigenstrain),
  [](models-PhaseTransformationEigenstrain),
  [](models-ThermalDeformationJacobian),
  [](models-SwellingAndPhaseChangeDeformationJacobian),
  [](models-VolumeAdjustDeformationGradient).
- [](syntax-catalog) — the full per-type option reference.
