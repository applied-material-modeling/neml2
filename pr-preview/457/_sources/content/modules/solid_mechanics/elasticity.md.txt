(modules-solid-mechanics-elasticity)=
# Elasticity

## Overview

Elasticity models describe the relationship between stress
$\boldsymbol{\sigma}$ and strain $\boldsymbol{\varepsilon}$ without
any history-dependent internal state. They are the smallest
self-contained block in almost every solid-mechanics composition —
a plasticity or viscoelasticity model produces an *elastic* strain
that an elasticity model then maps to stress.

NEML2 splits the catalog along two axes:

- **Stiffness vs. tensor form.** The "elasticity" leaves
  ([](models-LinearIsotropicElasticity), [](models-GeneralElasticity))
  evaluate the stress directly from a strain. The "elasticity tensor"
  leaves ([](models-IsotropicElasticityTensor),
  [](models-CubicElasticityTensor)) instead produce the fourth-order
  stiffness $\mathbb{C}$ itself, which other models consume.
- **Material symmetry.** Isotropic leaves take two independent
  constants drawn from the standard elastic constants ($E$, $\nu$,
  $K$, $G$, $\lambda$, $M$); which pairs each leaf accepts is
  documented in the per-type pages of the syntax catalog. The cubic
  leaf [](models-CubicElasticityTensor) takes three constants. The
  fully anisotropic leaf ([](models-GeneralElasticity)) accepts an
  arbitrary $\mathbb{C}$ and rotates it into the lab frame via a
  crystal orientation.

A standalone kinematics helper, [](models-GreenLagrangeStrain), is
shipped here as well: it converts a deformation gradient
$\boldsymbol{F}$ to the Green-Lagrange strain
$\boldsymbol{E} = \tfrac{1}{2}(\boldsymbol{F}^T\boldsymbol{F} - \boldsymbol{I})$,
the conjugate strain measure for finite-deformation elastic
formulations.

The strain-to-stress leaf [](models-LinearIsotropicElasticity)
carries two switches:

- `compliance` — when `true`, defines the inverse stress $\to$ strain
  map instead of the default strain $\to$ stress.
- `rate_form` — when `true`, suffixes the input/output variable names
  with `_rate` (the math is identical because the relation is linear
  in rate).

The tensor-form and general-anisotropic leaves don't expose these
switches — see the syntax catalog for their per-type option lists.

## Math

The general linear-elastic relation is

$$
  \boldsymbol{\sigma} = \mathbb{C} : \boldsymbol{\varepsilon},
$$

where $\mathbb{C}$ is the fourth-order elasticity tensor. The
catalog provides three specializations.

**Isotropic.** Two independent moduli (typically the bulk modulus
$K$ and shear modulus $G$) collapse $\mathbb{C}$ onto its volumetric
and deviatoric projectors:

$$
  \boldsymbol{\sigma}
  = 3K\,\operatorname{vol}\boldsymbol{\varepsilon}
  + 2G\,\operatorname{dev}\boldsymbol{\varepsilon}.
$$

Any pair drawn from $\{E,\,\nu,\,K,\,G,\,\lambda,\,M\}$ converts to
$(K,\,G)$ via the standard textbook relations; e.g. with
$E,\,\nu$ as inputs,

\begin{align}
  K &= \dfrac{E}{3(1 - 2\nu)}, \\
  G &= \dfrac{E}{2(1 + \nu)}.
\end{align}

**Cubic.** Three independent constants $(C_1, C_2, C_3)$ assemble the
fourth-order stiffness onto three cubic-symmetry projectors
$\mathbb{I}_{C_1},\,\mathbb{I}_{C_2},\,\mathbb{I}_{C_3}$:

$$
  \mathbb{C} = C_1\,\mathbb{I}_{C_1} + C_2\,\mathbb{I}_{C_2}
            + C_3\,\mathbb{I}_{C_3}.
$$

With $(G,\,E,\,\nu)$ as inputs, the canonical triple is

\begin{align}
  C_1 &= \dfrac{E\,(1 - \nu)}{(1 + \nu)(1 - 2\nu)}, \\
  C_2 &= \dfrac{E\,\nu}{(1 + \nu)(1 - 2\nu)}, \\
  C_3 &= 2G.
\end{align}

**General anisotropic.** A user-supplied lab-frame stiffness
$\boldsymbol{T}$ is rotated into the current configuration by the
crystal orientation $\boldsymbol{R}$ (an active rotation from the
reference frame), then contracted with the strain:

$$
  \boldsymbol{\sigma}
  = \bigl(\boldsymbol{R} \star \boldsymbol{T}\bigr)
    : \boldsymbol{\varepsilon},
$$

where $\boldsymbol{R} \star \boldsymbol{T}$ denotes the standard
fourth-order push-forward of $\boldsymbol{T}$ by $\boldsymbol{R}$.

The Green-Lagrange strain (used in finite-deformation formulations)
is the geometric pre-step

$$
  \boldsymbol{E} = \tfrac{1}{2}\bigl(\boldsymbol{F}^T \boldsymbol{F}
                                     - \boldsymbol{I}\bigr).
$$

## Example model composition

The smallest end-to-end elasticity input file uses
[](models-LinearIsotropicElasticity) directly, parameterized in
$(E,\,\nu)$:

```{literalinclude} ../../../../tests/models/solid_mechanics/elasticity/LinearIsotropicElasticity.i
:language: ini
:caption: tests/models/solid_mechanics/elasticity/LinearIsotropicElasticity.i
```

## Explanation

The `[Models]` block declares a single named object, `model`, of type
`LinearIsotropicElasticity`. Its option surface is two parallel
lists:

- `coefficient_types` lists the kind of each constant — for
  `LinearIsotropicElasticity` the supported pair is currently
  `(YOUNGS_MODULUS, POISSONS_RATIO)`. The full enumeration recognized
  by the option parser is documented in the syntax catalog at
  [](models-LinearIsotropicElasticity).
- `coefficients` lists the values, in the same order.

Here the pair `(100, 0.3)` is tagged `(YOUNGS_MODULUS, POISSONS_RATIO)`,
so the model derives $K$ and $G$ internally via the formulas above and
then evaluates the volumetric/deviatoric split.

The variable surface for `LinearIsotropicElasticity` is a single `SR2`
input named `strain` and a single `SR2` output named `stress`. Other models reach this elasticity block by
matching those names — a `[Models]` entry that produces an
`elastic_strain` and renames it to `strain` (or that renames this
block's input to `elastic_strain`) wires straight into the start of
the stress chain.

The `[Drivers]` block is a `ModelUnitTest`: it pins one input value
(a strain) and one expected output (the stress), then evaluates the
model and checks the result. The `[Tensors]` block
provides the literal `SR2` values referenced by the driver. The
driver is what makes the file runnable end-to-end with
`neml2-run input.i`, but everything you need to drop this elasticity
block into a larger material model lives inside `[Models]`.

If your material data is in a different parameterization, convert
externally to `(E, ν)` before plugging in. The companion
[](models-IsotropicElasticityTensor) (which outputs the SSR4
stiffness instead of the stress directly) currently also accepts
`(BULK_MODULUS, SHEAR_MODULUS)` if that pairing is more natural.
To define the inverse relation (stress $\to$ strain)
add `compliance = true`. To work in rate form (e.g. for a
[](models-ImplicitUpdate) residual that lives on stress and strain
*rates*) add `rate_form = true` — the input/output names then become
`strain_rate` and `stress_rate`.

The tensor-form leaves ([](models-IsotropicElasticityTensor),
[](models-CubicElasticityTensor)) take the same `coefficients` /
`coefficient_types` surface but output the fourth-order stiffness
itself as a single `SSR4` named after the HIT block. They're useful
when downstream models need the explicit stiffness — for example a
[](models-GeneralElasticity) that consumes
`elastic_stiffness_tensor` and an `orientation : MRP`, or any custom
composition that contracts $\mathbb{C}$ against something other than
a plain strain.

## See also

- [](tutorials-models-running-your-first-model) — the end-to-end
  "load this exact model and call it from Python" tutorial, also
  built around `LinearIsotropicElasticity`.
- [](tutorials-models-cross-referencing) — how the `strain` / `stress`
  names on an elasticity block get wired to other `[Models]` entries.
- [](tutorials-models-composition) — combining the elasticity leaves
  with strain decomposition, yield, and flow into a full constitutive
  chain via `ComposedModel`.
- [](models-LinearIsotropicElasticity), [](models-IsotropicElasticityTensor),
  [](models-CubicElasticityTensor), [](models-GeneralElasticity),
  [](models-GreenLagrangeStrain) — per-type option lists in the
  syntax catalog.
- [](modules-solid-mechanics-viscoelasticity),
  [](modules-solid-mechanics-plasticity) — sibling physics modules
  that consume an elasticity block as the first step of their stress
  chain.
