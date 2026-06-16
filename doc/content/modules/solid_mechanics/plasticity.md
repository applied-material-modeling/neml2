(modules-solid-mechanics-plasticity)=
# Plasticity

## Overview

Plasticity models describe the (typically irreversible and dissipative)
history-dependent deformation of solids. NEML2's macroscale plasticity catalog
under `neml2/models/solid_mechanics/plasticity/` does not ship a single
monolithic "plastic model"; instead, it provides the small composable
ingredients you wire together with the [](syntax-catalog) primitives in
`common/` to assemble a return-map step:

- **Stress measures** — `IsotropicMandelStress`, `MandelStress`, and
  [](models-SR2Invariant) (from `common/`) project the Cauchy/Mandel stress
  onto the effective stress used by the yield surface.
- **Yield surfaces** — [](models-YieldFunction) for classical
  $J_2$-style envelopes, [](models-GTNYieldFunction) and
  [](models-GursonCavitation) for porous metal plasticity.
- **Hardening models** — isotropic (`LinearIsotropicHardening`,
  `VoceIsotropicHardening`, `SlopeSaturationVoceIsotropicHardening`,
  `IsotropicHardeningStaticRecovery`,
  `PowerLawIsotropicHardeningStaticRecovery`), kinematic
  (`LinearKinematicHardening`, `ChabochePlasticHardening`,
  `FredrickArmstrongPlasticHardening`, `KinematicHardeningStaticRecovery`,
  `PowerLawKinematicHardeningStaticRecovery`), and rate-temperature couplings
  built from the Kocks-Mecking family (`KocksMeckingActivationEnergy`,
  `KocksMeckingFlowSwitch`, `KocksMeckingFlowViscosity`,
  `KocksMeckingIntercept`, `KocksMeckingRateSensitivity`,
  `KocksMeckingYieldStress`).
- **Flow directions** — [](models-Normality) takes the gradient of the yield
  function with respect to its variational arguments, or
  [](models-AssociativeJ2FlowDirection) provides a closed-form $J_2$ direction.
- **Flow rules and rates** — [](models-AssociativePlasticFlow) and the
  hardening-variable rates [](models-AssociativeIsotropicPlasticHardening) /
  [](models-AssociativeKinematicPlasticHardening) multiply the consistency
  parameter by the corresponding associative direction. For rate-dependent
  flow, [](models-PerzynaPlasticFlowRate) gives the consistency parameter
  itself as an explicit function of the yield function.

These pieces are glued together by [](models-ComposedModel) and closed by
either [](models-FBComplementarity) (consistent plasticity) or a Perzyna-style
rate equation (viscoplasticity), then wrapped in [](models-ImplicitUpdate) for
the implicit return map.

## Math

Plastic flow is governed by the Karush-Kuhn-Tucker (KKT) loading/unloading
conditions

$$
f^p \leq 0, \quad \dot{\gamma} \geq 0, \quad \dot{\gamma}\, f^p = 0,
$$

where $f^p$ is the yield function and $\gamma$ is the consistency parameter.

### Consistent (rate-independent) plasticity

NEML2 enforces the KKT conditions exactly (to machine precision) by recasting
them as the Fischer-Burmeister complementarity residual

$$
r = \dot{\gamma} - f^p - \sqrt{\dot{\gamma}^{\,2} + {f^p}^{\,2}},
$$

implemented by [](models-FBComplementarity) with `a_inequality = 'LE'`.

:::{note}
"Consistent" plasticity is sometimes called rate-*independent*, but that is a
misnomer — rate sensitivity can still be baked into the yield function or
hardening definitions in terms of internal-variable rates.
:::

### Viscoplasticity

The Perzyna regularization replaces the complementarity with an explicit
power-law overstress relation,

$$
\dot{\gamma} = \left( \frac{\langle f^p \rangle}{\eta} \right)^{n},
$$

where $\eta$ is the reference stress, $n$ is the rate exponent, and
$\langle \cdot \rangle$ denotes the Macaulay bracket. This is the role of
[](models-PerzynaPlasticFlowRate).

### Effective stress and yield function

For classical $J_2$ plasticity the effective stress is the von Mises norm of
the deviatoric (over-)stress,

\begin{align}
  \bar{\sigma} &= \sqrt{3 J_2},                                                \\
  J_2          &= \tfrac{1}{2}\,\mathrm{dev}\,\boldsymbol{\Xi}\,:\,
                  \mathrm{dev}\,\boldsymbol{\Xi},                              \\
  \boldsymbol{\Xi} &= \boldsymbol{\sigma} - \sum_i \boldsymbol{X}_i.
\end{align}

The yield function combines the effective stress with the yield strength
$\sigma_y$, isotropic hardening $k(\bar{\varepsilon}^p)$, and (implicitly,
through $\boldsymbol{\Xi}$) the kinematic back stresses $\boldsymbol{X}_i$:

$$
f^p = \bar{\sigma} - \sigma_y - k(\bar{\varepsilon}^p).
$$

### Associative flow

Associative flow rules derive the rates of the plastic strain and internal
variables from the maximum-dissipation principle,

\begin{align}
  \dot{\boldsymbol{\varepsilon}}^p &= \dot{\gamma}\,
        \frac{\partial f^p}{\partial \boldsymbol{\sigma}},                     \\
  \dot{\bar{\varepsilon}}^p        &= -\dot{\gamma}\,
        \frac{\partial f^p}{\partial k},                                       \\
  \dot{\boldsymbol{K}}^p           &= \dot{\gamma}\,
        \frac{\partial f^p}{\partial \boldsymbol{X}}.
\end{align}

[](models-Normality) computes these gradients by symbolically differentiating
a sub-model (the yield function) with respect to a list of input variables;
[](models-AssociativePlasticFlow),
[](models-AssociativeIsotropicPlasticHardening), and
[](models-AssociativeKinematicPlasticHardening) then scale the directions by
$\dot{\gamma}$.

## Example: rate-independent $J_2$ plasticity with mixed hardening

The input file below assembles a fully implicit return-map for elastic-plastic
behaviour with both isotropic and kinematic hardening: linear-elastic
stress-strain, [](models-VoceIsotropicHardening) isotropic hardening,
[](models-LinearKinematicHardening) back stress, associative $J_2$ flow, and
the Fischer-Burmeister consistency condition.

```{literalinclude} ../../../../tests/regression/solid_mechanics/rate_independent_plasticity/isokinharden/model.i
:language: ini
:caption: tests/regression/solid_mechanics/rate_independent_plasticity/isokinharden/model.i
```

### How the pieces wire together

- **Hardening laws** — `[isoharden]` ([](models-VoceIsotropicHardening)) maps
  `equivalent_plastic_strain` to `isotropic_hardening`; `[kinharden]`
  ([](models-LinearKinematicHardening)) maps `kinematic_plastic_strain` to
  the back-stress contribution `X`.
- **Stress side** — `[elastic_strain]` ([](models-SR2LinearCombination))
  subtracts the plastic strain from the total strain, `[elasticity]`
  ([](models-LinearIsotropicElasticity)) gives the Cauchy stress, and
  `[mandel_stress]` ([](models-IsotropicMandelStress)) lifts it to the
  work-conjugate Mandel stress consumed by the yield surface.
- **Yield surface** — `[overstress]` builds
  $\boldsymbol{\Xi} = \boldsymbol{\sigma} - \boldsymbol{X}$ as `O`,
  `[vonmises]` ([](models-SR2Invariant)) produces the effective stress, and
  `[yield_surface]` ([](models-YieldFunction)) assembles $f^p$ from the effective
  stress, yield strength, and isotropic hardening.
- **Flow** — `[flow]` is a [](models-ComposedModel) glueing the overstress,
  invariant, and yield steps so that [](models-Normality) can differentiate
  the resulting `yield_function` with respect to `mandel_stress`, `X`, and
  `isotropic_hardening`, producing the three associative directions.
  `[Eprate]`, `[Kprate]`, `[eprate]` scale each direction by the consistency
  parameter `flow_rate`.
- **Time integration** — three `BackwardEulerTimeIntegration` blocks
  ([](models-SR2BackwardEulerTimeIntegration) and its `Scalar` sibling) turn
  the rates into the residuals `plastic_strain_residual`,
  `kinematic_plastic_strain_residual`,
  `equivalent_plastic_strain_residual`.
- **Consistency** — `[consistency]` ([](models-FBComplementarity)) closes
  the system with the Fischer-Burmeister residual between `yield_function`
  and `flow_rate`.
- **Return map** — `[surface]` composes every residual-producing piece into
  one model. `NonlinearSystem` solves for
  `plastic_strain, kinematic_plastic_strain, equivalent_plastic_strain,
  flow_rate` using `Newton` + `DenseLU`. `[return_map]`
  ([](models-ImplicitUpdate)) wraps the equation system, warm-started by
  `[predictor]` ([](models-ConstantExtrapolationPredictor)). The top-level
  `[model]` re-evaluates `elastic_strain` and `elasticity` with the converged
  plastic strain to expose the final stress.

### Switching to viscoplasticity

Replacing `[consistency]` with a direct evaluation of the consistency
parameter turns this exact composition pattern into a Perzyna viscoplastic
model. The reference recipe lives at
`tests/regression/solid_mechanics/viscoplasticity/perfect/model.i`; the
structural changes are dropping [](models-FBComplementarity), adding
[](models-PerzynaPlasticFlowRate) as the `[flow_rate]` block, and integrating
the *stress* (rate form) rather than the plastic strain.

### Variations shipped in the test tree

The `tests/regression/solid_mechanics/` tree carries ready-to-read
compositions for the most common dialects:

- `rate_independent_plasticity/perfect/` — perfectly plastic, no hardening.
- `rate_independent_plasticity/isoharden/` — isotropic hardening only.
- `rate_independent_plasticity/kinharden/` — kinematic hardening only.
- `rate_independent_plasticity/isokinharden/` — combined (used above).
- `rate_independent_plasticity/radial_return/` — closed-form $J_2$
  radial-return via [](models-LinearIsotropicElasticJ2TrialStressUpdate).
- `rate_independent_plasticity/gurson/` — porous metal plasticity using
  [](models-GTNYieldFunction) and [](models-GursonCavitation).
- `viscoplasticity/perfect/` — Perzyna viscoplastic counterpart.
- `recovery/` — adds isotropic/kinematic static recovery
  ([](models-PowerLawIsotropicHardeningStaticRecovery),
  [](models-PowerLawKinematicHardeningStaticRecovery), and the power-law variants).
- `km_flow/` — Kocks-Mecking temperature- and rate-dependent flow.

## See also

- [](tutorials-models-implicit-model) — walks through `ImplicitUpdate`,
  `NonlinearSystem`, and the predictor with a minimal plasticity example.
- [](tutorials-models-composition) — the general pattern of gluing small
  models with [](models-ComposedModel) used pervasively above.
- [](modules-solid-mechanics-elasticity) — the stress-strain primitives that
  every plasticity composition starts from.
- [](modules-solid-mechanics-crystal-plasticity) — single- and poly-crystal
  plasticity built on the same composition primitives.
- [](modules-solid-mechanics-viscoelasticity) — sibling family for rate-
  dependent but unyielded behaviour.
- [](syntax-catalog) — autogenerated option reference for every type linked
  above.
