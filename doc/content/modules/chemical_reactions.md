(modules-chemical-reactions)=
# Chemical reactions

## Overview

The chemical-reactions module is a collection of objects that serve as
building blocks for composing models of macroscale chemical reactions —
pyrolysis, reactive infiltration, solidification, and similar
control-mass processes. The module provides empirical, homogenized
expressions for the **reaction rate** $\dot{\alpha}$ as a function of
the **conversion degree** $\alpha$, plus geometric/volumetric helpers
that bridge the reaction's control mass into a control-volume picture
other physics modules (solid mechanics, porous flow, phase-field
fracture) can consume.

The catalog carves the domain into two layers:

- **Reaction mechanisms** — typed maps $\alpha \mapsto \dot{\alpha}$
  for a single reacting phase. [](models-AvramiErofeevNucleation),
  [](models-ContractingGeometry), and [](models-DiffusionLimitedReaction)
  are the three concrete leaves; they share the canonical
  `conversion_degree` / `reaction_rate` variable names, so they can be
  swapped interchangeably in a composed model.
- **Geometry and volume bridges** — kinematic primitives that turn
  the reaction state into the volumetric quantities other modules need.
  [](models-CylindricalChannelGeometry) gives the dimensionless inner
  and outer radii of a cylindrical shrinking-core product;
  [](models-EffectiveVolume) sums per-component mass-fraction /
  density contributions into the total composite volume so the
  reaction's control mass can be tied into a control-volume framework.

Time integration, predictors, Arrhenius temperature dependence,
linear combinations, and the implicit Newton solve are provided by
NEML2's shared primitives. The example below shows how the
chemical-reactions types slot into that machinery.

## Math

A reaction mechanism is a constitutive relation between the conversion
degree $\alpha \in [0, 1]$ and the reaction rate
$\dot{\alpha} = f(\alpha)$. The three mechanisms in this module are

\begin{align}
\dot{\alpha} &= k \, (1 - \alpha)^n
  &&\text{(contracting geometry)} \\
\dot{\alpha} &= k \, (1 - \alpha) \, \bigl(-\ln(1 - \alpha)\bigr)^n
  &&\text{(Avrami--Erofeev nucleation)} \\
\dot{\alpha} &= \frac{2 \, D \, R_l \, R_s}{\omega}
              \, \frac{r_o}{r_o - r_i + \delta}
  &&\text{(diffusion-limited)}
\end{align}

where $k$ is the reaction coefficient (typically Arrhenius,
$k = k_0 \exp(-Q / R T)$), $n$ is the reaction order, $D$ is the
diffusion coefficient of the rate-limiting species through a product
layer of dimensionless inner / outer radii $(r_i, r_o)$,
$R_l, R_s \in [0, 1]$ are reactivities of the liquid and solid phases,
$\omega$ is the molar volume of the rate-limiting species, and
$\delta$ is a small "dummy thickness" that keeps the rate finite at
the start of the reaction.

For a cylindrical shrinking-core geometry parametrized by the volume
fractions $\phi_s$ (solid) and $\phi_p$ (product), the dimensionless
radii are

$$
r_i = \sqrt{1 - \phi_s - \phi_p}, \qquad r_o = \sqrt{1 - \phi_s}.
$$

To couple a control-mass reaction to a control-volume framework, the
total composite volume is reconstructed from per-component mass
fractions $\omega_i$ and densities $\rho_i$,

$$
V = \frac{M}{1 - \phi_o} \sum_i \frac{\omega_i}{\rho_i},
$$

where $M$ is the reference mass and $\phi_o \in [0, 1)$ is the open
volume fraction accounting for material leaving the control mass (for
example, gas escaping a pyrolyzing binder). When the input file omits
$\phi_o$, the prefactor reduces to $M$.

## Example: pyrolysis of a binder--particle composite

The pyrolysis regression test composes a contracting-geometry reaction
with an Arrhenius reaction coefficient, implicit integration of
$\alpha$, and explicit forward-Euler integration of the four mass /
volume fractions $(\omega_b, \omega_c, \omega_g, \varphi_o)$ tracking
binder, char, gas, and open pore.

```{literalinclude} ../../../tests/regression/chemical_reactions/pyrolysis/model.i
:language: ini
```

### Explanation

The reaction kinetics block — `reaction_coef`, `reaction_rate`,
`reaction_ode` composed as `reaction` — is the entire chemical-reactions
contribution:

- [](models-ArrheniusParameter) (`reaction_coef`) computes the
  temperature-dependent rate constant $k(T) = k_0 \exp(-Q/RT)$ from the
  `T` force.
- [](models-ContractingGeometry) (`reaction_rate`) maps the current
  $\alpha$ to $\dot{\alpha} = k (1 - \alpha)^n$ using that coefficient
  as a promoted parameter and the constant order $n$.
- [](models-ScalarBackwardEulerTimeIntegration) (`reaction_ode`) writes
  the residual $\alpha_{n+1} - \alpha_n - \Delta t \, \dot{\alpha} = 0$
  that the Newton solver consumes via the `eq_sys`
  [](models-ImplicitUpdate) block.

The remaining blocks downstream of `solve_reaction` are conservation
relations and time integration from NEML2's shared primitives:
the four [](models-ScalarLinearCombination) blocks (`binder_rate`,
`char_rate`, `gas_rate`, `open_pore_rate`) implement conservation
relations such as $\dot{\omega}_c = Y \, \dot{\alpha}$ (with $Y$ the
char yield) and $\dot{\omega}_g = -\mu (\dot{\omega}_b + \dot{\omega}_c)$
with $\mu$ the trapped-gas fraction, and four
[](models-ScalarForwardEulerTimeIntegration) blocks advance the mass /
volume fractions in time. The
[](models-ConstantExtrapolationPredictor) seeds the Newton iterate at
each step.

Variable names flow through the composition by string match: the
mechanism produces `alpha_rate`, which the linear combinations consume
and rename into `wb_rate`, `wc_rate`, `wg_rate`, `phio_rate`; the
time integrators then pair those rates with the state variables `wb`,
`wc`, `wg`, `phio`; the driver's initial conditions initialize `wb` and
`wc` explicitly, while `wg` and `phio` start from zero. The
final `model` [](models-ComposedModel) glues the implicit solve, the
rate computations, and the explicit integrations into one forward
operator that the [](drivers-TransientDriver) walks over the
prescribed `times` history. `additional_outputs = 'alpha'` keeps the
implicit unknown in the result file even though it is not consumed
downstream.

The companion reactive-infiltration regression scenario under
`tests/regression/chemical_reactions/reactive_infiltration/` follows
a similar pattern but swaps in
[](models-CylindricalChannelGeometry) +
[](models-DiffusionLimitedReaction), uses
[](models-HermiteSmoothStep) to give the liquid and solid
reactivities $R_l(\phi_L)$ and $R_s(\phi_S)$ a smooth on/off
behaviour, and Newton-solves on the volume fractions
$(\phi_P, \phi_S)$ instead of $\alpha$.

## See also

- [](tutorials-models-composition) — the canonical walkthrough of
  gluing primitives into a working forward operator with
  `ComposedModel`.
- [](modules-solid-mechanics) — the natural coupling partner: the
  effective volume from this module feeds volumetric eigenstrains and
  Jacobians on the mechanics side.
- [](modules-porous-flow) — the other typical coupling, where
  open-pore evolution from a pyrolysis or infiltration model drives
  the permeability / saturation update.
- [](syntax-catalog) — the full option list for every type linked
  above.
