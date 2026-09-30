(modules-solid-mechanics-traction-separation-law)=
# Traction–separation laws

## Overview

Cohesive zone models — also known as traction–separation laws (TSLs) —
describe the constitutive response of an interface as a relation
between the displacement jump
$\boldsymbol{\delta} = [\delta_n, \delta_{s1}, \delta_{s2}]$ and the
traction $\boldsymbol{T} = [T_n, T_{s1}, T_{s2}]$ in the local
interface frame. The first component is the normal jump
(opening / closing); the other two are the in-plane shear jumps. Most
TSLs are damage-driven: a scalar damage variable accumulates as the
effective separation grows, and the traction degrades from an initial
elastic response toward zero at full debonding.

NEML2 provides a catalog of small composable primitives that you
assemble into a TSL via a [](models-ComposedModel), so a custom
variant (a different damage formula with the bilinear envelope, the
Camanho–Davila initiation with a new propagation criterion) is
usually an input-file change rather than a new class. A genuinely
new building block — a damage formula not on the list, a new
initiation criterion — still wants a small `Model` subclass,
following the same pattern as the existing primitives.

The catalog is organized by role:

**Kinematic helpers** (general-purpose primitives reused across modules):
- [](models-VecComponents) — split the displacement-jump `Vec` into
  three named `Scalar` components.
- [](models-MacaulaySplit) — split the normal jump into its positive
  and negative parts $\langle \delta_n \rangle_\pm$.
- [](models-ScalarPNorm) — weighted $p$-norm of an arbitrary number
  of `Scalar` inputs,
  $y = \left(\sum_i w_i |x_i|^p + \varepsilon\right)^{1/p}$. With
  $p = 2$ and unit weights this is the Euclidean norm, used for the
  tangential magnitude $\delta_s = \sqrt{\delta_{s1}^2 + \delta_{s2}^2}$
  and the bilinear effective separation
  $\delta_m = \sqrt{(\delta_n^+)^2 + \delta_s^2}$.
- [](models-IrreversibleScalar) — monotonically-increasing ratchet
  $y = \max(y_{n-1}, x)$. Use it to enforce damage irreversibility or
  any history-max state.

**TSL-specific kinematics**:
- [](models-ModeMixity) — $\beta = \delta_s / \delta_n^+$ in the
  opening branch; $\beta = 0$ in compression.

**Constitutive formulas**:
- [](models-CamanhoDavilaCriticalSeparation) — Camanho–Davila
  mixed-mode critical (damage-onset) separation $\delta_c(\beta)$.
- [](models-BenzeggaghKenaneFullSeparation) — mixed-mode full
  (failure) separation under the Benzeggagh–Kenane criterion,
  $\delta_f(\beta, \delta_c)$.
- [](models-PowerLawFullSeparation) — mixed-mode full (failure)
  separation under the Alfano–Crisfield power-law criterion.

**Damage-envelope primitives** (each emits `trial_damage`):
- [](models-BilinearDamage) evaluates the bilinear envelope from the
  effective, critical, and full separations.
- [](models-SalehaniIraniDamage) evaluates the coupled exponential
  envelope from the three separation components.

**Damage-update primitives** consume `trial_damage`, previous-step
`damage~1`, and the `time` / `time~1` pair (named `t` and `t~1` by
default), then emit irreversible `damage`. Both accept a non-negative
`viscosity` $\eta_v$, defaulting to zero:
- [](models-BackwardEulerViscousDamage) uses
  $\alpha = \Delta t / (\eta_v + \Delta t)$.
- [](models-ExponentialViscousDamage) uses
  $\alpha = 1 - \exp(-\Delta t / \eta_v)$.

Both updates have the form

$$
d_n = d_{n-1} + \alpha(d^* - d_{n-1}), \qquad
 d^* = \max(d_{\mathrm{trial}}, d_{n-1}).
$$

The backward-Euler model matches the MOOSE update. The exponential
model analytically integrates the same evolution equation
$\dot d=(d^*-d)/\eta_v$ while holding $d^*$ constant over the step.
Positive viscosity delays damage growth, while `viscosity = 0`
recovers the irreversible rate-independent update exactly. A negative
viscosity is rejected when the updater is built.

**Traction-assembly primitives** consume the updated `damage` and emit
the public `traction` `Vec`:
- [](models-OrthotropicLinearTraction) provides orthotropic linear
  elasticity with no damage state.
- [](models-BilinearTraction) assembles the Camanho/Davila-style
  bilinear traction.
- [](models-SalehaniIraniTraction) assembles the 3D coupled
  Salehani-Irani traction.

The three responsibilities are separate `Model` leaves, so an envelope,
damage integration method, or traction assembly can be replaced without
changing the other two. The traction leaves optionally accept a
`normal_penetration` channel and penalty stiffness to resist
interpenetration.

## Math

A bilinear cohesive law in the spirit of Camanho & Dávila uses an
effective scalar separation

$$
\delta_m = \sqrt{\langle \delta_n \rangle_+^2 + \delta_s^2}, \qquad
\delta_s = \sqrt{\delta_{s1}^2 + \delta_{s2}^2},
$$

a mode-mixity ratio

$$
\beta = \frac{\delta_s}{\langle \delta_n \rangle_+},
$$

a mixed-mode damage-onset (critical) separation $\delta_c(\beta)$, and
a mixed-mode failure separation $\delta_f(\beta, \delta_c)$. Damage is
then defined piecewise from the historical maximum effective
separation $\bar{\delta}_m = \max_{t' \le t} \delta_m(t')$,

\begin{align}
d(\bar{\delta}_m) &= 0,
  & \bar{\delta}_m &\le \delta_c, \\
d(\bar{\delta}_m) &=
  \frac{\delta_f \, (\bar{\delta}_m - \delta_c)}
       {\bar{\delta}_m \, (\delta_f - \delta_c)},
  & \delta_c < \bar{\delta}_m &< \delta_f, \\
d(\bar{\delta}_m) &= 1,
  & \bar{\delta}_m &\ge \delta_f,
\end{align}

so that the traction is

\begin{align}
T_n   &= K\,(1-d)\,\langle \delta_n \rangle_+ \;+\; K\,\langle \delta_n \rangle_-, \notag \\
T_{si} &= K\,(1-d)\,\delta_{si}, \quad i \in \{1, 2\},
\end{align}

where $K$ is the penalty stiffness. The Macaulay split on $\delta_n$
keeps the compressive branch elastic so the interface resists
interpenetration without accruing damage. The propagation criterion
enters through $\delta_f$: under the Benzeggagh–Kenane criterion,

$$
\delta_f(\beta, \delta_c) =
  \frac{2}{K\,\delta_c}
  \left[
    G_{Ic} + (G_{IIc} - G_{Ic})
    \left( \frac{\beta^2}{1 + \beta^2} \right)^{\eta}
  \right],
$$

while [](models-PowerLawFullSeparation) implements the Alfano–Crisfield
power-law form. The [](models-SalehaniIraniDamage) envelope replaces
the bilinear expression with a coupled exponential: the normal
direction enters linearly and the two tangential directions enter
quadratically in a single coupling exponent. Its updated damage is
then consumed by [](models-SalehaniIraniTraction).

## Example model composition

The input below assembles the Camanho–Davila bilinear law (BK
criterion) from the primitives above, drives it through a monotonic
mixed-mode opening ramp, and pins the result against a reference:

```{literalinclude} ../../../../tests/regression/solid_mechanics/traction_separation_law/bilinear_monotonic/model.i
:language: ini
```

## Explanation

Reading the `[Models]` block top to bottom:

1. **`decompose`** ([](models-VecComponents)) splits the prescribed
   `separation` `Vec` into the three scalar channels
   `signed_normal_separation`, `tangential_separation_1`, and
   `tangential_separation_2`. The two tangential channels are then
   collapsed to a scalar magnitude `tangential_separation` by
   **`tangential_separation`** ([](models-ScalarPNorm) with
   $p = 2$), giving $\delta_s$ from the math above.
2. **`macaulay_n`** ([](models-MacaulaySplit)) takes the signed normal
   jump and emits its positive part as `normal_separation`
   ($\langle \delta_n \rangle_+$) and its negative part as
   `normal_penetration`
   ($\langle \delta_n \rangle_- = \delta_n - \langle \delta_n \rangle_+$,
   i.e. the signed-negative branch that vanishes whenever
   $\delta_n \ge 0$). The positive part drives damage and softening;
   the negative part is handed to the traction assembly as the
   elastic penetration channel that keeps compression damage-free.
3. **`mode_mixity`** ([](models-ModeMixity)) computes
   $\beta = \delta_s / \langle \delta_n \rangle_+$.
4. **`critical_separation`**
   ([](models-CamanhoDavilaCriticalSeparation)) evaluates
   $\delta_c(\beta)$ from the penalty stiffness $K$ and the per-mode
   strengths.
5. **`full_separation`** ([](models-BenzeggaghKenaneFullSeparation))
   evaluates $\delta_f(\beta, \delta_c)$ from the per-mode fracture
   toughnesses and the BK exponent $\eta$. To swap to the
   Alfano–Crisfield criterion, replace this block with
   [](models-PowerLawFullSeparation) — every other sub-model stays
   the same.
6. **`effective_separation`** assembles
   $\delta_m = \sqrt{(\delta_n^+)^2 + \delta_s^2}$ with another
   [](models-ScalarPNorm).
7. **`damage_envelope`** ([](models-BilinearDamage)) consumes
   $\delta_m$, $\delta_c$, and $\delta_f$ and emits the bilinear
   `trial_damage`.
8. **`damage_update`** ([](models-BackwardEulerViscousDamage)) caps the
   trial value against `damage~1`, applies the selected viscous update,
   and emits `damage`. The composed model exposes it through
   `additional_outputs = 'damage'` for inspection.
9. **`traction`** ([](models-BilinearTraction)) consumes the updated
   damage, per-component jumps, and elastic penetration channel and
   assembles the traction vector.

The `[Drivers]` block prescribes the time grid and the displacement
jump history through a `TransientDriver`, and a `TransientRegression`
pins the run against `gold/result.pt`.

:::{note}
The bilinear and Salehani-Irani graphs carry irreversible state through
their damage-update leaf. Under load–unload–reload schedules the
unloading limb returns elastically along the secant of the current
damaged stiffness, and reloading resumes the softening branch only
after the historical peak separation is exceeded. The `bilinear_unload`
scenario under
`tests/regression/solid_mechanics/traction_separation_law/` exercises
this behavior.
:::

## Grain-boundary cavitation (Sham–Needleman)

### Overview

The laws above degrade an interface through a scalar damage variable
driven by the separation history. Creep rupture at a grain boundary
works the other way round: the interface carries a *physical*
population of cavities whose area fraction $f$ and number density $N$
evolve in time under the traction, and the separation is a consequence
of that population rather than its cause. The simplified Sham–Needleman
model in NEML2 is this second kind of law.

The primitives are deliberately small so the model can be assembled in
three forms — with nucleation, without it, and with the
nucleation-driving strain rate supplied from outside — by changing
which leaves appear in the `[Models]` block, not by toggling options:

**Cavity-state kinematics**:
- [](models-EffectiveCavityFraction) — the GTN-style coalescence map
  $f \mapsto f^*$ that accelerates the softening functions once the
  cavities start to link up.
- [](models-CavityVolumeOpening) — the viscous normal opening
  $[\![u]\!]_n^v = NV - N_0V_0$, i.e. the smeared cavity volume.

**Rates**:
- [](models-DiffusiveCavityGrowthRate) — Hull–Rimmer diffusive growth
  of a single cavity's volume, $\dot V$.
- [](models-CavityGrowthDamageRate) — the geometric conversion from
  $\dot V$ to an area-fraction rate.
- [](models-CavityNucleationRate) — stress-driven nucleation of new
  cavities, $\dot N$.
- [](models-CavityNucleationDamageRate) — the area-fraction rate the
  newly nucleated cavities contribute.
- [](models-GrainBoundarySlidingRate) — viscous sliding of the
  boundary, softened by the cavitated fraction.
- [](models-SlidingDrivenStrainRate) — the local strain rate that
  sliding implies, used as the nucleation driver when no external one
  is available.

**Traction assembly**:
- [](models-ShamNeedlemanTraction) — linear springs in *series* with
  the cavitation and sliding, with a compressive contact penalty and a
  separated branch past $f_F$.

### Math

A cavity of radius $a$ on a boundary with number density $N$ occupies

$$
f = \pi a^2 N, \qquad V = \tfrac{4}{3}\pi a^3 h(\psi),
$$

where $h(\psi)$ is the cavity shape factor ($h \approx 0.61$ at a tip
angle of 75°). Eliminating $a$ gives both the smeared opening and the
geometric conversion used below,

$$
NV = \frac{4h}{3\sqrt{\pi}} \frac{f^{3/2}}{N^{1/2}}, \qquad
\dot{f}_\text{growth} = \frac{\sqrt{\pi}\,N^{3/2}}{2h\sqrt{f}}\,\dot{V}.
$$

Cavities soften the boundary through the *effective* fraction $f^*$,
which equals $f$ until the coalescence threshold $f_c$ and then rises
on a steeper line to $f_u$ at final separation $f_F$,

$$
f^* = \begin{cases}
  f, & f \le f_c, \\[2pt]
  f_c + \dfrac{f_u - f_c}{f_F - f_c}\,(f - f_c), & f_c < f < f_F, \\[2pt]
  f_u, & f \ge f_F.
\end{cases}
$$

The map saturates past $f_F$ rather than extrapolating. Keeping the
accelerated line going would drive $f^*$ through one at
$f = f_c + (1 - f_c)(f_F - f_c)/(f_u - f_c)$ — a physical fraction
below one, so an ordinary Newton iterate reaches it — and both
softening denominators, $q(f^*)$ and $1 - f^*$, vanish there together.
Saturating bounds every consumer of $f^*$ by $f_u < 1$ for any $f$, so
choosing $f_u$ strictly below one is what keeps the rates finite.

Only the softening functions see $f^*$; the geometry above keeps the
physical $f$, so the kinematic identity $[\![u]\!]_n^v = NV - N_0V_0$
holds exactly whatever the coalescence map does.

Cavity volume grows by grain-boundary diffusion (Hull–Rimmer),

$$
\dot{V} = \frac{8\pi\mathcal{D}\,\langle T_n \rangle}{q(f^*)}, \qquad
q(f^*) = -2\ln f^* - (3 - f^*)(1 - f^*),
$$

which combines with the geometric conversion into the Sham–Needleman
growth term

$$
\dot{f}_\text{growth}
  = \frac{4\pi^{3/2}\mathcal{D}\,N^{3/2}\,\langle T_n \rangle}
         {h\sqrt{f}\,q(f^*)}.
$$

Keeping $\dot V$ and the conversion in separate models is what lets a
second growth mechanism be summed onto $\dot V$ later without
restating the geometry.

New cavities nucleate at a rate set by the normal traction and a
driving strain rate, and they arrive at the current mean size:

$$
\dot{N} = F_N \left\langle \frac{T_n}{\Sigma_0} \right\rangle^{\beta}
          \dot{\varepsilon}, \qquad
\dot{f}_\text{nucleation} = \frac{f}{N}\,\dot{N}, \qquad
\dot{f} = \dot{f}_\text{growth} + \dot{f}_\text{nucleation}.
$$

The boundary also slides viscously, softened by the cavitated
ligament, and that sliding supplies $\dot\varepsilon$ when no external
strain rate is prescribed:

$$
\dot{[\![u]\!]}_t^v = \frac{T_t}{(1 - f^*)\,\eta}, \qquad
\dot{\varepsilon} = \frac{\left|\dot{[\![u]\!]}_t^v\right|}{d}.
$$

Finally, each jump splits additively into an elastic and a viscous
part, and the spring acts on the elastic part only — the springs are
in **series** with the dashpots. A parallel spring would carry load
forever and the interface would never separate completely. Writing
$[\![u]\!]_n^v$ for the smeared cavity volume above,

\begin{align}
T_n &= K_n\left([\![u]\!]_n - [\![u]\!]_n^v\right),
  & [\![u]\!]_n &\ge 0,\ f < f_F, \notag \\
T_n &= K_p\,[\![u]\!]_n,
  & [\![u]\!]_n &< 0, \notag \\
T_t &= K_t\left([\![u]\!]_t - [\![u]\!]_t^v\right),
  & f &< f_F,
\end{align}

and past final separation, $f \ge f_F$, the interface carries
$T_n = K_p \min([\![u]\!]_n, 0)$ and $T_t = 0$ — the compressive
penalty survives separation, the tensile and shear load paths do not.

:::{note}
Freezing $f$, $N$ and $[\![u]\!]_t^v$ after separation needs no
clamping. Every rate above is driven by $T_n$ or $T_t$, and on the
separated branch $T_t = 0$ while $T_n \le 0$ — so the sliding rate,
both factors of the nucleation rate, and the Macaulay-bracketed growth
rate all vanish identically and the state holds itself.
:::

:::{warning}
The separation test $f < f_F$ in [](models-ShamNeedlemanTraction) must
read the **old** (converged) damage state, which in an implicit
composition means wiring it explicitly:

```
[traction]
  type = ShamNeedlemanTraction
  cavity_fraction = 'cavity_fraction~1'
  ...
[]
```

Testing the trial unknown instead puts a jump discontinuity in the
backward-Euler residual at $f_F$. Below the threshold the residual is
$f - f_\text{old} - \Delta t\,\dot f$, which the accelerated softening
drives sharply negative as $f \to f_F^-$; at and above it every rate is
switched off and the residual is $f - f_\text{old} > 0$. The residual
steps over zero without touching it, so the first step that would
separate the interface has no root and simply fails to converge.

Switching on the converged state removes the discontinuity: the whole
step is taken with intact rates, $f$ is free to pass $f_F$ (the
saturated $f^*$ keeps every softening denominator bounded while it
does), and the next step freezes as described above. Separation is then
resolved to within one time step — the usual first-order accuracy of an
explicit switch.
:::

### Example model composition

The input below assembles the complete model, integrating $f$, $N$ and
$[\![u]\!]_t^v$ implicitly. Note that the normal viscous opening is
*not* integrated: it is the algebraic function $NV(f, N) - N_0V_0$, so
the kinematic identity holds exactly at every converged step rather
than only to integration order.

```{literalinclude} ../../../../tests/unit/fixtures/sham_needleman/complete.i
:language: ini
```

The nucleation-free variant drops
[](models-CavityNucleationRate), [](models-CavityNucleationDamageRate)
and the [](models-ScalarLinearCombination) that summed the two
contributions, renames the growth rate straight onto
`cavity_fraction_rate`, and prescribes `cavity_density` as a constant
force — two unknowns instead of three. The prescribed-strain-rate
variant drops only [](models-SlidingDrivenStrainRate) and makes
`strain_rate` a force, which is the right form when the surrounding
grains are creeping and *their* strain rate, not the boundary sliding,
sets how fast cavities appear. Both live beside `complete.i` under
`tests/unit/fixtures/sham_needleman/`.

## See also

- [](tutorials-models-composition) — the general pattern of stitching
  primitive `Model`s together with [](models-ComposedModel) and
  letting the dependency resolver wire variable names automatically.
- [](tutorials-models-input-file) — anatomy of the HIT input format
  used throughout this page.
- [](modules-solid-mechanics-elasticity) — bulk-elastic primitives
  often paired with a TSL when modelling a bonded interface in a
  larger solid-mechanics calculation.
- [](syntax-catalog) — the complete autogenerated catalog of every
  registered `Model` type, with the full option list for each
  primitive referenced above.
