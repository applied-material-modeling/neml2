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

Rather than ship monolithic TSL classes, NEML2 provides a catalog of
small composable primitives that you assemble into a TSL via a
[](models-ComposedModel). Each primitive does exactly one mathematical
step, so a custom TSL — e.g. a different damage formula paired with
the bilinear envelope, or the Camanho–Davila initiation paired with a
new propagation criterion — needs no Python at all; just write a new
input file.

The catalog is organized by role:

**Kinematic helpers** (live in `models/common/`):
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

**Traction-assembly primitives** (each emits the public `traction`
`Vec`):
- [](models-OrthotropicLinearTraction) — orthotropic linear elasticity
  with no internal state. Optionally accepts a `normal_penetration`
  channel and a separate `penalty_stiffness` to elastically resist
  interpenetration.
- [](models-BilinearTraction) — Camanho/Davila-style cohesive-zone
  assembly. Computes the bilinear damage internally, caps it for
  irreversibility against the previous-step value, and exposes the
  damage as a secondary output for inspection.
- [](models-SalehaniIraniTraction) — 3D coupled exponential law of
  Salehani & Irani; internally damaged with a previous-step cap, so
  load–unload–reload freezes the softness at its historical peak.

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
power-law form. The exponential law of Salehani & Irani replaces the
bilinear envelope with a coupled exponential — the normal direction
enters linearly and the two tangential directions enter quadratically
in a single coupling exponent — and is assembled as a single
[](models-SalehaniIraniTraction) primitive.

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
   ($\langle \delta_n \rangle_+$) and its magnitude-of-negative part
   as `normal_penetration` ($\langle \delta_n \rangle_-$). The
   positive part drives damage and softening; the negative part is
   handed to the traction assembly as the elastic penetration channel
   that keeps compression damage-free.
3. **`mode_mixity`** ([](models-ModeMixity)) computes
   $\beta = \delta_s / \langle \delta_n \rangle_+$. This is the only
   mode-mixity number the rest of the chain consumes.
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
7. **`traction`** ([](models-BilinearTraction)) consumes
   $\delta_m$, $\delta_c$, $\delta_f$, the per-component jumps, and
   the elastic penetration channel; computes the bilinear damage $d$;
   caps it for irreversibility against the previous-step value; and
   assembles the traction vector. Damage is exposed as a secondary
   output (via `additional_outputs = 'damage'` on the composed model)
   for inspection but is internal state of this primitive.

The `[Drivers]` block prescribes the time grid and the displacement
jump history through a `TransientDriver`, and a `TransientRegression`
pins the run against `gold/result.pt`.

:::{note}
The bilinear TSLs ([](models-BilinearTraction), with `damage` capped
internally) and [](models-SalehaniIraniTraction) both carry
irreversible internal state. Under load–unload–reload schedules the
unloading limb returns elastically along the secant of the current
damaged stiffness, and reloading resumes the softening branch only
after the historical peak separation is exceeded. The `bilinear_unload`
scenario under
`tests/regression/solid_mechanics/traction_separation_law/` exercises
this behavior.
:::

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
