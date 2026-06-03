(modules-porous-flow)=
# Porous flow

## Overview

The `porous_flow` module collects the constitutive ingredients that
appear in (unsaturated) flow through a porous medium: the
capillary-pressure correlations that close the two-phase
pressure-saturation relationship, the porosity-permeability laws
that close Darcy's law, the effective-saturation transformation
that maps a flowing-phase volume fraction onto the normalized
saturation those correlations expect, and a variational advective
stress that ties a porous-flow problem back to a finite-deformation
solid driven by the same Jacobian. The catalog is intentionally a
set of small leaves — every type takes one or two `Scalar` inputs
and returns a `Scalar`, so they snap together with
[`ComposedModel`](models-ComposedModel) into the full
pressure→saturation→permeability chain that a host code (or another
NEML2 driver) needs.

## Math

Let $\varphi$ denote the volume fraction of the flowing fluid in
the medium and $\varphi_{\mathrm{max}}$ the maximum fraction it
can occupy at full saturation; the residual (immobile) saturation
is $S_r$. The **effective saturation** is the normalized fraction

$$
S_e = \frac{\varphi/\varphi_{\mathrm{max}} - S_r}{1 - S_r},
$$

so that $S_e \in [0, 1]$ runs between irreducible and full
saturation. Capillary-pressure correlations are written against
$S_e$ rather than $\varphi$ directly. The catalog ships the two
standard correlations:

$$
P_c^{\mathrm{BC}}(S_e) = P_t \, S_e^{-1/p},
\qquad
P_c^{\mathrm{vG}}(S_e) = a \left( S_e^{-1/m} - 1 \right)^{1-m},
$$

— the **Brooks-Corey** law with threshold pressure $P_t$ and shape
exponent $p$, and the **van Genuchten** law with shape parameters
$(a, m)$. Both diverge as $S_e \to 0$; both leaves take an optional
log-linear extension below a transition saturation $S_p$ that keeps
$P_c$ finite for the numerics.

Permeability $K$ is a function of porosity $\varphi$ (not
saturation), and three closures are provided. With reference values
$(\varphi_0, K_0)$ they are

\begin{align}
K^{\mathrm{pow}}(\varphi) &= K_0 \left( \frac{\varphi}{\varphi_0} \right)^p, \\
K^{\mathrm{exp}}(\varphi) &= K_0 \exp\!\left[ -a\,(\varphi - \varphi_0) \right], \\
K^{\mathrm{KC}}(\varphi)  &= K_0 \, \frac{(\varphi / (1 - \varphi))^n}{(\varphi_0 / (1 - \varphi_0))^m}.
\end{align}

The first is a plain **power law** with exponent $p$, the second
an **exponential law** with scale $a$, and the third a
**Kozeny-Carman** form parametrized by shape exponents $(n, m)$.

Finally, when the porous flow is coupled to a finite-deformation
solid, the **advective stress** associated with swelling- or
phase-change-induced volume change is the scalar

$$
p_s = -\frac{c}{3\, J_s^{5/3} J_t^{2/3}} \, P_{ij} F_{ij},
$$

where $F$ is the deformation gradient, $P$ the first
Piola-Kirchhoff stress, $c$ a volume-change coefficient, and
$J_s, J_t$ optional swelling and thermal Jacobians (default to $1$
when omitted).

## Example

The example below is a `ModelUnitTest` for the Brooks-Corey
capillary-pressure law with its log-linear extension turned on. It
takes the effective saturation $S$ as the only input and produces
the capillary pressure $P_c$:

```{literalinclude} ../../../tests/models/porous_flow/BrooksCoreyCapillaryPressure.i
:language: ini
```

## Explanation

The `[Models]` block declares a single
[](models-BrooksCoreyCapillaryPressure) instance. Two parameters
close the constitutive law — `threshold_pressure` ($P_t = 2.3$) and
`exponent` ($p = 0.7$) — and the two I/O lines wire the model's
canonical port names to the tensors used by the driver:
`effective_saturation = 'S'` maps the input port to the `[S]`
tensor declared above, and `capillary_pressure = 'Pc'` names the
output that the driver compares against the reference values in
`[Pc]`. Switching `log_extension = true` together with
`transition_saturation = 0.1` activates the log-linear extension
below $S_e = 0.1$, which is why the smallest sample point
($S = 0.05$) lands on a finite $P_c \approx 126$ rather than the
formal divergence of $P_t S_e^{-1/p}$.

In a real composition, the upstream of this model is
[](models-EffectiveSaturation), which converts a raw fluid volume
fraction $\varphi$ to $S_e$, and the downstream consumer is
whatever solver assembles the two-phase flow residual. The
porosity-permeability leaves —
[](models-PowerLawPermeability),
[](models-ExponentialLawPermeability), and
[](models-KozenyCarmanPermeability) — slot in independently against
$\varphi$, since permeability is a property of the medium rather
than the fluid state. The drop-in alternative for the capillary
branch is [](models-VanGenuchtenCapillaryPressure), which has the
same I/O signature with a different functional shape. For finite-deformation coupling,
[](models-AdvectiveStress) consumes $(F, P)$ plus the optional
$J_s, J_t$ Jacobians and emits a single scalar that the host stress
divergence picks up. The full catalog and per-type option list
lives in the [](syntax-catalog) under the `Models` section.

## See also

- [](tutorials-models-composition) — the canonical walkthrough for
  wiring multiple `Model` instances into a `ComposedModel` via the
  port-name rules used above.
- [](syntax-catalog) — option-by-option reference for every type
  in this module (`Models` section).
- [](modules-solid-mechanics-elasticity),
  [](modules-phase-field-fracture) — sibling physics modules that
  pair naturally with `AdvectiveStress` for coupled
  porous-flow / solid problems.
