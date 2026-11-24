(modules-pbm)=
# Population balance (fragmentation)

## Overview

The population balance (PBM) primitives model the evolution of a
particle size distribution under fragmentation -- grinding and milling,
where coarse particles break into finer ones. They build directly on the
[](modules-finite-volume) module: the size axis is discretized into
cell-centered finite-volume bins, and fragmentation is written as a
conservative flux between bins so that total mass is conserved by
construction. The formulation follows the general population balance
framework
([Chemical Engineering Science, 2022](https://doi.org/10.1016/j.ces.2022.117925)).

The module adds two `Model` primitives on top of the finite-volume catalog;
the rest of the pipeline -- boundary conditions, flux divergence, time
integration, and the implicit solve -- reuses the finite-volume and common
primitives unchanged.

## Math

For fragmentation alone, the number density $n(v,t)$ of particles with
volume $v$ evolves by a birth and death balance: a parent of volume $w > v$
breaks at rate $\gamma(w)$ into a distribution of daughters $p(v,w)$, while
particles of volume $v$ are consumed at rate $\gamma(v)$,

$$
\frac{\partial n(v,t)}{\partial t}
= \int_v^\infty \gamma(w)\, p(v,w)\, n(w,t)\, dw
- \gamma(v)\, n(v,t).
$$

Working in the mass density $u = \rho v n$ (mass per unit volume coordinate)
makes total mass a linear functional of the state and lets fragmentation be
written in conservative form. Substituting gives

$$
\frac{\partial u}{\partial t}
= \int_v^\infty \gamma(w)\,
  \frac{\rho(v) v}{\rho(w) w}\, p(v,w)\, u(w,t)\, dw
- \gamma(v)\, u(v,t),
$$

which, using the Leibniz rule, is equivalent to the conservation law
$\partial_t u + \partial_v J = 0$ with flux

$$
J(v,t) = -\int_v^\infty \gamma(w)\, u(w,t)
  \left(\int_0^v \frac{\rho(\zeta)\zeta}{\rho(w)w}\, p(\zeta,w)\, d\zeta\right) dw.
$$

### Discrete fragment flux operator

On $N$ cells with centers $v_j$, widths $\Delta v_j$, and densities
$\rho_j$, the double integral collapses to a kernel matrix

$$
K_{kj} = \Delta v_j\, \Delta v_k\, \gamma_j\,
         \frac{\rho_k v_k}{\rho_j v_j}\, p_{kj},
$$

where the row index $k$ is the daughter (child) class and the column index
$j$ is the parent. The flux at interior edge $i+\tfrac{1}{2}$ is the total
rate at which mass crosses from bins above $i$ into bins at or below $i$,
which is a cumulative sum of $K$ over the child axis restricted to
$j > i$,

$$
J_{i+\frac{1}{2}} = \sum_{j} M_{ij}\, u_j,
\qquad
M_{ij} =
\begin{cases}
-\sum_{k \le i} K_{kj}, & j > i, \\
0, & \text{otherwise.}
\end{cases}
$$

$M$ is thus a negative cumulative sum of $K$ over the child axis, masked to
the strict upper triangle, giving an $(N-1) \times N$ operator that is
independent of $u$. [](models-FiniteVolumeFragmentationFlux) assembles
$M$; [](models-IntermediateLinearContraction) forms $J = M u$. Appending
zero-flux boundary conditions with [](models-FiniteVolumeAppendBoundaryCondition)
and taking the divergence with [](models-FiniteVolumeGradient) recovers the
cell update $\dot{\bar u}_i = -(J_{i+\frac{1}{2}} - J_{i-\frac{1}{2}})/\Delta v_i$.

### Constitutive choices

The model is closed by two constitutive functions.

**Fragmentation rate** $\gamma(v)$ -- the rate at which a particle of size
$v$ breaks. Larger particles typically break more readily, so a form that
increases with size (e.g. a power law $\gamma \propto v^\alpha$, or a rate
obtained from DEM) is common. Bins with $\gamma = 0$ act as stable sinks
that accumulate fines without breaking further.

**Daughter distribution** $p(v,w)$ -- the expected number of fragments of
size $v$ from one breakage of a parent of size $w$. Two constraints make it
physical:

$$
p(v,w) = 0 \ \text{for}\ v > w,
\qquad
\int_0^w \rho(v)\, v\, p(v,w)\, dv = \rho(w)\, w,
$$

i.e. no fragment exceeds its parent, and breakage conserves mass. Common
families include Broadbent-Callcott (1956), the empirical Austin $B_{ij}$
function (Austin, Klimpel &amp; Luckie 1984), and the JKMRC $t_{10}$
correlation.

:::{note}
Total mass is conserved by the flux form for any daughter matrix. Total
particle *volume* $\sum_i (u_i/\rho_i)\,\Delta v_i$ is conserved only when
$p$ additionally satisfies $\sum_k p_{kj} v_k (1 - \rho_k/\rho_j) = 0$. For
uniform density this is automatic; for non-uniform density it has a
non-trivial solution only when the density is non-monotone and the breakage
targets stable sink bins. The
`tests/verification/finite_volume/pbm/conservation_volume` case demonstrates
both invariants holding at non-uniform density.
:::

## Catalog

The PBM module adds two primitives to the finite-volume catalog; both are
composed inside a `ComposedModel` driven by an `ImplicitUpdate`.

| Type                                        | Role                                                                        |
| :------------------------------------------ | :-------------------------------------------------------------------------- |
| [](models-FiniteVolumeFragmentationFlux)    | Assembles the $(N-1)\times N$ fragment flux operator $M$ from the per cell density, volume, width, fragmentation rate, and breakage matrix |
| [](models-IntermediateLinearContraction)    | Applies an operator to a field over the last intermediate axis, $J = M u$ (generic; reusable beyond fragmentation) |

## Example: grinding

The input below grinds a coarse feed on 20 uniform density bins: the
fragmentation rate increases with bin size (bin 0 is a sink), and each
parent splits its mass equally over all smaller bins,
$p_{kj} = v_j / (j\, v_k)$ for $k < j$, which conserves mass at uniform
density.

```{literalinclude} grinding.i
:language: ini
```

### Walkthrough

- Five [](models-ScalarParameterToVariable) instances expose the
  constitutive fields (`cell_density`, `cell_volume`, `cell_width`,
  `fragmentation_rate`, `breakage_matrix`) as variables so the composed
  graph produces them rather than requiring them as external inputs.
- [](models-FiniteVolumeFragmentationFlux) (`frag_flux`) assembles the
  fragment flux operator `M` from those fields; it does not depend on the
  size distribution `u`.
- [](models-IntermediateLinearContraction) (`flux`) contracts `M` against
  the distribution `u` to give the interior edge flux `J`.
- Two [](models-FiniteVolumeAppendBoundaryCondition) calls (`left_bc`,
  `right_bc`) append zero-flux values at both ends, extending `J` from the
  $N-1$ interior edges to the full $N+1$-edge axis.
- [](models-FiniteVolumeGradient) (`flux_divergence`) takes the divergence
  of the padded flux with `dx = cell_width_val`, giving the cell-centered
  rate `u_rate`.
- [](models-ScalarBackwardEulerTimeIntegration) (`integrate_u`) turns the
  rate into a backward-Euler residual, `implicit_rate` bundles the residual
  graph, and [](models-ImplicitUpdate) (`model`) drives the
  [](solvers-Newton) + [](solvers-DenseLU) solve that
  `TransientDriver` evaluates each step.

## Examples

End-to-end scenarios live under `tests/`:

- `tests/regression/finite_volume/pbm/grinding/` -- the grinding case above,
  pinned against a checked-in trajectory.
- `tests/regression/finite_volume/pbm/conservation/` -- non-uniform density
  fragmentation that conserves both mass and volume.
- `tests/verification/finite_volume/pbm/` -- `two_bin` (closed-form decay),
  `distributed_ic` (reproduces the initial condition), and
  `conservation_volume` (mass + volume conservation).

## Worked example

An end-to-end notebook that runs the grinding model and plots the evolving
size distribution and the conserved mass:

```{toctree}
:maxdepth: 1

grinding
```

## See also

- [](modules-finite-volume) -- the discretization operators, boundary
  conditions, and transport pattern this module extends.
- [](tutorials-models-implicit-model) and [](tutorials-models-transient-driver)
  -- how `ImplicitUpdate`, `NonlinearSystem`, and `TransientDriver` compose,
  the same pattern used above with the rate supplied by the fragmentation
  operator.
- [](syntax-catalog) -- per type option lists for
  [](models-FiniteVolumeFragmentationFlux) and
  [](models-IntermediateLinearContraction).
