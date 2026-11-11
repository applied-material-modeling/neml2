(modules-finite-volume)=
# Finite volume

## Overview

The `finite_volume` module provides composable building blocks for
solving 1D PDEs with the cell-centered finite volume method. The
primitives batch over arbitrary leading dimensions and plug into
`TransientDriver`, `ImplicitUpdate`, and `ComposedModel` like any
other NEML2 model, so finite-volume transport models compose with
the rest of the framework through the same interfaces a
constitutive model uses.

The primitives generalize to a range of cell-centered finite-volume
problems; the provided examples and regression tests focus on 1D
transport (advection-diffusion-reaction). The module ships:

- **Discretization operators** — `LinearlyInterpolateToCellEdges`,
  `FiniteVolumeGradient`, `FiniteVolumeUpwindedAdvectiveFlux`.
- **Boundary-condition utilities** — `FiniteVolumeAppendBoundaryCondition`.
- **Coordinate and source helpers** — `SemiInfiniteCoordinateTransform`,
  `SmearedDeltaSource`, `DumpInSmallestBin`.

Composition (via `ComposedModel`) wires these together with generic
algebraic models like `ScalarLinearCombination` and time-integration
primitives like `ScalarBackwardEulerTimeIntegration` to produce a
full discretized transport equation that `ImplicitUpdate` then
advances in time.

## Math

Consider a conserved quantity $u(x,t)$ with total flux $J$ and
reaction source $R$ on a domain $\Omega = [a,b]$, governed by

$$
u_t + J_x = R.
$$

Partition $[a,b]$ into $N$ cells

$$
I_i = \left[x_{i-\tfrac{1}{2}}, x_{i+\tfrac{1}{2}}\right],
\quad i = 1, \dots, N,
$$

with widths $\Delta x_i = x_{i+\tfrac{1}{2}} - x_{i-\tfrac{1}{2}}$
and cell centers $x_i$. The cell average is

$$
\bar{u}_i = \frac{1}{\Delta x_i}\int_{I_i} u\, dx.
$$

Integrating the PDE over $I_i$ and dividing by $\Delta x_i$ yields
the canonical semi-discrete finite-volume update

$$
\frac{d \bar{u}_i}{dt}
+ \frac{1}{\Delta x_i}\left(J\big\rvert_{x_{i+\tfrac{1}{2}}}
- J\big\rvert_{x_{i-\tfrac{1}{2}}}\right) = \bar{R}_i,
$$

which the catalog reconstructs in three pieces — edge values,
gradients, and advective fluxes — each provided by its own
primitive.

### Cell-edge interpolation

For nonuniform grids, cell-edge values are obtained by linear
interpolation between adjacent cell centers,

$$
q_{i+\tfrac{1}{2}}
= \frac{x_{i+1} - x_{i+\tfrac{1}{2}}}{x_{i+1} - x_i}\, q_i
+ \frac{x_{i+\tfrac{1}{2}} - x_i}{x_{i+1} - x_i}\, q_{i+1}.
$$

This is implemented by [](models-LinearlyInterpolateToCellEdges),
and is used to lift cell-centered diffusivities or velocities onto
edges where the fluxes live.

### Gradients

Gradient (and, in 1D, divergence) terms use the first-order
expression

$$
\mathrm{grad}_{u, i+\tfrac{1}{2}}
= -p_{i+\tfrac{1}{2}}\,
\frac{\bar{u}_{i+1} - \bar{u}_i}{x_{i+1} - x_i},
$$

where $p$ is an optional cell-edge prefactor (e.g. a diffusivity
for Fickian diffusion). [](models-FiniteVolumeGradient) computes
this for both diffusive fluxes (edge prefactor = $D$, spacing =
cell-center spacing) and flux divergences (no prefactor, spacing =
cell widths).

### Advective flux

Advective fluxes are stabilized with first-order upwinding,

\begin{align}
\hat{J}_{\mathrm{adv}, i+\tfrac{1}{2}}
&= v^{+}_{i+\tfrac{1}{2}}\, \bar{u}_i
 + v^{-}_{i+\tfrac{1}{2}}\, \bar{u}_{i+1}, \\
v^{\pm}_{i+\tfrac{1}{2}}
&= \tfrac{1}{2}\left(v_{i+\tfrac{1}{2}}
 \pm \left|v_{i+\tfrac{1}{2}}\right|\right),
\end{align}

and computed by [](models-FiniteVolumeUpwindedAdvectiveFlux) from
cell-centered $\bar{u}$ and an edge velocity $v_{\mathrm{edge}}$.

### Boundary conditions

Dirichlet and Neumann boundary conditions are imposed by appending
a prescribed value to the left or right end of an intermediate
tensor with [](models-FiniteVolumeAppendBoundaryCondition). Applied
to the flux array $J$, this realizes a Neumann condition; applied
to $u$ on a ghost edge, it realizes a Dirichlet condition. The two
ends are configured by chaining the primitive twice, once per side.

## Example: combined advection-diffusion-reaction

The following input file discretizes

$$
\partial_t u + \partial_x\!\left(v u - D\, \partial_x u\right)
= -k\, u,
\qquad u(x, 0) = e^{-\frac{(x - 0.25)^2}{2\,(0.05)^2}},
$$

on $x \in [0, 1]$ with $v = 0.4$, $D = 0.5$, $k = 0.05$, zero-flux
boundary conditions, and backward-Euler time integration:

```{literalinclude} ../../../tests/regression/finite_volume/combined/model.i
:language: ini
```

## Explanation

The example builds a single implicit-update model out of three
layers.

**Mesh and field tensors** (`[Tensors]`) — `edges` is a 1D
`Scalar` carrying the cell edges with `sub_batch_ndim=1`. That
puts the spatial axis in sub-batch, so any leading batch dimension
(parameter sweep, ensemble of simulations) composes on top without
any code change here. `centers` and
`dx_centers` derive cell-center positions and center-to-center
spacings; `dx` is the cell-width vector. The initial condition
`ic` is a Gaussian centered at $x = 0.25$, and `D_cells`,
`v_cells` populate cell-centered diffusivity and velocity.

**Edge lift** — two [](models-LinearlyInterpolateToCellEdges)
instances map `D_cells` and `v_cells` from cell centers onto cell
edges, producing the edge tensors `D` and `v_edge` that the flux
operators expect.

**Flux assembly** — the diffusive flux is computed by
[](models-FiniteVolumeGradient) with `prefactor = D` and
`dx = dx_centers`, giving $-D\,\partial_x u$ at each interior edge.
The advective flux $v u$ is computed by
[](models-FiniteVolumeUpwindedAdvectiveFlux). Both are scalars on
the $(N-1)$-sized edge axis. A `ScalarLinearCombination` sums them
into the total flux `J`.

**Boundary conditions** — `J` lives on the $N-1$ interior edges,
so two [](models-FiniteVolumeAppendBoundaryCondition) calls append
zero-flux values at the left and right ends, producing
`J_with_bc_left_with_bc_right` on the full $N+1$-edge axis.

**Reaction and divergence** — a `ScalarLinearCombination` with
weight $-k$ gives the reaction term $R = -k u$. A second
[](models-FiniteVolumeGradient) (no prefactor; `dx = dx`) computes
the flux divergence $J_x$ per cell. Their sum is the
cell-centered rate of change.

**Time integration and solve** — [](models-ScalarBackwardEulerTimeIntegration)
turns the rate-of-change expression into a residual whose root in
$u$ is the backward-Euler update. The `[EquationSystems]` block
wraps it as a `NonlinearSystem` in the unknown `concentration`,
solved each step by [](solvers-Newton) with the [](solvers-DenseLU)
linear solver and a [](models-ConstantExtrapolationPredictor)
initial guess. The outer [](models-ImplicitUpdate) is what
`TransientDriver` actually evaluates at every step of the
prescribed time grid.

The data flow is therefore:
`concentration → {diffusive, advective} flux → J → bc → flux_div`
and `concentration → R`, summed into `concentration_rate`, then
implicitly integrated to advance `concentration` to the next step.

## See also

- [](tutorials-models-composition) and [](tutorials-models-cross-referencing)
  — the general rules for composing models and wiring variables
  between them.
- [](tutorials-models-implicit-model) and [](tutorials-models-transient-driver)
  — how `ImplicitUpdate`, `NonlinearSystem`, and `TransientDriver`
  fit together. The finite-volume example reuses exactly this
  pattern, with the rate-of-change expression supplied by the
  discretization operators instead of a constitutive law.
- [](syntax-catalog) — per-type option lists for every primitive
  used above, including the helpers
  [](models-SemiInfiniteCoordinateTransform),
  [](models-SmearedDeltaSource), and [](models-DumpInSmallestBin)
  that did not appear in the worked example.
