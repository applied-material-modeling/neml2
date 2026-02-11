# Finite Volume Transport Physics {#finite_volume}

[TOC]

The finite volume transport physics module provides composable building blocks for 1D finite-volume advection–diffusion–reaction systems. The models are designed for arbitrary batch dimensions and integrate with the standard NEML2 time integration and nonlinear solve infrastructure.

## Governing equation

We consider a conserved quantity \f$u(x,t)\f$ with total flux \f$J\f$ and reaction term \f$R\f$:

\f[
  u_t + J_x = R,
\f]

with

\f[
  J = J_{diffusion} + J_{advection}, \quad
  J_{diffusion} = -D u_x, \quad
  J_{advection} = v u.
\f]

This yields the standard advection–diffusion–reaction equation

\f[
  u_t + (v u)_x - (D u_x)_x = R.
\f]

Boundary conditions can be Dirichlet or Neumann (flux) on the left/right ends of the domain.

## Finite-volume discretization

Partition \f$[a,b]\f$ into cells

\f[
  I_i = \left[x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}}\right], \quad i=1,\dots,N,
\f]

with \f$\Delta x_i = x_{i+\frac{1}{2}} - x_{i-\frac{1}{2}}\f$ and cell centers \f$x_i\f$.
Define the cell average \f$\bar{u}_i\f$ as

\f[
  \bar{u}_i = \frac{1}{\Delta x_i}\int_{I_i} u\, dx.
\f]

Integrating the PDE over \f$I_i\f$ gives

\f[
  \frac{d \bar{u}_i}{dt} + \frac{1}{\Delta x_i}\left(J\big\rvert_{x_{i+\frac{1}{2}}} - J\big\rvert_{x_{i-\frac{1}{2}}}\right) = \bar{R}_i.
\f]

### Cell-edge interpolation

For nonuniform grids, cell-edge values are obtained by linear interpolation:

\f[
  q_{i+\frac{1}{2}} = \frac{x_{i+1}-x_{i+\frac{1}{2}}}{x_{i+1}-x_i} q_i
  + \frac{x_{i+\frac{1}{2}}-x_i}{x_{i+1}-x_i} q_{i+1}.
\f]

### Prefactor-weighted gradient

The prefactor-weighted gradient uses a first-order reconstructed gradient:

\f[
  \mathrm{grad}_{u, i+\frac{1}{2}} = -p_{i+\frac{1}{2}} \frac{\bar{u}_{i+1}-\bar{u}_i}{x_{i+1}-x_i}.
\f]

### Advective flux

The advective flux uses first-order upwinding:

\f[
  \hat{J}_{advection, i+\frac{1}{2}} = v^+_{i+\frac{1}{2}} \bar{u}_i + v^-_{i+\frac{1}{2}} \bar{u}_{i+1},
\f]

with

\f[
  v^\pm_{i+\frac{1}{2}} = \frac{1}{2}\left(v_{i+\frac{1}{2}} \pm |v_{i+\frac{1}{2}}|\right).
\f]

## Finite volume transport model building blocks

The finite volume transport module introduces several small models intended to be composed together:

- **LinearlyInterpolateToCellEdges**: Interpolates cell-centered values onto cell edges on nonuniform grids.
- **FiniteVolumeGradient**: Computes prefactor-weighted gradients at cell edges from \f$u\f$, edge prefactor values, and the spacing between cell centers.
- **AdvectiveFlux**: Computes \f$J_{advection}\f$ at cell edges with first-order upwinding from \f$u\f$ and edge velocity \f$v_{edge}\f$.
- **TransportBoundaryCondition**: Appends a boundary value to the left or right side of an intermediate dimension (useful for flux boundary conditions).
- **ScalarLinearCombination**: Combines flux divergence and reaction to form \f$\dot{u}\f$.

In addition, helper tensors are provided for common 1D mesh setups:

- **LinspaceScalar**: Uniform edge locations or time grids.
- **CenterScalar**: Cell centers from edge locations.
- **DifferenceScalar**: Cell sizes from edge locations.
- **GaussianScalar**: Convenient Gaussian initial conditions.

## Example: combined advection–diffusion–reaction

Below is a compact example that assembles a full transport system, applies boundary conditions, and advances in time using backward Euler. The full regression test can be found in tests/regression/finite_volume/combined/model.i.

```
[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 201
    dim = 0
    group = 'intermediate'
  []
  [centers]
    type = CenterScalar
    points = 'edges'
  []
  [dx_centers]
    type = DifferenceScalar
    points = 'centers'
  []
  [dx]
    type = DifferenceScalar
    points = 'edges'
  []
  [ic]
    type = GaussianScalar
    points = 'centers'
    width = 0.05
    height = 1.0
    center = 0.25
  []
[]

[Models]
  [diffusivity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 0.5
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/prefactor'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 0.4
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'state/v_edge'
  []
  [diffusive_flux]
    type = FiniteVolumeGradient
    u = 'state/concentration'
    prefactor = 'state/prefactor'
    dx = 'dx_centers'
  []
  [advective_flux]
    type = AdvectiveFlux
    u = 'state/concentration'
    v_edge = 'state/v_edge'
  []
  [reaction]
    type = ScalarLineCombination
    from_var = 'state/concentration'
    to_var = 'state/R'
    coefficients = 0.05
  []
  [total_flux]
    type = ScalarLinearCombination
    from_var = 'state/grad_u state/J_advection'
    to_var = 'state/J'
    coefficients = '1 1'
  []
  [left_bc]
    type = TransportBoundaryCondition
    input = 'state/J'
    bc_value = 0.0
    side = 'left'
  []
  [right_bc]
    type = TransportBoundaryCondition
    input = 'state/J_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [flux_divergence]
    type = FiniteVolumeGradient
    u = 'state/J_with_bc_left_with_bc_right'
    prefactor = 1
    dx = 'dx'
    grad_u = 'state/flux_div'
  []
  [rate_of_change]
    type = ScalarLinearCombination
    from_var = 'state/R state/flux_div'
    to_var = 'state/concentration_rate'
    coefficients = '1 1'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/concentration'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'diffusivity advection_velocity diffusive_flux advective_flux reaction total_flux left_bc right_bc flux_divergence rate_of_change integrate_u'
  []
[]
```

## Verification and regression tests

The module includes unit tests for each component and verification tests for advection, diffusion, and combined advection–diffusion–reaction problems. These are located under tests/unit/models/finite_volume and tests/verification/finite_volume.
