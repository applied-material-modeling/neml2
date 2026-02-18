# Finite Volume Transport Physics {#finite-volume}

[TOC]

The finite volume transport physics module provides composable building blocks for solve 1D PDEs with finite volume methods. The models are designed for arbitrary batch dimensions and
integrate with the standard NEML2 time integration and nonlinear solve infrastructure.  The module could be used for arbitrary PDEs, but the provided examples and tests focus
on 1D transport (advection-diffusion-reaction) PDEs.

## Finite volume discretization

We consider a conserved quantity \f$u(x,t)\f$ with total flux \f$J\f$ and reaction term \f$R\f$:

\f[
  u_t + J_x = R,
\f]

over a domain \f$\Omega \in [a,b]\f$.

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

### Gradients

Gradient (and in 1D divergence) terms are handled with the first order expression

\f[
  \mathrm{grad}_{u, i+\frac{1}{2}} = -p_{i+\frac{1}{2}} \frac{\bar{u}_{i+1}-\bar{u}_i}{x_{i+1}-x_i}.
\f]

### Advective flux

The module provides an expression for stabilized advective fluxes via simple upwinding:

\f[
  \hat{J}_{\mathrm{advection}, i+\frac{1}{2}} = v^+_{i+\frac{1}{2}} \bar{u}_i + v^-_{i+\frac{1}{2}} \bar{u}_{i+1},
\f]

with

\f[
  v^\pm_{i+\frac{1}{2}} = \frac{1}{2}\left(v_{i+\frac{1}{2}} \pm |v_{i+\frac{1}{2}}|\right).
\f]


## Finite volume transport model building blocks

The finite volume transport module introduces several small models intended to be composed together:

- **LinearlyInterpolateToCellEdges**: Interpolates cell-centered values onto cell edges on nonuniform grids.
- **FiniteVolumeGradient**: Computes prefactor-weighted gradients at cell edges from \f$u\f$, edge prefactor values, and the spacing between cell centers.
- **FiniteVolumeUpwindedAdvectiveFlux**: Computes \f$J_{\mathrm{advection}}\f$ at cell edges with first-order upwinding from \f$u\f$ and edge velocity \f$v_{edge}\f$.
- **FiniteVolumeAppendBoundaryCondition**: Appends a boundary value to the left or right side of an intermediate dimension (useful for both Dirichlet and Neumann boundary conditions).

In addition, helper tensors are provided for common 1D mesh setups:

- **LinspaceScalar**: Uniform edge locations or time grids.
- **CenterScalar**: Cell centers from edge locations.
- **DifferenceScalar**: Cell sizes from edge locations.
- **GaussianScalar**: Convenient for initializing Gaussian initial conditions.

## Example: combined advection–diffusion–reaction

Below is a compact example that assembles a full transport system, applies boundary conditions, and advances in time using backward Euler. The full regression test can be found in tests/regression/finite_volume/combined/model.i.

@list-input:../../../tests/regression/finite_volume/combined/model.i

## Verification and regression tests

The module includes unit tests for each component and verification tests for advection, diffusion, and combined advection–diffusion–reaction problems. These are located under tests/unit/models/finite_volume and tests/verification/finite_volume.
