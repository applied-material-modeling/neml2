# NEML2 1D finite volume transport equation solver

## Strong form 

Consider a control volume $[x,x+\Delta x]$ with some conserved quantity $u(x)$, some arbitrary flux $J(x,t)$, and reaction term $R(u,x,t)$.  We can write conservation of mass as
$$
\frac{d}{dt}\int_{x}^{x+\Delta x} u dx = J(x,t) - J(x + \Delta x, t) + \int_x^{x+\Delta x} R dx
$$
Rewrite the flux term
$$
    \frac{d}{dt}\int_{x}^{x+\Delta x} u dx = - \int_x^{x+\Delta x} \frac{\partial J}{\partial x} dx + \int_x^{x+\Delta x} R dx
$$
or
$$
    \int_{x}^{x+\Delta x} \left(u_t + J_x - R \right) dx = 0
$$
or as it holds for all volumes
$$
    u_t + J_x = R
$$

We have our common choice of the flux as a sum of
$$
    J = J_{diffusion} + J_{advection}
$$
with
$$
    J_{diffusion} = -D u_x
$$
and
$$
    J_{advection} = v u
$$
with $D$ the diffusivity and $v$ the advection velocity.  This gives the classical equation
$$
    u_t + \left(v u \right)_x - \left( D u_x \right)_x = R
$$

We will also need boundary conditions, which can be Dirichlet or Neumann on either end of the domain $[a,b]$.

## Finite volume discretization

### Basics

Let's say our domain is $[a,b]$ and partition it into control volumes of
$$
    I_i = \left[ x_{i-\frac{1}{2}}, x_{i+\frac{1}{2}} \right]\, i=1,\dots,N
$$
with $\Delta x_i= x_{i+\frac{1}{2}} - x_{i-\frac{1}{2}}$ so the cells are all centered at $x_i$.

Integrate the PDE over $I_i$
$$
    \int_{I_i} \left(u_t + J_x \right) dx = \int_{I_i} R dx
$$
which we can rewrite as
$$
    \frac{d}{dt} \int_{I_i} u dx + \left( J\big\rvert_{x_{i+\frac{1}{2}}} - J\big\rvert_{x_{i-\frac{1}{2}}} \right) = \int_{I_i} R dx
$$

Define the cell average operator as 
$$
    \bar{f} = \frac{1}{\Delta x_i} \int_{I_i} f dx.
$$

We can then write this as 
$$
    \frac{d \bar{u}_i}{dt} + \frac{1}{\Delta x_i} \left( J\big\rvert_{x_{i+\frac{1}{2}}} - J\big\rvert_{x_{i-\frac{1}{2}}} \right) =\bar{R}_i
$$
which is the basic semi-discrete form.  To make a numerical method we need to chose a time integration scheme and, in practice, do things to the fluxes to stabilize the advective terms.

### Fluxes

Let our flux be
$$
    J = J_{diffusion} + J_{advection}
$$
again.

#### Diffusive

The diffusive flux is relatively easy, all we need to do is reconstruct the derivative:
$$
    J_{diffusion, i + \frac{1}{2}} = -D_{i+\frac{1}{2}} \frac{\bar{u}_{i+1} - \bar{u}_i}{x_{i+1} - x_i}
$$
and, for now, until someone complains choose
$$
    D_{i+\frac{1}{2}} = \frac{1}{2} \left( D_i + D_{i+1} \right)
$$
for a uniform grid.

#### Advection

Let's just do first order upwinding.  Define the stabilized advective flux as
$$
    \hat{J}_{advection,i+\frac{1}{2}} = v^+_{i+\frac{1}{2}} \bar{u}_i + v^-_{i+\frac{1}{2}} \bar{u}_{i+1}
$$
with
$$
    v^\pm_{i+\frac{1}{2}} = \frac{1}{2} \left(v_{i+\frac{1}{2}} \pm \left| v_{i + \frac{1}{2}} \right| \right)
$$
and
$$
    v_{i+\frac{1}{2}} = \frac{1}{2} \left(v_i + v_{i+1} \right)
$$
at least for a uniform grid.

### Boundary conditions

We'll want give the user choices of Dirichlet and Neumann boundary conditions on either end of the domain.  They should allow for user-defined functions of time to give the fixed value/flux value.

### Coordinate system transformations

I want to define our finite volume discretizations over the interval $x^\prime \in [0,1]$ and then map this to the physical domain $x$.  I want to consider at least two maps:

$$
x \in [a,b]
$$
such that
$$
x = \left(b-a \right) x^\prime + a
$$

and 
$$
x \in [0,\infty)
$$
such that
$$
x = \frac{x^\prime}{1-x^\prime}
$$

To implement this we just need maps from the reference $[0,1]$ domain and the Jacobian of that maps, so we can alter any spatial derivatives in the above.

## Rules for NEML2 implementation

The neml2 documentation is available here for general context: https://applied-material-modeling.github.io/neml2/

For context on what the different types of tensor dimensions mean in NEML2 see `include/tensors/TensorBase.h`

1. We'll make this a new submodule, with new subfolders in `include/models`, `src/models`, and the tests called `transport`.
2. For each model we need a corresponding unit test that checks correctness of both the model itself and the derivatives.  Examples are in `tests/unit/models`.
3. When we have a working system we'll need regression tests in `tests/regression/models/transport` and at least one verification test against an analytic solution in `tests/verification/models/transport`.
4. All models must be able to handle arbitrary dynamic batch dimensions.
5. Our fundemental values will be $\bar{u}_i$ in cells.  We will store this in a neml2 `Scalar` variable with an intermediate dimension representing the discretized quantities.
6. The user will provide functions of the form $f(t,x,\bar{u})$ to calculate the advection velocity $v$, the diffusion coefficient $D$, and the reaction $R$ at cells.  The details of these functions are outside the scope of the transport system, but we'll implement a few simple examples to help debug and verify the system.
7. We will reuse the existing time integration routines, nonlinear implicit system, and solvers to do the discrete time integration.  I may ask you to implement the generalized alpha method at some point, but we'll start with backward Euler. 
8. Our main task then will be to work towards formulating the  $\frac{d \bar{u}_i}{dt} = \bar{R}_i - \frac{1}{\Delta x_i} \left( J\big\rvert_{x_{i+\frac{1}{2}}} - J\big\rvert_{x_{i-\frac{1}{2}}} \right)$ to feed into the time integration routines.  We'll break this into logical submodels, especially to calculate the cell boundary fluxes.
9. Our reference domain will always be equal-sized cells in $[0,1]$, but we need to accomodate the coordinate mapping described above.

