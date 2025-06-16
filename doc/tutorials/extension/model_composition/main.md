@insert-title:tutorials-extension-model-composition

[TOC]

## Problem definition

Recall that the complete model for simulating the projectile trajectory is
\f{align}
  \dot{\boldsymbol{x}} & = \boldsymbol{v}, \label{1} \\
  \dot{\boldsymbol{v}} & = \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}, \label{2} \\
  \mathbf{r} = & = \begin{Bmatrix}
    \boldsymbol{x} - \boldsymbol{x}_n - \left(t - t_n\right) \dot{\boldsymbol{x}} \\
    \boldsymbol{v} - \boldsymbol{v}_n - \left(t - t_n\right) \dot{\boldsymbol{v}} \\
  \end{Bmatrix}, \label{3} \\
  \begin{Bmatrix}
    \boldsymbol{x} \\
    \boldsymbol{v}
  \end{Bmatrix} & = \mathop{\mathrm{root}}\limits_{\boldsymbol{x}, \boldsymbol{v}} \left( \mathbf{r} \right), \label{4}
\f}
subject to appropriate initial conditions \f$\boldsymbol{x}_0\f$ and \f$\boldsymbol{v}_0\f$

Among these equations:
- \f$\eqref{1}\f$ and \f$\eqref{3}\f$ can be defined using [VecBackwardEulerTimeIntegration](#vecbackwardeulertimeintegration).
- \f$\eqref{2}\f$ is the custom model `ProjectileAcceleration` which we have implemented in previous tutorials.
- \f$\eqref{4}\f$ is the [ImplicitUpdate](#implicitupdate).

This tutorial demonstrates that our custom model `ProjectileAcceleration` can be composed with other existing, predefined NEML2 models.

## Input file

The following input file composes the constitutive model for a single-step update for 5 projectiles each with a different dynamic viscosity, i.e., the shape of "mu" is \f$(5;)\f$.

```
[Tensors]
  [g]
    type = Vec
    values = '0 -9.81 0'
  []
  [mu]
    type = Scalar
    values = '0.01 0.05 0.1 0.5 1'
    batch_shape = (5)
  []
[]

[Models]
  [eq2]
    type = ProjectileAcceleration
    velocity = 'state/v'
    acceleration = 'state/a'
    gravitational_acceleration = 'g'
    dynamic_viscosity = 'mu'
  []
  [eq3a]
    type = VecBackwardEulerTimeIntegration
    variable = 'state/x'
    rate = 'state/v'
  []
  [eq3b]
    type = VecBackwardEulerTimeIntegration
    variable = 'state/v'
    rate = 'state/a'
  []
  [eq3]
    type = ComposedModel
    models = 'eq3a eq3b'
  []
  [system]
    type = ComposedModel
    models = 'eq2 eq3'
  []
  [eq4]
    type = ImplicitUpdate
    implicit_model = 'system'
    solver = 'newton'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
  []
[]
```

## Lauching the projectiles

To obtain the entire trajectory, the constitutive model need to be recursively integrated. As discussed in a previous [tutorial](#tutorials-models-transient-driver), a transient driver should be used to perform the recursive constitutive update. Conveniently, in this example, since we are dealing with an autonomous system (i.e., no external "forces" are needed to drive the constitutive update), the vanilla [TransientDriver](#transientdriver) can be used:
```
[Drivers]
  [driver]
    type = TransientDriver
    model = 'eq4'
    prescribed_time = 'times'
    ic_Vec_names = 'state/x state/v'
    ic_Vec_values = 'x0 v0'
    save_as = 'result.pt'
    show_input_axis = true
    show_output_axis = true
  []
[]

[Tensors]
  [end_time]
    type = Scalar
    values = '1'
    batch_shape = (1,1)
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = 'end_time'
    nstep = 100
  []
  [x0]
    type = Vec
    values = '0 0 0'
  []
  [v0]
    type = Vec
    values = "10 5 0
              8 6 0"
    batch_shape = (2,1)
  []
[]
```
The five projectiles are launched from the same position (the origin) but with two different lauching velocities. Note how broadcasting is used to simultaneously simulate the trajectories of all 10 combinations.

@source:src1
```cpp
#include "neml2/drivers/Driver.h"

int
main()
{
  using namespace neml2;
  set_default_dtype(kFloat64);
  auto factory = load_input("input.i");
  auto driver = factory->get_driver("driver");
  driver->run();
}
```
@endsource

Output
```
@attach-output:src1
```

The following Python script plots the trajectories loaded from "result.pt" written by the driver.
@source:src2
```python
import torch
from matplotlib import pyplot as plt

# Load the result
res = torch.jit.load("result.pt")
O = dict(res.output.named_buffers())

# Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
nstep = 100
nvel = 2 # two launching velocities
nproj = 5 # five projectiles (with different dynamic viscosity)
for i in range(nvel):
    for j in range(nproj):
        x = [O["{}.state/x".format(n)][i, j, 0].item() for n in range(1, nstep)]
        y = [O["{}.state/x".format(n)][i, j, 1].item() for n in range(1, nstep)]
        ax[i].plot(x, y, "--")
    ax[i].set_xlabel("x")
    ax[i].set_ylabel("y")
    ax[i].grid()
fig.tight_layout()
fig.savefig("trajectories.svg")
```
@endsource

![Projectile trajectories](tutorials/extension/model_composition/trajectories.svg){html: width=85%}

@insert-page-navigation
