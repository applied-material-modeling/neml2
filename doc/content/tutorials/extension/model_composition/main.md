@insert-title:tutorials-extension-model-composition

[TOC]

## Problem definition

Recall that the complete model for simulating the projectile trajectory is
\f{align}
  \dot{\boldsymbol{x}} & = \boldsymbol{v}, \label{1} \\
  \dot{\boldsymbol{v}} & = \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}, \label{2} \\
  \mathbf{r} = & = \begin{Bmatrix}
    \tilde{\boldsymbol{x}} - \boldsymbol{x}_n - \left(t - t_n\right) \dot{\boldsymbol{x}} \\
    \tilde{\boldsymbol{v}} - \boldsymbol{v}_n - \left(t - t_n\right) \dot{\boldsymbol{v}} \\
  \end{Bmatrix}, \label{3} \\
  \begin{Bmatrix}
    \boldsymbol{x} \\
    \boldsymbol{v}
  \end{Bmatrix} & = \mathop{\mathrm{root}}\limits_{\tilde{\boldsymbol{x}}, \tilde{\boldsymbol{v}}} \left( \mathbf{r} \right), \label{4}
\f}
subject to appropriate initial conditions \f$\boldsymbol{x}_0\f$ and \f$\boldsymbol{v}_0\f$

Among these equations:
- \f$\eqref{1}\f$ and \f$\eqref{3}\f$ can be defined using [VecBackwardEulerTimeIntegration](#vecbackwardeulertimeintegration).
- \f$\eqref{2}\f$ is the custom model `ProjectileAcceleration` which we have implemented in previous tutorials.
- \f$\eqref{4}\f$ is the [ImplicitUpdate](#implicitupdate).

This tutorial demonstrates that our custom model `ProjectileAcceleration` can be composed with other existing, predefined NEML2 models.

## Input file

The following input file composes the constitutive model for a single-step update for 5 projectiles each with a different dynamic viscosity, i.e., the shape of "mu" is \f$(5;)\f$.

@list-input:extension/model_composition/input.i:Models,Solvers,Tensors/g,Tensors/mu

## Lauching the projectiles

To obtain the entire trajectory, the constitutive model need to be recursively integrated. As discussed in a previous [tutorial](#tutorials-models-transient-driver), a transient driver should be used to perform the recursive constitutive update. Conveniently, in this example, since we are dealing with an autonomous system (i.e., no external "forces" are needed to drive the constitutive update), the vanilla [TransientDriver](#transientdriver) can be used:
@list-input:extension/model_composition/input.i:Drivers,Tensors/end_time,Tensors/times,Tensors/x0,Tensors/v0

The five projectiles are launched from the same position (the origin) but with two different lauching velocities. Note how broadcasting is used to simultaneously simulate the trajectories of all 10 combinations.

@list:cpp:extension/model_composition/ex1.cxx:main

Output
@list-output:ex1

The following Python script plots the trajectories loaded from "result.pt" written by the driver.
@list:python:extension/model_composition/ex2.py

![Projectile trajectories](tutorials/extension/model_composition/trajectories.svg){html: width=85%}

@insert-page-navigation
