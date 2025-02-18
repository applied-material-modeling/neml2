@insert-title:tutorials-extension

[TOC]

NEML2 models can be easily extended or adapted to model the material behavior of your interest.

Before diving into the following tutorials, it is optional but recommended to read the following articles from the developer guide:
- @ref dev-getting-started
- @ref naming-conventions
- @ref model-visualization

## Problem description

This set of tutorials define a custom equation to model the conventional *projectile motion* problem:
\f{align}
  \dot{\boldsymbol{x}} & = \boldsymbol{v}, \label{1} \\
  \dot{\boldsymbol{v}} & = \boldsymbol{a} = \boldsymbol{g} - \mu \boldsymbol{v}, \label{2}
\f}
where \f$\boldsymbol{x}\f$ and \f$\boldsymbol{v}\f$ are the position and velocity of the cannonball, respectively. \f$\boldsymbol{g}\f$ is the gravitational acceleration, and \f$\mu\f$ is the dynamic viscosity. The projectile's trajectory can be numerically integrated using the backward-Euler method:
\f{align}
  \mathbf{r} = \begin{Bmatrix}
    r_x \\
    r_v
  \end{Bmatrix} & = \begin{Bmatrix}
    \boldsymbol{x} - \boldsymbol{x}_n - \left(t - t_n\right) \dot{\boldsymbol{x}} \\
    \boldsymbol{v} - \boldsymbol{v}_n - \left(t - t_n\right) \dot{\boldsymbol{v}} \\
  \end{Bmatrix}, \label{3} \\
  \begin{Bmatrix}
    \boldsymbol{x} \\
    \boldsymbol{v}
  \end{Bmatrix} & = \mathop{\mathrm{root}}\limits_{\boldsymbol{x}, \boldsymbol{v}} \left( \mathbf{r} \right), \label{4}
\f}
subject to appropriate initial conditions \f$\boldsymbol{x}_0\f$ and \f$\boldsymbol{v}_0\f$

## Seaching for available models

Among these equations:
- \f$\eqref{1}\f$ and \f$\eqref{3}\f$ can be defined using [VecBackwardEulerTimeIntegration](#vecbackwardeulertimeintegration).
- \f$\eqref{2}\f$ is not available in NEML2, and is the focus of this set of tutorials. We will name it as `ProjectileAcceleration`.
- \f$\eqref{4}\f$ is the [ImplicitUpdate](#implicitupdate).

## Outline

Each tutorial builds on top of the previous tutorials, introduces and explains one small piece in model development. It is therefore recommended to follow these tutorials in order.

@insert-subsection-list

The final model is then used in another set of [tutorials](#tutorials-optimization) on parameter calibration and inference.
