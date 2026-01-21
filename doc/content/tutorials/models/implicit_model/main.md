@insert-title:tutorials-models-implicit-model

[TOC]

## Problem description

One of the most notable differences between constitutive models and feed-forward neural networks is that updating certain stiff systems with implicit methods is often more computationally efficient compared to explicit algorithms.

A generally nonlinear, recursive, implicit system of equations take the following form
\f{align*}
  \mathbf{r}(\tilde{\mathbf{s}}) & = f(\tilde{\mathbf{s}}, \mathbf{f}, \mathbf{s}_n, \mathbf{f}_n; \mathbf{p}), \\
  \mathbf{s} &= \mathop{\mathrm{root}}\limits_{\tilde{\mathbf{s}}} (\mathbf{r}).
\f}
Here \f$ \mathbf{r} \f$ represents the residual (for root-fiding) of the system of equations, and \f$ \mathbf{s} \f$, \f$ \mathbf{f} \f$, \f$ \mathbf{s}_n \f$, \f$ \mathbf{f}_n \f$ are defined by four reserved sub-axes in NEML2:
- The `state` sub-axis hosts input variables in set \f$ \tilde{\mathbf{s}} \f$. The variables on this sub-axis are the primary unknowns to be solved for. After solving the system, the `state` sub-axis hosts output variables in set \f$ \mathbf{s} \f$.
- The `forces` sub-axis hosts *prescribed* input variables in set \f$ \mathbf{f} \f$. These variables are prescribed and, by definition, do not change while the system is being solved.
- The `old_state` and `old_forces` sub-axes respectively correspond to \f$ \mathbf{s}_n \f$ and \f$ \mathbf{f}_n \f$. These variables correspond to the *previous* solution to the system to facilitate the recursive definition of internal variables in history-dependent models. The equivalent plastic strain in plasticity models is a well-known example.

The Perzyna viscoplasticity model mentioned in a [previous tutorial](#tutorials-models-model-composition) takes this form:
\f{align}
  \boldsymbol{\varepsilon}^e & = \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p, \label{1} \\
  \boldsymbol{\sigma} & = 3K\operatorname{vol}\boldsymbol{\varepsilon}^e + 2G\operatorname{dev}\boldsymbol{\varepsilon}^e, \label{2} \\
  \bar{\sigma} & = J_2(\boldsymbol{\sigma}), \label{3} \\
  f^p & = \bar{\sigma} - \sigma_y, \label{4} \\
  \boldsymbol{N} & = \pdv{f^p}{\boldsymbol{\sigma}}, \label{5} \\
  \dot{\gamma} & = \left( \dfrac{\left< f^p \right>}{\eta} \right)^n, \label{6} \\
  \dot{\boldsymbol{\varepsilon}}^p & = \dot{\gamma} \boldsymbol{N}. \label{7}
\f}
Its residual can be defined using backward-Euler time integration as
\f{align}
  \mathbf{r}\left( \boldsymbol{\varepsilon}^p \right) = \boldsymbol{\varepsilon}^p - \boldsymbol{\varepsilon}^p_n - \left( t - t_n \right) \dot{\boldsymbol{\varepsilon}}^p, \label{8}
\f}
the solution procedure of which is often referred to as the *return-mapping algorithm* in the solid mechanics community.

This tutorial illustrates the use of [ImplicitUpdate](#implicitupdate) in conjunction with a Newton-Raphson solver to perform the implicit update.

## Defining the system of equations

After searching among existing models provided by NEML2, we are able to translate each of these equations into a NEML2 model. The input file looks like

@list:input:implicit_model/input1.i

Note the one-to-one correspondance between the models and the equations.

The structure of the system of equations can be summarized using the code below.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:implicit_model/ex1.cxx

  Output:
  @list-output:ex1
- <b class="tab-title">Python</b>
  @list:python:implicit_model/ex2.py

  Output:
  @list-output:ex2

</div>

## Solving the system of equations

Once the system of equations are properly defined, we can use the [ImplicitUpdate](#implicitupdate) to solve the system of equations. The [ImplicitUpdate](#implicitupdate) is responsible for the following:
- (Re)declare the solution to the system of equations as output variables.
- Validate the shape of input variables and residual variables to make sure the system is square.
- Assemble the residual vector and Jacobian matrix of the underlying linear system.
- Invoke a *solver* to solve the system of equations.
- Apply the implicit function theorem to calculate exact derivatives (up to machine precision).

NEML2 offers three fully vectorized Newton solvers to be used in conjunction with [ImplicitUpdate](#implicitupdate):
- [Newton](#newton), the (vectorized) vanilla version of the Newton-Raphson algorithm which always takes the "full" step.
- [NewtonWithLineSearch](#newtonwithlinesearch), similar to Newton but offers several commonly used (again fully vectorized) line search strategies.

In addition, the assembly routines as well as the application of the implicit function theorem are also implemented in a vectorized fashion.

The additional sections needed in the input file are
@list-input:implicit_model/input2.i:Models/model,Solvers/newton

The [ImplicitUpdate](#implicitupdate) model can then be invoked in the same way as regular models.

<div class="tabbed">

- <b class="tab-title">C++</b>
  @list:cpp:implicit_model/ex3.cxx

  Output:
  @list-output:ex3
- <b class="tab-title">Python</b>
  @list:cpp:implicit_model/ex4.py

  Output:
  @list-output:ex4

</div>

## Remarks on the implicit function theorem

Unlike other regular models, declaring variables on the *correct* sub-axes is important because NEML2 relies on the reserved sub-axes (`state`, `forces`, etc.)
- To determine whether the variable value and derivatives should be calculated during the assembly of residual and Jacobian. For example, the derivatives with respect to all variables on the `forces` sub-axis are skipped because they are not required in the assembly of the linear system.
- To efficiently reuse the factorization of the system Jacobian when applying the implicit function theorem.

As long as models are defined using the *correct* sub-axis definitions and satisfy some mild continuity requirements, **NEML2 guarantees the correctness of the variable derivatives** after one or more implicit updates, up to machine precision. The same guarantee also applies to user-defined custom models.

This is a significant advantage compared to some of the alternative constitutive model libraries, especially in the context of coupling with external PDE solvers. For example, in the context of finite element method, thermodynamic forces (e.g. strain) are calculated at each quadrature point, and the constitutive library (e.g. NEML2) is responsible for updating the thermodynamic state variables (e.g. stress, plastic strain, etc.), which are then used in the residual definition of the discretized PDE. Therefore, the exact derivatives of the state variables with respect to the forces are the key to the assembly of the exact Jacobian of the descretized PDE, which is in turn the fundamental requirement for optimal convergence for many nonlinear solvers.

@insert-page-navigation
