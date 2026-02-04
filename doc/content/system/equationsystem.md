# Equation System {#system-equationsystems}

[TOC]

Refer to [Syntax Documentation](@ref syntax-equationsystems) for the list of available objects.

Equation systems define the algebraic systems that implicit solvers work on. They are created under
the `EquationSystems` section in input files and are typically consumed by [ImplicitUpdate](#implicitupdate)
or by nonlinear solvers directly. Equation systems are tightly coupled with solvers because the
solver queries the system for the assembled operator and right-hand side.

## Base class

All equation systems derive from [EquationSystem](@ref neml2::EquationSystem), which is a
manufacturable object with its options section set to `EquationSystems`. The base class provides
the `to` interface for moving internal data to a different device or dtype.

## Linear system interface

[LinearSystem](@ref neml2::LinearSystem) defines the interface for a system of the form
\f[
  A u = b,
\f]
with the following roles:
- Unknowns \f$u\f$: set with `set_u` and retrieved with `u`.
- Given variables \f$g\f$: set with `set_g` and retrieved with `g`.
- Operator \f$A\f$: assembled with `A()` or the combined `A_and_*` methods.
- Right-hand side \f$b\f$: assembled with `b()` or the combined `A_and_*` methods.
- Auxiliary matrix \f$B = \partial r / \partial g\f$: assembled by `A_and_B` or
  `A_and_B_and_b` when available.

The assembled objects are stored in [SparseTensorList](@ref neml2::SparseTensorList), which is
a vector of tensors where missing entries represent zeros in a sparse representation. The order of
entries is controlled by the ID-to-variable maps returned by `umap`, `gmap`, and `bmap`, with
associated layouts (`ulayout`, `glayout`, `blayout`) and optional intermediate layouts.

`LinearSystem` tracks whether \f$A\f$, \f$B\f$, and \f$b\f$ are up-to-date. By default,
`g_changed` invalidates all cached assemblies, while `u_changed` is a no-op for linear systems.
Derived classes can override these hooks and the `assemble` method to control caching and
assembly behavior.

## Nonlinear system

[NonlinearSystem](@ref neml2::NonlinearSystem) represents a nonlinear system
\f[
  r(u; g) = 0.
\f]
Its linearization is expressed as
\f[
  \frac{\partial r}{\partial u} \Delta u = -r,
\f]
so the assembled matrices are interpreted as \f$A := \partial r / \partial u\f$ and
\f$b := -r\f$. The auxiliary matrix \f$B := \partial r / \partial g\f$ is also available when
requested. In a nonlinear system, `u_changed` invalidates \f$A\f$, \f$B\f$, and \f$b\f$ because the
linearization depends on the current unknowns.

## Model-based nonlinear system

[ModelNonlinearSystem](@ref neml2::ModelNonlinearSystem) is the default implementation and is
registered under the alias `NonlinearSystem`. It wraps a [Model](@ref neml2::Model) and uses the
model's variables to define the system:
- Unknowns \f$u\f$ are the input variables on the `state` sub-axis.
- Given variables \f$g\f$ are all other input variables.
- Residuals \f$r\f$ are the output variables on the `residual` sub-axis.
- \f$A\f$ and \f$B\f$ are assembled from the model's derivatives of `residual` with respect to
  `state` and non-state inputs.

The `state` and `residual` sub-axes must be conformal so that the Jacobian \f$A\f$ is square.
This is the same requirement imposed by solvers (see [Solver](#system-solvers)).

In an input file, a model-based nonlinear system looks like:
```
[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'system'
  []
[]
```
This system can then be referenced by [ImplicitUpdate](#implicitupdate) or passed directly to a
nonlinear solver.
