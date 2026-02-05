# Solver {#system-solvers}

[TOC]

Refer to [Syntax Documentation](@ref syntax-solvers) for the list of available objects.

Many material models are _implicit_, meaning that the update of the material model is the solution to one or more nonlinear systems of equations. While a model or a composition of models can define such nonlinear system, a solver is required to actually _solve_ the system.

## Nonlinear solver

All nonlinear solvers derive from the common base class `NonlinearSolver`. The base class defines 3 public members: `atol` for the absolute tolerance, `rtol` for the relative tolerance, and `miters` for the maximum number of iterations.

Derived classes must override the method
```cpp
NonlinearSolver::Result solve(NonlinearSystem & system)
```
The argument is the nonlinear system of equations to be solved. The return value contains a return code and the number of iterations taken before convergence. The nonlinear solver reads the current unknowns from the system and updates them in-place during the solve.

Each nonlinear solver also owns a linear solver, configured via the `linear_solver` option, that is used to solve the linearized system in each iteration.

While the convergence criteria are defined by the specific solvers derived from the base class, it is generally recommended to use both `atol` and `rtol` in the convergence check. Below is an example convergence criteria
```cpp
bool
MySolver::converged(const ATensor & nR, const ATensor & nR0) const
{
  return at::all(at::logical_or(nR < atol, nR / nR0 < rtol)).item<bool>();
}
```
where `nR` is the vector norm of the current residual, and `nR0` is the vector norm of the initial residual (evaluated at the initial guess). The above statement makes sure the current residual is either below the absolute tolerance or has been sufficiently reduced, and the condition is applied to _all_ batches of the residual norm.

## Nonlinear system

The first argument passed to the `NonlinearSolver::solve` method is of type `NonlinearSystem &`. Nonlinear systems are defined under the `EquationSystems` section in input files. NEML2 provides `ModelNonlinearSystem` (use `type = NonlinearSystem`) to wrap a `Model` as a nonlinear system. The wrapped model must satisfy the following requirements:
1. The input axis must have a sub-axis named "state".
2. The output axis must have a sub-axis named "residual".
3. The input "state" sub-axis and the output "residual" sub-axis must be conformal, i.e., the variable names on the two axes must have one-to-one correspondence.

With these requirements, the following rules are implied during the evaluation of a nonlinear system:
1. The input "state" sub-axis is used as the initial guess. The driver is responsible for setting the appropriate initial guess.
2. The output "residual" sub-axis is the residual of the nonlinear system.
3. The derivative sub-block "(residual, state)" is the Jacobian of the nonlinear system.

Since the input "state" sub-axis and the output "residual" sub-axis are required to be conformal, the Jacobian of the nonlinear system must be square (while not necessarily symmetric).

In input files, [ImplicitUpdate](#implicitupdate) references a nonlinear system via the `equation_system` option, which points to an entry under the `EquationSystems` section.
