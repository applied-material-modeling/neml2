# Fixture for the eager C++ convergence-error path (tests/cpp/test_eager.cpp).
#
# Same minimal ImplicitUpdate as tests/aoti/implicit_simple, but the Newton
# solver is configured with `max_its = 0`: against any non-zero initial residual
# the C++ Newton in libneml2.so throws a *recoverable* ConvergenceError without
# taking a step. The eager runtime must surface that across the embedded-Python
# boundary as neml2::aoti::ConvergenceError (recoverable() == true), not a plain
# FatalError. Kept under tests/cpp (NOT tests/aoti) so the AOTI compile sweep,
# which runs every tests/aoti/<scenario>/model.i, never tries to drive it.

[Models]
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'integrate'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 0
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
[]
