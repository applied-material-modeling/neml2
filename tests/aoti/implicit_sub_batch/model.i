# implicit_sub_batch: ImplicitUpdate with a per-site sub-batched unknown.
# The single-group Newton-of-DenseLU path still goes through the Block*
# export wrappers (the "block" in their name refers to the multi-group
# layout support, not to SchurComplement) -- they're the sub-batch-aware
# disassemble/assemble path. With ScalarBackwardEulerTimeIntegration on
# `x` and ``sub_batch=(4,)``, the assembled Jacobian is ``(*B, 4, 1, 1)``
# (per-site 1x1 blocks) and torch.linalg.solve batches over the leading
# (B, 4) axes naturally -- one independent 1x1 LU per site.

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
    max_its = 25
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

[Settings]
  example_batch_shape = '(2; 4)'
[]
