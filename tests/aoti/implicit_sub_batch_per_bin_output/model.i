# implicit_sub_batch_per_bin_output: an ImplicitUpdate over a sub-batched
# unknown followed by a forward segment whose OUTPUT keeps the sub-batch axis.
#
# ``implicit_sub_batch_forward`` reduces over the bins before returning (``vf``),
# so it never returns a per-bin forward output. Here ``y = R x`` scales the
# solved per-bin unknown by the sub-batched constant ``R`` bin by bin (the KWN /
# PBM "unscale" step). With the solved unknown's sub-batch axis read as plain
# batch (#479), the product with ``R`` inserts an extra axis and the compiled
# graph returns ``y`` at ``(2, 4, 4)`` instead of ``(2, 4)``; the eager-vs-AOTI
# comparison in ``test_aoti_export_reload_matches_eager`` catches it.

[Settings]
  [example_batch_shape]
    x = '(2; 4)'
    x_rate = '(2; 4)'
    t = '(2,)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'logspace(Scalar(-1.0).sub_batch, Scalar(0.0).sub_batch, 4)'
  []
[]

[Models]
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [y]
    type = ScalarLinearCombination
    from = 'x'
    to = 'y'
    weights = 'R'
  []
  [model]
    type = ComposedModel
    models = 'update y'
    additional_outputs = 'x'
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
