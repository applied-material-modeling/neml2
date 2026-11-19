# implicit_sub_batch_ift_reduce over a (2, 3) site grid.

[Settings]
  [example_batch_shape]
    x = '(2; 2, 3)'
    z = '(2; 2, 3)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'Scalar(torch.logspace(-1.0, 0.0, 6, dtype=torch.float64).reshape(2, 3), sub_batch_ndim=2)'
  []
  [k]
    type = Python
    expr = 'Scalar(-torch.logspace(0.0, 1.0, 6, dtype=torch.float64).reshape(2, 3), sub_batch_ndim=2)'
  []
[]

[Models]
  [rate]
    type = ScalarLinearCombination
    from = 'x s'
    to = 'x_rate'
    weights = 'k 1'
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'x'
    time = 't'
  []
  [residual]
    type = ComposedModel
    models = 'rate integrate'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [vf]
    type = PrecipitateVolumeFraction
    number_density = 'x'
    volume_fraction = 'vf'
    radius = 'R'
  []
  [y]
    type = ScalarLinearCombination
    from = 'z'
    to = 'y'
    weights = 'R'
  []
  [model]
    type = ComposedModel
    models = 'update vf y'
    additional_outputs = 'x'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'residual'
    unknowns = 'x'
    residuals = 'x_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-12
    rel_tol = 1e-12
    max_its = 10
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]
