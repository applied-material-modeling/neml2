# A per-site given g = R s, computed before the solve from the plain s:
#   x_i = (x~1_i + dt R_i s) / (1 + dt),  d vf / d s = sum_i 4/3 pi R_i^4 dt / (1 + dt).

[Settings]
  [example_batch_shape]
    x = '(2; 4)'
    g = '(2; 4)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'logspace(Scalar(-1.0).sub_batch, Scalar(0.0).sub_batch, 4)'
  []
[]

[Models]
  [g]
    type = ScalarLinearCombination
    from = 's'
    to = 'g'
    weights = 'R'
  []
  [rate]
    type = ScalarLinearCombination
    from = 'x g'
    to = 'x_rate'
    weights = '-1 1'
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
  [model]
    type = ComposedModel
    models = 'g update vf'
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
