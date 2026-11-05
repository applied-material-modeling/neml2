# implicit_sub_batch_ift_reduce with the unknown x declared only through its
# history x~1 and guessed by a predictor, so x is no model input: the post-solve
# segment's x must still be traced, and recorded, per site.

[Settings]
  [example_batch_shape]
    x~1 = '(2; 4)'
    z = '(2; 4)'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = 'logspace(Scalar(-1.0).sub_batch, Scalar(0.0).sub_batch, 4)'
  []
  [k]
    type = Python
    expr = '-logspace(Scalar(0.0).sub_batch, Scalar(1.0).sub_batch, 4)'
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
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'x'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
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
