# Per-site backward Euler x_i = (x~1_i + dt s) / (1 - k_i dt) with per-site rates
# k_i (so dx_i/ds differs by site), reduced by vf = sum_i 4/3 pi R_i^3 x_i:
#   d vf / d s = sum_i 4/3 pi R_i^3 dt / (1 - k_i dt).
# y = R z never meets the solve: y:z is "paired", pairs through the solve "full".

[Settings]
  [example_batch_shape]
    x = '(2; 4)'
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
