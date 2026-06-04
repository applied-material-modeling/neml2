# Compose the custom ProjectileAcceleration with built-in time-integration
# and implicit-update blocks to integrate a projectile trajectory.

[Tensors]
  [mu]
    type = Python
    expr = 'Scalar(0.1)'
  []
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 2.0, 100)'
  []
  [x0]
    type = Python
    expr = 'Vec.zeros(5)'
  []
  [v0]
    type = Python
    expr = '''
      v0 = Vec.fill(6.0, 4.0, 0.0)
      v1 = Vec.fill(8.0, 5.0, 0.0)
      v2 = Vec.fill(10.0, 5.0, 0.0)
      v3 = Vec.fill(12.0, 5.0, 0.0)
      v4 = Vec.fill(14.0, 5.0, 0.0)
      result = v0.dynamic_batch.stack([v1, v2, v3, v4])
    '''
  []
[]

[Models]
  [eq1]
    type = ProjectileAcceleration
    velocity = 'v'
    acceleration = 'a'
    dynamic_viscosity = 'mu'
  []
  [eq2]
    type = VecBackwardEulerTimeIntegration
    variable = 'x'
    rate = 'v'
  []
  [eq3]
    type = VecBackwardEulerTimeIntegration
    variable = 'v'
    rate = 'a'
  []
  [residual]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]

[EquationSystems]
  [system]
    type = NonlinearSystem
    model = 'residual'
    unknowns = 'x v'
    residuals = 'x_residual v_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [eq4]
    type = ImplicitUpdate
    equation_system = 'system'
    solver = 'newton'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'eq4'
    prescribed_time = 'times'
    ic_Vec_names = 'x v'
    ic_Vec_values = 'x0 v0'
  []
[]
