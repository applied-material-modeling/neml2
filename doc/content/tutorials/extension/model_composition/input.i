# Compose the custom ProjectileAcceleration with built-in time-integration
# and implicit-update blocks to integrate a projectile trajectory.

[Tensors]
  [mu]
    type = Python
    expr = 'Scalar([0.01, 0.05, 0.1, 0.5, 1.0])'
  []
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.0, 100)'
  []
  [x0]
    type = Python
    expr = 'Vec.zeros()'
  []
  [v0]
    type = Python
    expr = 'Vec.fill(10.0, 5.0, 0.0)'
  []
[]

[Models]
  [eq2]
    type = ProjectileAcceleration
    velocity = 'v'
    acceleration = 'a'
    dynamic_viscosity = 'mu'
  []
  [eq3a]
    type = VecBackwardEulerTimeIntegration
    variable = 'x'
    rate = 'v'
  []
  [eq3b]
    type = VecBackwardEulerTimeIntegration
    variable = 'v'
    rate = 'a'
  []
  [eq3]
    type = ComposedModel
    models = 'eq3a eq3b'
  []
  [system]
    type = ComposedModel
    models = 'eq2 eq3'
  []
[]

[EquationSystems]
  [eq4]
    type = NonlinearSystem
    model = 'system'
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
  [implicit]
    type = ImplicitUpdate
    equation_system = 'eq4'
    solver = 'newton'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'implicit'
    prescribed_time = 'times'
    ic_Vec_names = 'x v'
    ic_Vec_values = 'x0 v0'
    save_as = 'result.pt'
  []
[]
