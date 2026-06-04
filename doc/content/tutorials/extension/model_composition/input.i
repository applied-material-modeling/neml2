# Compose the custom ProjectileAcceleration with built-in time-integration
# and implicit-update blocks to integrate a projectile trajectory.

[Tensors]
  # Three different bags of balls, each bag a different drag coefficient.
  [mu]
    type = Python
    expr = 'Scalar(torch.tensor([0.1, 0.5, 1.0]))'
  []
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 2.0, 100)'
  []
  # Initial conditions carry a TRAILING size-1 dynamic-batch placeholder
  # (-> shape (5, 1, *base)) so the (3,) viscosity broadcasts cleanly into
  # the second batch axis. The composed run evaluates 5 launches x 3
  # viscosities = 15 trajectories in one Newton solve per step.
  [x0]
    type = Python
    expr = 'Vec.zeros(5).dynamic_batch.unsqueeze(-1)'
  []
  [v0]
    type = Python
    expr = '''
      v0 = Vec.fill(6.0, 8.0, 0.0)
      v1 = Vec.fill(8.0, 7.0, 0.0)
      v2 = Vec.fill(10.0, 6.0, 0.0)
      v3 = Vec.fill(12.0, 5.0, 0.0)
      v4 = Vec.fill(14.0, 4.0, 0.0)
      stacked = stack([v0.dynamic_batch, v1.dynamic_batch, v2.dynamic_batch, v3.dynamic_batch, v4.dynamic_batch])
      result = stacked.dynamic_batch.unsqueeze(-1)
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
