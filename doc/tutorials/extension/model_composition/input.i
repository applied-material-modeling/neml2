[Drivers]
  [driver]
    type = TransientDriver
    model = 'eq4'
    prescribed_time = 'times'
    ic_Vec_names = 'state/x state/v'
    ic_Vec_values = 'x0 v0'
    save_as = 'result.pt'
  []
[]

[Tensors]
  [g]
    type = Vec
    values = '0 -9.81 0'
  []
  [mu]
    type = Scalar
    values = '0.01 0.05 0.1 0.5 1'
    batch_shape = (5)
  []
  [end_time]
    type = Scalar
    values = '1'
    batch_shape = (1,1)
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = 'end_time'
    nstep = 100
  []
  [x0]
    type = Vec
    values = '0 0 0'
  []
  [v0]
    type = Vec
    values = "10 5 0
              8 6 0"
    batch_shape = (2,1)
  []
[]

[Models]
  [eq2]
    type = ProjectileAcceleration
    velocity = 'state/v'
    acceleration = 'state/a'
    gravitational_acceleration = 'g'
    dynamic_viscosity = 'mu'
  []
  [eq3a]
    type = VecBackwardEulerTimeIntegration
    variable = 'state/x'
    rate = 'state/v'
  []
  [eq3b]
    type = VecBackwardEulerTimeIntegration
    variable = 'state/v'
    rate = 'state/a'
  []
  [eq3]
    type = ComposedModel
    models = 'eq3a eq3b'
  []
  [system]
    type = ComposedModel
    models = 'eq2 eq3'
  []
  [eq4]
    type = ImplicitUpdate
    implicit_model = 'system'
    solver = 'newton'
  []
[]

[Solvers]
  [newton]
    type = Newton
    rel_tol = 1e-08
    abs_tol = 1e-10
    max_its = 50
  []
[]
