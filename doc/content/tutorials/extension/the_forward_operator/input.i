[Tensors]
  [g]
    type = Python
    expr = 'Vec(torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64))'
  []
  [mu]
    type = Python
    expr = 'Scalar(0.001)'
  []
[]

[Models]
  [accel]
    type = ProjectileAcceleration
    velocity = 'v'
    acceleration = 'a'
    gravitational_acceleration = 'g'
    dynamic_viscosity = 'mu'
  []
[]
