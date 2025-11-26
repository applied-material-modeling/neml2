[Tensors]
  [g]
    type = Vec
    values = '0 -9.81 0'
  []
  [mu]
    type = Scalar
    values = 0.001
  []
[]

[Models]
  [accel]
    type = ProjectileAcceleration
    velocity = 'state/v'
    acceleration = 'state/a'
    gravitational_acceleration = 'g'
    dynamic_viscosity = 'mu'
  []
[]
