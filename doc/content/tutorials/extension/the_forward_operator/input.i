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
    velocity = 'v'
    acceleration = 'a'
    gravitational_acceleration = 'g'
    dynamic_viscosity = 'mu'
  []
[]
