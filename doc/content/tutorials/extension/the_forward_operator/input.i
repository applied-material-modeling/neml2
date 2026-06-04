[Tensors]
  [g]
    type = Python
    expr = 'Vec.fill(0.0, -9.81, 0.0)'
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
