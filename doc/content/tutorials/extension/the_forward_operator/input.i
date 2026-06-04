[Tensors]
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
    dynamic_viscosity = 'mu'
  []
[]
