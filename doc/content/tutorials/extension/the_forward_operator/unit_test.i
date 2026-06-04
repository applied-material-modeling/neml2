[Tensors]
  [g]
    type = Python
    expr = 'Vec.fill(0.0, -9.81, 0.0)'
  []
  [mu]
    type = Python
    expr = 'Scalar(0.001)'
  []
  [v_in]
    type = Python
    expr = 'Vec.fill(10.0, 2.0, 0.0)'
  []
  [a_expected]
    type = Python
    expr = 'Vec.fill(-0.01, -9.812, 0.0)'
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

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'accel'
    input_Vec_names = 'v'
    input_Vec_values = 'v_in'
    output_Vec_names = 'a'
    output_Vec_values = 'a_expected'
  []
[]
