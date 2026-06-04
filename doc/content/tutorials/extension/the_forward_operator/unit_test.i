[Tensors]
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
    dynamic_viscosity = '0.001'
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
