[Tensors]
  [g]
    type = Python
    expr = 'Vec(torch.tensor([0.0, -9.81, 0.0], dtype=torch.float64))'
  []
  [mu]
    type = Python
    expr = 'Scalar(torch.tensor(0.001, dtype=torch.float64))'
  []
  [v_in]
    type = Python
    expr = 'Vec(torch.tensor([10.0, 2.0, 0.0], dtype=torch.float64))'
  []
  [a_expected]
    type = Python
    expr = 'Vec(torch.tensor([-0.01, -9.812, 0.0], dtype=torch.float64))'
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
