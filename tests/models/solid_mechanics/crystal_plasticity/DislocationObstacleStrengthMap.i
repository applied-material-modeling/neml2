# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/DislocationObstacleStrengthMap.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density'
    input_Scalar_values = 'rho'
    output_Scalar_names = 'slip_strengths'
    output_Scalar_values = 'tau'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 4.0, 9.0, 16.0])).with_sub_batch(1)'
  []
  [tau]
    type = Python
    expr = 'Scalar(torch.tensor([15.0, 25.0, 35.0, 45.0])).with_sub_batch(1)'
  []
[]

[Models]
  [model]
    type = DislocationObstacleStrengthMap
    constant_strength = 5.0
    alpha = 0.5
    mu = 80.0
    b = 0.25
  []
[]
