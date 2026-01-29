[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/dislocation_density'
    input_Scalar_values = 'rho'
    output_Scalar_names = 'state/internal/slip_strengths'
    output_Scalar_values = 'tau'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
    second_derivative_rel_tol = 0
    second_derivative_abs_tol = 5e-6
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [rho]
    type = Scalar
    values = '1 4 9 16'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [tau]
    type = Scalar
    values = '15 25 35 45'
    batch_shape = '(4)'
    intermediate_dimension = 1
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
