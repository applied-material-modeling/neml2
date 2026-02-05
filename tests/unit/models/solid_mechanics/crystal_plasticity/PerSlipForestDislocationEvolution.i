[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/dislocation_density state/internal/slip_rates'
    input_Scalar_values = 'rho gamma_dot'
    output_Scalar_names = 'state/internal/dislocation_density_rate'
    output_Scalar_values = 'rho_dot'
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
  [gamma_dot]
    type = Scalar
    values = '0.1 -0.2 0.3 -0.4'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [rho_dot]
    type = Scalar
    values = '0.15 0.4 0.45 0.0'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = PerSlipForestDislocationEvolution
    k1 = 2.0
    k2 = 0.5
  []
[]
