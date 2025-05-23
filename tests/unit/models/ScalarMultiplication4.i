[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/B state/C state/D'
    input_Scalar_values = '4.8 9.2 56.8 0.8'
    output_Scalar_names = 'state/E'
    output_Scalar_values = '0.040493'
    check_second_derivatives = true
    second_derivative_abs_tol = 2e-8
  []
[]

[Models]
  [model]
    type = ScalarMultiplication
    from_var = 'state/A state/B state/C state/D'
    reciprocal = 'true false true false'
    to_var = 'state/E'
    coefficient = 1.5
  []
[]
