[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B C D'
    input_Scalar_values = '3.1 2.5 4.6 7.8'
    output_Scalar_names = 'E'
    output_Scalar_values = '2.051192146'
    check_second_derivatives = true
    second_derivative_abs_tol = 2e-8
  []
[]

[Models]
  [model]
    type = ScalarMultiplication
    from = 'A B C D'
    to = 'E'
    scaling = 1.5
    reciprocal = 'true false true false'
  []
[]
