
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'degrade'
    input_Scalar_names = 'state/d'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'state/g'
    output_Scalar_values = '0.000438428' 
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [p]
    type = Scalar
    values = 5
  []
[]

[Models]
  [degrade]
    type = PowerDegradationFunction
    phase = 'state/d'
    degradation = 'state/g'
    power = 'p'
  []
[]