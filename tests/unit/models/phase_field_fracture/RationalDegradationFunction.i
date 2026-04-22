[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'degrade'
    input_Scalar_names = 'd'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'g'
    output_Scalar_values = '0.0212478'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [p]
    type = Scalar
    values = 2
  []
[]

[Models]
  [degrade]
    type = RationalDegradationFunction
    phase = 'd'
    degradation = 'g'
    power = 'p'
    b1 = 1
    b2 = 1.3868
    b3 = 0.6567
  []
[]
