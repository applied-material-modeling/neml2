
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'degrade'
    input_Scalar_names = 'state/d'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'state/g'
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
    phase = 'state/d'
    degradation = 'state/g'
    power = 'p'
    fitting_param_1 = 1
    fitting_param_2 = 1.3868
    fitting_param_3 = 0.6567
  []
[]