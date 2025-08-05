[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'E'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = 'Tval'
    output_Scalar_names = 'parameters/E'
    output_Scalar_values = 'should'
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
    check_exact_shapes = true
  []
[]

[Models]
  [E]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_control'
    ordinate = 'E_values'
  []
[]

[Tensors]
  [T_control]
    type = Scalar
    values = '0 100'
    batch_shape = '(2)'
  []
  [E_values]
    type = Scalar
    values = '8.0 10.0'
    batch_shape = '(1,2)'
  []
  [should]
    type = FullScalar
    batch_shape = '(7,8,1)'
    value = 9.0
  []

  [Tval]
    type = FullScalar
    batch_shape = '(7,8)'
    value = 50.0
  []
[]
