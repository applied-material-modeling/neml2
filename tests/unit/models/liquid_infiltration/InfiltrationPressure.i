[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/alpha_ratio'
      input_Scalar_values = 'arat'
      output_Scalar_names = 'state/P'
      output_Scalar_values = 'P'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [arat]
        type = Scalar
        values = "0.1 0.5 0.9"
        batch_shape = '(3)'
    []
    [P]
        type = Scalar
        values = "2.2222222 4 20"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = InfiltrationPressure
        initial_pressure = 2
        alpha_alphamax_ratio = 'state/alpha_ratio'
        pressure = 'state/P'
    []
[]