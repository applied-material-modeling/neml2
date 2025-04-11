[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/S'
      input_Scalar_values = 'S'
      output_Scalar_names = 'state/Pc'
      output_Scalar_values = 'Pc'
      check_AD_parameter_derivatives = false
      derivative_abs_tol = 1e-7
  []
[]

[Tensors]
    [S]
        type = Scalar
        values = "0.1 0.65 0.9"
        batch_shape = '(3)'
    []
    [Pc]
        type = Scalar
        values = "0.8840986388 0.3175165612 0.1932327962"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = VanGenuchtenPressure
        scaling_constant = 3
        power = 0.7
        effective_saturation = 'state/S'
        capillary_pressure = 'state/Pc'
    []
[]