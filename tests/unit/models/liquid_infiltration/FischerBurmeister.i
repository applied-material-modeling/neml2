[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/a state/b'
      input_Scalar_values = 'var1 var2'
      output_Scalar_names = 'state/fb'
      output_Scalar_values = 'fbcond'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [var1]
        type = Scalar
        values = "0.5713 12.579 -5.6668"
        batch_shape = '(3)'
    []
    [var2]
        type = Scalar
        values = "11.220 23.12 44.445"
        batch_shape = '(3)'
    []
    [fbcond]
        type = Scalar
        values = "0.55676469 9.3785585 -6.0266061"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = FischerBurmeister
        condition_a = 'state/a'
        condition_b = 'state/b'
        fischer_burmeister = 'state/fb'
    []
[]