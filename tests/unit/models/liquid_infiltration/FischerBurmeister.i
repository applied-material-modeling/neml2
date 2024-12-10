[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/a state/b'
      input_scalar_values = 'var1 var2'
      output_scalar_names = 'state/fb'
      output_scalar_values = 'fbcond'
      check_derivatives = true
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
        Condition_A = 'state/a'
        Condition_B = 'state/b'
        Fischer_Burmeister = 'state/fb'
    []
[]