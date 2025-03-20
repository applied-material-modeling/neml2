[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/a state/b'
      input_Scalar_values = 'a b'
      output_Scalar_names = 'state/out'
      output_Scalar_values = 'out'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [a]
        type = Scalar
        values = "100 52300 -7"
        batch_shape = '(3)'
    []
    [b]
        type = Scalar
        values = "0.1 123 44"
        batch_shape = '(3)'
    []
    [out]
        type = Scalar
        values = "1000 425.2032520325 -0.1590909090"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = Ratio
        numerator = 'state/a'
        denominator = 'state/b'
        out = 'state/out'
    []
[]