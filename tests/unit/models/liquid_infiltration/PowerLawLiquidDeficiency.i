[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/hP state/hL'
      input_Scalar_values = 'hP hL'
      output_Scalar_names = 'state/scale'
      output_Scalar_values = 'scale'
      check_derivatives = true
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [hP]
        type = Scalar
        values = "0.1 0.5 0.2"
        batch_shape = '(3)'
    []
    [hL]
        type = Scalar
        values = "14232.80051 37.84514655 13091032.86"
        batch_shape = '(3)'
    []
    [scale]
        type = Scalar
        values = "13262.7105 31.85900071 1789596.55"
        batch_shape = '(3)'
    []
    [p]
        type = Scalar
        values = "0.8"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = PowerLawLiquidDeficiency
        product_span = 'state/hP'
        liquid_span = 'state/hL'
        exponent = 'p'
        scale = 'state/scale'
    []
[]