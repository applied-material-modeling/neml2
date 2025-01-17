[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/r1 state/alpha params/oL'
      input_Scalar_values = 'rr alpha oLiquid'
      output_Scalar_names = 'state/hL'
      output_Scalar_values = 'hL'
      check_derivatives = true
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [oLiquid]
        type = Scalar
        values = "0.7 12 3"
        batch_shape = '(3)'
    []
    [alpha]
        type = Scalar
        values = "10000 1 500000"
        batch_shape = '(3)'
    []
    [rr]
        type = Scalar
        values = "0.7013 0.5631 0.3385"
        batch_shape = '(3)'
    []
    [hL]
        type = Scalar
        values = "14232.80051 37.84514655 13091032.86"
        batch_shape = '(3)'
    []
[]

[Models]
    [omega_L]
        type = ScalarInputParameter
        from = 'params/oL'
    []
    [model0]
        type = LiquidSpan
        liquid_molar_volume = 'omega_L'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        liquid_span = 'state/hL'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]