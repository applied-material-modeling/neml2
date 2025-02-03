[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/r1 state/alpha params/oL'
      input_Scalar_values = 'r1 alpha 1.2'
      output_Scalar_names = 'state/alphamax'
      output_Scalar_values = 'max'
      check_derivatives = true
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [r1]
        type = Scalar
        values = "0.1 0.5 0.2"
        batch_shape = '(3)'
    []
    [alpha]
        type = Scalar
        values = "0.008333333 0.20833333 0.03333333"
        batch_shape = '(3)'
    []
    [max]
        type = Scalar
        values = "1.0"
        batch_shape = '(3)'
    []
[]

[Models]
    [omega_L]
        type = ScalarInputParameter
        from = 'params/oL'
    []
    [model0]
        type = LiquidSaturationLimitRatio
        liquid_molar_volume = 'omega_L'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        limit_saturation_ratio = 'state/alphamax'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]