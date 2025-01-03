[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/h state/r1 state/alpha params/oL'
      input_Scalar_values = 'h rr alpha oLiquid'
      output_Scalar_names = 'state/pcond'
      output_Scalar_values = 'phi_cond'
      check_derivatives = true
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [h]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
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
    [phi_cond]
        type = Scalar
        values = "-6999.9508112692519717 -11.841455376490863216 -1499999.9083521903958"
        batch_shape = '(3)'
    []
[]

[Models]
    [omega_L]
        type = ScalarInputParameter
        from = 'params/oL'
    []
    [model0]
        type = ProductGrowthWithLiquid
        liquid_molar_volume = 'omega_L'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        phi_condition = 'state/pcond'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]