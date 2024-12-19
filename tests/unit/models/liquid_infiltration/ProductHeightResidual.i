[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/delta state/h old_state/h state/alpha forces/t old_forces/t'
      input_Scalar_values = 'delta h hn alpha t tn'
      output_Scalar_names = 'residual/r_h'
      output_Scalar_values = 'r_h'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [delta]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [h]
        type = Scalar
        values = "0.1 0.5 0.2"
        batch_shape = '(3)'
    []
    [hn]
        type = Scalar
        values = "0.0 0.45 0.77"
        batch_shape = '(3)'
    []
    [alpha]
        type = Scalar
        values = "10000.0 1.0 500000.0"
        batch_shape = '(3)'
    []
    [t]
        type = Scalar
        values = "1000.0 5.0 108 "
        batch_shape = '(3)'
    []
    [tn]
        type = Scalar
        values = "0 0 0"
        batch_shape = '(3)'
    []
    [r_h]
        type = Scalar
        values = "-0.1527226 0.0096849195 -12.519456"
        batch_shape = '(3)'
    []
[]

[Models]
    [inlet_gap]
        type = InletGap
        product_thickness_growth_ratio = 0.576
        initial_porosity = 0.5
        product_thickness = 'state/delta'
        inlet_gap = 'state/r1'
    []
    [product_growth]
        type = ProductGrowthWithLiquid
        liquid_molar_volume = 0.00001256
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        liquid_saturation = 'state/alpha'
        phi_condition = 'state/pcond'
    []
    [hdot]
        type = ScalarVariableRate
        variable = 'state/h'
        time = 'forces/t'
        rate = 'state/hdot'
    []
    [model0]
        type = FischerBurmeister
        condition_a = 'state/pcond'
        condition_b = 'state/hdot'
        fischer_burmeister = 'residual/r_h'
    []
    [model]
        type = ComposedModel
        models = 'model0 inlet_gap product_growth hdot'
    []
[]