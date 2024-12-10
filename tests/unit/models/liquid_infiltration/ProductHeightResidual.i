[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/delta state/h old_state/h state/alpha forces/t old_forces/t'
      input_scalar_values = 'delta h hn alpha t tn'
      output_scalar_names = 'residual/r_h'
      output_scalar_values = 'r_h'
      check_derivatives = true
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
        Product_Thickness_Growth_Ratio = 0.576
        Initial_Porosity = 0.5
        Product_Thickness = 'state/delta'
        Inlet_Gap = 'state/r1'
    []
    [product_growth]
        type = ProductGrowthWithLiquid
        Liquid_Molar_Volume = 0.00001256
        Product_Height = 'state/h'
        Inlet_Gap = 'state/r1'
        Liquid_Saturation = 'state/alpha'
        Phi_Condition = 'state/pcond'
    []
    [hdot]
        type = ScalarVariableRate
        variable = 'state/h'
        time = 'forces/t'
        rate = 'state/hdot'
    []
    [model0]
        type = FischerBurmeister
        Condition_A = 'state/pcond'
        Condition_B = 'state/hdot'
        Fischer_Burmeister = 'residual/r_h'
    []
    [model]
        type = ComposedModel
        models = 'model0 inlet_gap product_growth hdot'
    []
[]