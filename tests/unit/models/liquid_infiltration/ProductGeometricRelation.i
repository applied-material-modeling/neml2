[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/h state/r1 state/delta params/oP'
      input_Scalar_values = 'h rr delta oProduct'
      output_Scalar_names = 'state/alphaP'
      output_Scalar_values = 'alphaP'
      check_derivatives = true
  []
[]

[Tensors]
    [h]
        type = Scalar
        values = "0.1 0.5 0.2"
        batch_shape = '(3)'
    []
    [oProduct]
        type = Scalar
        values = "12.5 1.0 0.256"
        batch_shape = '(3)'
    []
    [delta]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [rr]
        type = Scalar
        values = "0.7013 0.5631 0.3385"
        batch_shape = '(3)'
    []
    [alphaP]
        type = Scalar
        values = "0.000113008 0.172025 0.6585"
        batch_shape = '(3)'
    []
[]

[Models]
    [omega_P]
        type = ScalarInputParameter
        from = 'params/oP'
    []
    [model0]
        type = ProductGeometricRelation
        product_molar_volume = 'omega_P'
        product_height = 'state/h'
        inlet_gap = 'state/r1'
        product_thickness = 'state/delta'
        product_saturation = 'state/alphaP'
    []
    [model]
        type = ComposedModel
        models = 'model0'
    []
[]