[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model1'
      input_Scalar_names = 'state/delta'
      input_Scalar_values = 'delta'
      output_Scalar_names = 'state/dratio'
      output_Scalar_values = 'dratio1'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [M]
        type = Scalar
        values = "0.1 0.576 0.8"
        batch_shape = '(3)'
    []
    [delta]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [dratio1]
        type = Scalar
        values = "0.030727922061358 0.361906637611548 0.724077343935025"
        batch_shape = '(3)'
    []
[]

[Models]
    [model1]
        type = ProductThicknessLimit
        initial_porosity = 0.5
        product_thickness_growth_ratio = 'M'
        product_thickness = 'state/delta'
        limit_ratio = 'state/dratio'
    []
[]