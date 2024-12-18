[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/var'
      input_Scalar_values = 'in'
      output_Scalar_names = 'state/out'
      output_Scalar_values = 'out'
      check_AD_parameter_derivatives = false
  []
[]

[Tensors]
    [in]
        type = Scalar
        values = "1.5 0.01 0.97"
        batch_shape = '(3)'
    []
    [out]
        type = Scalar
        values = "0.0000453978687 0.9999999975 0.6456563062"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = SwitchingFunction
        smooth_degree = 10.0
        variable = 'state/var'
        switch_out = 'state/out'
    []
[]