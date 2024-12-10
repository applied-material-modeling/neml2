[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/var'
      input_scalar_values = 'in'
      output_scalar_names = 'state/out'
      output_scalar_values = 'out'
      check_derivatives = true
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
        Smooth_Degree = 10.0
        variable = 'state/var'
        switch_out = 'state/out'
    []
[]