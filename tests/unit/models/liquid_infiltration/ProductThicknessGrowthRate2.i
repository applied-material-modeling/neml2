[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/ddot state/dideal state/switch state/d'
      input_Scalar_values = 'ddot dideal switch d'
      output_Scalar_names = 'residual/rdelta'
      output_Scalar_values = 'rdelta'
      check_AD_parameter_derivatives = false
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [ddot]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [d]
        type = Scalar
        values = "10"
        batch_shape = '(3)'
    []
    [dideal]
        type = Scalar
        values = "10 1 5.773"
        batch_shape = '(3)'
    []
    [switch]
        type = Scalar
        values = "0.7013 0.5631 0.3385"
        batch_shape = '(3)'
    []
    [rdelta]
        type = Scalar
        values = "-6.913 -0.0631 -1.1541605"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = ProductThicknessGrowthRate
        thickness_rate = 'state/ddot'
        scale = 'state/d'
        scaling_condition = false
        ideal_thickness_growth = 'state/dideal'
        switch = 'state/switch'
        residual_delta = 'residual/rdelta'
    []
[]