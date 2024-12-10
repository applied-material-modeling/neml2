[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/ddot state/scale state/dideal state/switch'
      input_scalar_values = 'ddot scale dideal switch'
      output_scalar_names = 'residual/rdelta'
      output_scalar_values = 'rdelta'
      check_derivatives = true
      derivative_abs_tol = 1e-4
  []
[]

[Tensors]
    [ddot]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [scale]
        type = Scalar
        values = "0.7 12 3"
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
        values = "-4.8091 -6.2572 -5.0624815"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = ProductWidthGrowthRate
        Thickness_Rate = 'state/ddot'
        Scale = 'state/scale'
        Ideal_Thickness_Growth = 'state/dideal'
        Switch = 'state/switch'
        Residual_Delta = 'residual/rdelta'
    []
[]