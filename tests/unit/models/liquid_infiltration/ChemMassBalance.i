[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/in state/switch state/mreact state/current'
      input_scalar_values = 'in switch mreact current'
      output_scalar_names = 'residual/total'
      output_scalar_values = 'total'
      check_derivatives = true
  []
[]

[Tensors]
    [current]
        type = Scalar
        values = "0 1000 200000"
        batch_shape = '(3)'
    []
    [switch]
        type = Scalar
        values = "1.0 0.5 0.0"
        batch_shape = '(3)'
    []
    [in]
        type = Scalar
        values = "1000 0.5 30"
        batch_shape = '(3)'
    []
    [mreact]
        type = Scalar
        values = "20.0 10.0 -5.0"
        batch_shape = '(3)'
    []
    [total]
        type = Scalar
        values = "-980 1009.75 199995"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = ChemMassBalance
        In = 'state/in'
        Switch = 'state/switch'
        Minus_Reaction = 'state/mreact'
        Current = 'state/current'
        Total = 'residual/total'
    []
[]