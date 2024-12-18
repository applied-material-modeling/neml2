[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/in state/switch state/mreact state/current'
      input_Scalar_values = 'in switch mreact current'
      output_Scalar_names = 'residual/total'
      output_Scalar_values = 'total'
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
        in = 'state/in'
        switch = 'state/switch'
        minus_reaction = 'state/mreact'
        current = 'state/current'
        total = 'residual/total'
    []
[]