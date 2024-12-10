[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_scalar_names = 'state/delta/' #params/M params/phi0'
      input_scalar_values = 'dd'
      output_scalar_names = 'state/r1'
      output_scalar_values = 'rr'
      check_derivatives = true
  []
[]

[Tensors]
    [dd]
        type = Scalar
        values = "0.1 0.5 0.8"
        batch_shape = '(3)'
    []
    [rr]
        type = Scalar
        values = "0.70134678 0.56310678 0.33846678"
        batch_shape = '(3)'
    []
[]

[Models]
    [model]
        type = InletGap
        Product_Thickness_Growth_Ratio = 0.576
        Initial_Porosity = 0.5
        Product_Thickness = 'state/delta'
        Inlet_Gap = 'state/r1'
    []
[]