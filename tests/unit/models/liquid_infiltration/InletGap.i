[Drivers]
    [unit]
      type = ModelUnitTest
      model = 'model'
      input_Scalar_names = 'state/delta/' #params/M params/phi0'
      input_Scalar_values = 'dd'
      output_Scalar_names = 'state/r1'
      output_Scalar_values = 'rr'
      check_AD_parameter_derivatives = false
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
        product_thickness_growth_ratio = 0.576
        initial_porosity = 0.5
        product_thickness = 'state/delta'
        inlet_gap = 'state/r1'
    []
[]