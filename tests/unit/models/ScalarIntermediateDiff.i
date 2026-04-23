[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A'
    input_Scalar_values = 'A'
    output_Scalar_names = 'B'
    output_Scalar_values = 'B'
    input_with_intrsc_intmd_dims = 'A'
    input_intrsc_intmd_dims = '1'
    output_with_intrsc_intmd_dims = 'B'
    output_intrsc_intmd_dims = '1'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [A]
    type = Scalar
    values = "1 2 4
              3 5 9
              -1 -2 -4
              10 10 10"
    batch_shape = '(4,3)'
    intermediate_dimension = 1
  []
  [B]
    type = Scalar
    values = "1 2
              2 4
              -1 -2
              0 0"
    batch_shape = '(4,2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = ScalarIntermediateDiff
    from = 'A'
    to = 'B'
    dim = -1
  []
[]
