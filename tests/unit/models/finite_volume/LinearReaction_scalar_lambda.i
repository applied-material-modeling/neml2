[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u'
    input_Scalar_values = 'u'
    output_Scalar_names = 'state/R'
    output_Scalar_values = 'R'
    input_with_intrsc_intmd_dims = 'state/u'
    input_intrsc_intmd_dims = '1'
    output_with_intrsc_intmd_dims = 'state/R'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [u]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [R]
    type = Scalar
    values = '-2 -4 -6'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = LinearReaction
    lambda = 2.0
    u = 'state/u'
    reaction = 'state/R'
  []
[]
