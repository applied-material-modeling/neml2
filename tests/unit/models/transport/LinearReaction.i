[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/lambda state/u'
    input_Scalar_values = 'lambda u'
    output_Scalar_names = 'state/R'
    output_Scalar_values = 'R'
    input_with_intrsc_intmd_dims = 'state/lambda state/u'
    input_intrsc_intmd_dims = '1 1'
    output_with_intrsc_intmd_dims = 'state/R'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [lambda]
    type = Scalar
    values = '2 4 6'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [u]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [R]
    type = Scalar
    values = '-2 -8 -18'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = LinearReaction
    lambda = 'state/lambda'
    u = 'state/u'
    reaction = 'state/R'
  []
[]
