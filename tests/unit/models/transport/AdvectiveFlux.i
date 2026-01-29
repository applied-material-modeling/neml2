[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/u state/v'
    input_Scalar_values = 'u v'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
    input_with_intrsc_intmd_dims = 'state/u state/v'
    input_intrsc_intmd_dims = '1 1'
    output_with_intrsc_intmd_dims = 'state/J'
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
  [v]
    type = Scalar
    values = '-6 -2 4'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [J]
    type = Scalar
    values = '-8 2'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = AdvectiveFlux
    u = 'state/u'
    v = 'state/v'
    flux = 'state/J'
  []
[]
