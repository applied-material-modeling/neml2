[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/u state/D'
    input_Scalar_values = 'u D'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
    input_with_intrsc_intmd_dims = 'state/u state/D'
    input_intrsc_intmd_dims = '1 1'
    output_with_intrsc_intmd_dims = 'state/J'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [u]
    type = Scalar
    values = '1 2 4'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [D]
    type = Scalar
    values = '2 4 6'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [J]
    type = Scalar
    values = '-3 -10'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = DiffusiveFlux
    u = 'state/u'
    D = 'state/D'
    flux = 'state/J'
  []
[]
