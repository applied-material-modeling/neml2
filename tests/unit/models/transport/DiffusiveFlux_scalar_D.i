[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u state/D_edge'
    input_Scalar_values = 'u D_edge'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
    input_with_intrsc_intmd_dims = 'state/u'
    input_intrsc_intmd_dims = '1'
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
  [J]
    type = Scalar
    values = '-2 -4'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [D_edge]
    type = Scalar
    values = 2.0
  []
[]

[Models]
  [model]
    type = DiffusiveFlux
    jit = false
    u = 'state/u'
    D_edge = 'state/D_edge'
    flux = 'state/J'
  []
[]
