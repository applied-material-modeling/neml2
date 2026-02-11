[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u state/v_edge'
    input_Scalar_values = 'u v_edge'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
    input_with_intrsc_intmd_dims = 'state/u state/v_edge'
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
  [v_edge]
    type = Scalar
    values = '-4 3'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [J]
    type = Scalar
    values = '-8 6'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = AdvectiveFlux
    u = 'state/u'
    v_edge = 'state/v_edge'
    flux = 'state/J'
  []
[]
