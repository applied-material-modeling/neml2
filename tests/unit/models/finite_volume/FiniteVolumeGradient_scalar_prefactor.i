[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u'
    input_Scalar_values = 'u'
    output_Scalar_names = 'state/grad_u'
    output_Scalar_values = 'grad_u'
    input_with_intrsc_intmd_dims = 'state/u'
    input_intrsc_intmd_dims = '1'
    output_with_intrsc_intmd_dims = 'state/grad_u'
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
  [grad_u]
    type = Scalar
    values = '-2 -4'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [dx]
    type = Scalar
    values = '1 1'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
  [prefactor]
    type = Scalar
    values = 2.0
  []
[]

[Models]
  [model]
    type = FiniteVolumeGradient
    jit = false
    u = 'state/u'
    prefactor = 'prefactor'
    dx = 'dx'
    grad_u = 'state/grad_u'
  []
[]
