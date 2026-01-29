[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u_bar'
    input_Scalar_values = 'u_bar'
    output_Scalar_names = 'state/u_bar_with_bc_left'
    output_Scalar_values = 'u_bar_with_bc_left'
    input_with_intrsc_intmd_dims = 'state/u_bar'
    input_intrsc_intmd_dims = '1'
    output_with_intrsc_intmd_dims = 'state/u_bar_with_bc_left'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [u_bar]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [u_bc]
    type = Scalar
    values = '10'
    batch_shape = '(1)'
    intermediate_dimension = 1
  []
  [u_bar_with_bc_left]
    type = Scalar
    values = '10 1 2 3'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = TransportBoundaryCondition
    input = 'state/u_bar'
    bc_value = 'u_bc'
    side = 'left'
  []
[]
