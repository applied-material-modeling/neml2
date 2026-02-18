[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/u_bar'
    input_Scalar_values = 'u_bar'
    output_Scalar_names = 'state/u_bar_with_bc_right'
    output_Scalar_values = 'u_bar_with_bc_right'
    input_with_intrsc_intmd_dims = 'state/u_bar'
    input_intrsc_intmd_dims = '1'
    output_with_intrsc_intmd_dims = 'state/u_bar_with_bc_right'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [u_bar]
    type = Scalar
    values = '1 2 3 4 5 6'
    batch_shape = '(2,3)'
    intermediate_dimension = 1
  []
  [u_bc]
    type = Scalar
    values = '10 20'
    batch_shape = '(2,1)'
    intermediate_dimension = 1
  []
  [u_bar_with_bc_right]
    type = Scalar
    values = '1 2 3 10 4 5 6 20'
    batch_shape = '(2,4)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'state/u_bar'
    bc_value = 'u_bc'
    side = 'right'
  []
[]
