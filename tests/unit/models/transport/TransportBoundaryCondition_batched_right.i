[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/u_bar state/u_bc'
    input_Scalar_values = 'u_bar u_bc'
    output_Scalar_names = 'state/u_bar_with_bc'
    output_Scalar_values = 'u_bar_with_bc'
    input_with_intrsc_intmd_dims = 'state/u_bar state/u_bc'
    input_intrsc_intmd_dims = '1 1'
    output_with_intrsc_intmd_dims = 'state/u_bar_with_bc'
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
  [u_bar_with_bc]
    type = Scalar
    values = '1 2 3 10 4 5 6 20'
    batch_shape = '(2,4)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = TransportBoundaryCondition
    input = 'state/u_bar'
    bc_value = 'state/u_bc'
    side = 'right'
  []
[]
