[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = 'state/J state/R'
    input_Scalar_values = 'J R'
    output_Scalar_names = 'state/u_rate'
    output_Scalar_values = 'u_rate'
    input_with_intrsc_intmd_dims = 'state/J state/R'
    input_intrsc_intmd_dims = '1 1'
    output_with_intrsc_intmd_dims = 'state/u_rate'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [J]
    type = Scalar
    values = '0 1 1 0'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [dx]
    type = Scalar
    values = '0.5 0.5 0.5'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [R]
    type = Scalar
    values = '0.1 0.2 0.3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [u_rate]
    type = Scalar
    values = '-1.9 0.2 2.3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = CellRateOfChange
    flux = 'state/J'
    cell_size = 'dx'
    reaction = 'state/R'
    rate = 'state/u_rate'
  []
[]
