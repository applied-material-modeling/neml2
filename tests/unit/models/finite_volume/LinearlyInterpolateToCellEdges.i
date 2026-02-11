[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    check_AD_parameter_derivatives = false
    input_Scalar_names = ''
    input_Scalar_values = ''
    output_Scalar_names = 'state/q_edge'
    output_Scalar_values = 'q_edge'
    output_with_intrsc_intmd_dims = 'state/q_edge'
    output_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [cell_values]
    type = Scalar
    values = '2 4 7'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [cell_centers]
    type = Scalar
    values = '0.0 0.5 1.5'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [cell_edges]
    type = Scalar
    values = '0.0 0.2 1.0 2.0'
    batch_shape = '(4)'
    intermediate_dimension = 1
  []
  [q_edge]
    type = Scalar
    values = '2.8 5.5'
    batch_shape = '(2)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'cell_values'
    cell_centers = 'cell_centers'
    cell_edges = 'cell_edges'
    edge_values = 'state/q_edge'
  []
[]
