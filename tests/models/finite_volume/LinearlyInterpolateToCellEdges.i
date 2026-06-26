# Translated from tests/unit/models/finite_volume/LinearlyInterpolateToCellEdges.i.
# cell_values is promoted to a runtime input (mode 4) so the JVP check has a
# tangent direction; cell_centers / cell_edges stay as static [Tensors] params.
# All Scalars carry the cell-axis sub-batch (sub_batch_ndim=1).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'q_cell'
    input_Scalar_values = 'q_cell_in'
    output_Scalar_names = 'q_edge'
    output_Scalar_values = 'q_edge_out'
  []
[]

[Tensors]
  [q_cell_in]
    type = Python
    expr = 'Scalar(torch.tensor([2.0, 4.0, 7.0]), sub_batch_ndim=1)'
  []
  [cell_centers]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 0.5, 1.5]), sub_batch_ndim=1)'
  []
  [cell_edges]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 0.2, 1.0, 2.0]), sub_batch_ndim=1)'
  []
  [q_edge_out]
    type = Python
    expr = 'Scalar(torch.tensor([2.8, 5.5]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'q_cell'
    cell_centers = 'cell_centers'
    cell_edges = 'cell_edges'
    edge_values = 'q_edge'
  []
[]
