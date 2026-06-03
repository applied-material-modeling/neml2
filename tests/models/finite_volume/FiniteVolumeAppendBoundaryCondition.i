# Translated from tests/unit/models/finite_volume/FiniteVolumeAppendBoundaryCondition.i.
# bc_value is a Scalar parameter; the output name is auto-derived from input + side
# as 'u_bar_with_bc_left'. All Scalars carry the cell-axis sub-batch (sub_batch_ndim=1).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'u_bar'
    input_Scalar_values = 'u_bar'
    output_Scalar_names = 'u_bar_with_bc_left'
    output_Scalar_values = 'u_bar_with_bc_left'
  []
[]

[Tensors]
  [u_bar]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 3.0]), sub_batch_ndim=1)'
  []
  [u_bc]
    type = Python
    expr = 'Scalar(torch.tensor([10.0]), sub_batch_ndim=1)'
  []
  [u_bar_with_bc_left]
    type = Python
    expr = 'Scalar(torch.tensor([10.0, 1.0, 2.0, 3.0]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'u_bar'
    bc_value = 'u_bc'
    side = 'left'
  []
[]
