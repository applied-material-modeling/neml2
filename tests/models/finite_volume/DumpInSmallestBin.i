# Translated from tests/unit/models/finite_volume/DumpInSmallestBin.i.
# magnitude is a global Scalar (no sub-batch); cell_centers and the
# dumped_source output carry sub_batch_ndim=1 (3 cells).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mag'
    input_Scalar_values = 'mag'
    output_Scalar_names = 'state/source'
    output_Scalar_values = 'source'
  []
[]

[Tensors]
  [mag]
    type = Python
    expr = 'Scalar(torch.tensor(2.5))'
  []
  [centers]
    type = Python
    expr = 'Scalar(torch.tensor([0.1, 0.5, 1.0]), sub_batch_ndim=1)'
  []
  [source]
    type = Python
    expr = 'Scalar(torch.tensor([2.5, 0.0, 0.0]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = DumpInSmallestBin
    magnitude = 'state/mag'
    cell_centers = 'centers'
    dumped_source = 'state/source'
  []
[]
