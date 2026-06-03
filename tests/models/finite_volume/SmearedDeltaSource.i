# Translated from tests/unit/models/finite_volume/SmearedDeltaSource.i.
# magnitude / location are global Scalars (no sub-batch); cell_centers and the
# smeared_source output carry sub_batch_ndim=1 (3 cells).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/mag state/loc'
    input_Scalar_values = 'mag loc'
    output_Scalar_names = 'state/source'
    output_Scalar_values = 'source'
  []
[]

[Tensors]
  [mag]
    type = Python
    expr = 'Scalar(torch.tensor(2.0))'
  []
  [loc]
    type = Python
    expr = 'Scalar(torch.tensor(0.2))'
  []
  [centers]
    type = Python
    expr = 'Scalar(torch.tensor([-1.0, 0.0, 1.0]), sub_batch_ndim=1)'
  []
  [source]
    type = Python
    expr = 'Scalar(torch.tensor([0.3883721099664259, 0.7820853879509118, 0.5793831055229656]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = SmearedDeltaSource
    magnitude = 'state/mag'
    location = 'state/loc'
    width = 1.0
    cell_centers = 'centers'
    smeared_source = 'state/source'
  []
[]
