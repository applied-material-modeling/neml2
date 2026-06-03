# Translated from tests/unit/models/kwn/PrecipitateVolumeFraction.i.
# R and n are per-bin Scalars with sub_batch_ndim=1 (3 size bins); the output
# f reduces across the bin axis to a sub_batch_ndim=0 Scalar.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/n'
    input_Scalar_values = 'n'
    output_Scalar_names = 'state/f'
    output_Scalar_values = 'f'
  []
[]

[Tensors]
  [R]
    type = Python
    expr = "Scalar(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), sub_batch_ndim=1)"
  []
  [n]
    type = Python
    expr = "Scalar(torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64), sub_batch_ndim=1)"
  []
  [f]
    type = Python
    expr = "Scalar(torch.tensor(4105.014400690663, dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = PrecipitateVolumeFraction
    radius = 'R'
    number_density = 'state/n'
    volume_fraction = 'state/f'
  []
[]
