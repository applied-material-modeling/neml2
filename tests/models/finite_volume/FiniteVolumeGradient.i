# Translated from tests/unit/models/finite_volume/FiniteVolumeGradient.i.
# Scalar values with intermediate_dimension=1 -> sub_batch_ndim=1; u has 3 cells,
# grad_u/prefactor/dx are per-edge (2).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'u'
    input_Scalar_values = 'u'
    output_Scalar_names = 'grad_u'
    output_Scalar_values = 'grad_u'
  []
[]

[Tensors]
  [u]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 4.0]), sub_batch_ndim=1)'
  []
  [prefactor]
    type = Python
    expr = 'Scalar(torch.tensor([3.0, 5.0]), sub_batch_ndim=1)'
  []
  [dx]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0]), sub_batch_ndim=1)'
  []
  [grad_u]
    type = Python
    expr = 'Scalar(torch.tensor([-3.0, -10.0]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = FiniteVolumeGradient
    u = 'u'
    prefactor = 'prefactor'
    dx = 'dx'
    grad_u = 'grad_u'
  []
[]
