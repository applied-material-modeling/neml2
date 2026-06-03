# Translated from tests/unit/models/finite_volume/FiniteVolumeUpwindedAdvectiveFlux.i.
# u is cell-averaged (3 cells); v_edge and the flux J are cell-edge (2 edges).
# Both inputs and the output carry sub_batch_ndim=1.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'u v_edge'
    input_Scalar_values = 'u v_edge'
    output_Scalar_names = 'J'
    output_Scalar_values = 'J'
  []
[]

[Tensors]
  [u]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 3.0]), sub_batch_ndim=1)'
  []
  [v_edge]
    type = Python
    expr = 'Scalar(torch.tensor([-4.0, 3.0]), sub_batch_ndim=1)'
  []
  [J]
    type = Python
    expr = 'Scalar(torch.tensor([-8.0, 6.0]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'u'
    v_edge = 'v_edge'
    flux = 'J'
  []
[]
