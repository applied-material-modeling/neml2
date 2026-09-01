# N=3 fragment flux operator. Uniform density, dv=1, gamma_0=0 (sink bin),
# upper-triangular daughter matrix p. The (N-1, N) = (2, 3) reference M is
# hand-computed from K_{kj} = dv_j dv_k gamma_j (rho_k v_k)/(rho_j v_j) p_{kj},
# M_{ij} = -sum_{k<=i} K_{kj} for j>i. ModelUnitTest auto-checks all five
# analytic derivatives against autograd.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'rho v dv gamma p'
    input_Scalar_values = 'rho v dv gamma p'
    output_Scalar_names = 'M'
    output_Scalar_values = 'M'
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0, 1.0]), sub_batch_ndim=1)'
  []
  [v]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 2.0, 3.0]), sub_batch_ndim=1)'
  []
  [dv]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0, 1.0]), sub_batch_ndim=1)'
  []
  [gamma]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 1.0, 2.0]), sub_batch_ndim=1)'
  []
  [p]
    type = Python
    expr = 'Scalar(torch.tensor([[0.5, 0.7, 0.3], [0.0, 0.4, 0.6], [0.0, 0.0, 0.2]]), sub_batch_ndim=2)'
  []
  [M]
    type = Python
    expr = 'Scalar(torch.tensor([[0.0, -0.35, -0.2], [0.0, 0.0, -1.0]]), sub_batch_ndim=2)'
  []
[]

[Models]
  [model]
    type = FiniteVolumeFragmentationFlux
    cell_density = 'rho'
    cell_volume = 'v'
    cell_width = 'dv'
    fragmentation_rate = 'gamma'
    breakage_matrix = 'p'
    flux_operator = 'M'
  []
[]
