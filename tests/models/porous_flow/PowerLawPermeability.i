[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'phi'
    input_Scalar_values = 'phi'
    output_Scalar_names = 'K'
    output_Scalar_values = 'K'
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [phi]
    type = Python
    expr = 'Scalar(torch.tensor([0.01, 0.65, 0.98], dtype=torch.float64))'
  []
  [phi0]
    type = Python
    expr = 'Scalar(torch.tensor([0.71, 0.35, 1.0], dtype=torch.float64))'
  []
  [p]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 0.4, 1.8], dtype=torch.float64))'
  []
  [K]
    type = Python
    expr = 'Scalar(torch.tensor([3.0, 3.842902621, 2.892865159], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = PowerLawPermeability
    reference_permeability = 3
    reference_porosity = 'phi0'
    exponent = 'p'
    porosity = 'phi'
    permeability = 'K'
  []
[]
