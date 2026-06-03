# Translated from tests/unit/models/porous_flow/KozenyCarmanPermeability.i.
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
    expr = "Scalar(torch.tensor([0.01, 0.65, 0.98], dtype=torch.float64))"
  []
  [phi0]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.35, 1.0], dtype=torch.float64))"
  []
  [m]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.4, 1.8], dtype=torch.float64))"
  []
  [K]
    type = Python
    expr = "Scalar(torch.tensor([0.07596492249, 6.305733751, 0.0], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = KozenyCarmanPermeability
    reference_permeability = 3
    reference_porosity = 'phi0'
    n = 0.8
    m = 'm'
    porosity = 'phi'
    permeability = 'K'
  []
[]
