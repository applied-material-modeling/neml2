# Translated from tests/unit/models/kwn/NucleationFluxMagnitude.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/Z state/beta state/dg state/T'
    input_Scalar_values = 'Z beta dg T'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 'J'
  []
[]

[Tensors]
  [Z]
    type = Python
    expr = "Scalar(torch.tensor(2.0, dtype=torch.float64))"
  []
  [beta]
    type = Python
    expr = "Scalar(torch.tensor(3.0, dtype=torch.float64))"
  []
  [dg]
    type = Python
    expr = "Scalar(torch.tensor(4.0, dtype=torch.float64))"
  []
  [T]
    type = Python
    expr = "Scalar(torch.tensor(4.0, dtype=torch.float64))"
  []
  [J]
    type = Python
    expr = "Scalar(torch.tensor(18.195919791379007, dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = NucleationFluxMagnitude
    zeldovich_factor = 'state/Z'
    kinetic_factor = 'state/beta'
    nucleation_barrier = 'state/dg'
    temperature = 'state/T'
    nucleation_site_density = 5.0
    boltzmann_constant = 2.0
    nucleation_flux_magnitude = 'state/J'
  []
[]
