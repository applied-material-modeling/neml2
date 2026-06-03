# Translated from tests/unit/models/kwn/ChemicalGibbsFreeEnergyDifference.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/dx1 state/dx2'
    input_Scalar_values = 'dx1 dx2'
    output_Scalar_names = 'state/dg'
    output_Scalar_values = 'dg'
  []
[]

[Tensors]
  [dx1]
    type = Python
    expr = "Scalar(torch.tensor([0.2, 0.1], dtype=torch.float64))"
  []
  [dx2]
    type = Python
    expr = "Scalar(torch.tensor([0.05, 0.15], dtype=torch.float64))"
  []
  [dg]
    type = Python
    expr = "Scalar(torch.tensor([0.145, 0.135], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = ChemicalGibbsFreeEnergyDifference
    concentration_differences = 'state/dx1 state/dx2'
    chemical_potentials = '1.0 2.0'
    equilibrium_potentials = '0.4 1.5'
    chemical_gibbs_free_energy = 'state/dg'
  []
[]
