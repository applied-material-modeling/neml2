# Translated from tests/unit/models/chemical_reactions/EffectiveVolume.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'wb ws wp wg phiop'
    input_Scalar_values = 'wb ws wp wg phiop'
    output_Scalar_names = 'V'
    output_Scalar_values = 'Vref'
  []
[]

[Tensors]
  [wb]
    type = Python
    expr = "Scalar(torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64))"
  []
  [ws]
    type = Python
    expr = "Scalar(torch.tensor([0.5, 0.6, 0.2], dtype=torch.float64))"
  []
  [wp]
    type = Python
    expr = "Scalar(torch.tensor([0.1, 0.22, 0.4], dtype=torch.float64))"
  []
  [wg]
    type = Python
    expr = "Scalar(torch.tensor([0.15, 0.23, 0.33], dtype=torch.float64))"
  []
  [phiop]
    type = Python
    expr = "Scalar(torch.tensor([0.9, 0.01, 0.55], dtype=torch.float64))"
  []
  [Vref]
    type = Python
    expr = "Scalar(torch.tensor([0.6024819194, 0.09441082427, 0.2818082603], dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = EffectiveVolume
    reference_mass = 4.1
    open_volume_fraction = 'phiop'
    mass_fractions = 'wb ws wp wg'
    densities = '1123 576 988 11'
    composite_volume = 'V'
  []
[]
