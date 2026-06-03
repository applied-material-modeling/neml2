# Translated from tests/unit/models/kwn/KineticFactor.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R_crit state/sum'
    input_Scalar_values = 'R_crit sum'
    output_Scalar_names = 'state/beta'
    output_Scalar_values = 'beta'
  []
[]

[Tensors]
  [R_crit]
    type = Python
    expr = "Scalar(torch.tensor(2.0, dtype=torch.float64))"
  []
  [sum]
    type = Python
    expr = "Scalar(torch.tensor(4.0, dtype=torch.float64))"
  []
  [beta]
    type = Python
    expr = "Scalar(torch.tensor(12.566370614359172, dtype=torch.float64))"
  []
[]

[Models]
  [model]
    type = KineticFactor
    critical_radius = 'state/R_crit'
    projected_diffusivity_sum = 'state/sum'
    molar_volume = 1.0
    avogadro_number = 1.0
    kinetic_factor = 'state/beta'
  []
[]
