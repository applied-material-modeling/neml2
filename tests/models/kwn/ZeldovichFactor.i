# Translated from tests/unit/models/kwn/ZeldovichFactor.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R_crit state/T'
    input_Scalar_values = 'R_crit T'
    output_Scalar_names = 'state/Z'
    output_Scalar_values = 'Z'
  []
[]

[Tensors]
  [R_crit]
    type = Python
    expr = 'Scalar(2.0)'
  []
  [gamma]
    type = Python
    expr = 'Scalar(4.0)'
  []
  [T]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [Z]
    type = Python
    expr = 'Scalar(0.07957747154594767)'
  []
[]

[Models]
  [model]
    type = ZeldovichFactor
    critical_radius = 'state/R_crit'
    surface_energy = 'gamma'
    temperature = 'state/T'
    molar_volume = 2.0
    avogadro_number = 2.0
    boltzmann_constant = 1.0
    zeldovich_factor = 'state/Z'
  []
[]
