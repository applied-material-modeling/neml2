# Translated from tests/unit/models/solid_mechanics/kinematics/ThermalEigenstrain.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature'
    input_Scalar_values = '400'
    output_SR2_names = 'eigenstrain'
    output_SR2_values = 'Eg_correct'
  []
[]

[Tensors]
  [Eg_correct]
    type = Python
    expr = "SR2.fill(1e-3, 1e-3, 1e-3)"
  []
[]

[Models]
  [model]
    type = ThermalEigenstrain
    reference_temperature = 300
    CTE = 1e-5
  []
[]
