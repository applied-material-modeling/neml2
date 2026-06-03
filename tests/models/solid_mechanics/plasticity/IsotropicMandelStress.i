# Translated from tests/unit/models/solid_mechanics/plasticity/IsotropicMandelStress.i
# (FillSR2 6-value -> SR2.fill; shear slots scaled by sqrt(2)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'cauchy_stress'
    input_SR2_values = 'S'
    output_SR2_names = 'mandel_stress'
    output_SR2_values = 'M'
  []
[]

[Tensors]
  [S]
    type = Python
    expr = 'SR2.fill(50, -10, 20, 40, 30, -60)'
  []
  [M]
    type = Python
    expr = 'SR2.fill(50, -10, 20, 40, 30, -60)'
  []
[]

[Models]
  [model]
    type = IsotropicMandelStress
  []
[]
