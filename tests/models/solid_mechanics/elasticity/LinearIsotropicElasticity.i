# Translated from tests/unit/models/solid_mechanics/elasticity/LinearIsotropicElasticity.i
# (FillSR2 'a b c' -> SR2.fill, zero shear).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain'
    input_SR2_values = 'Ee'
    output_SR2_names = 'stress'
    output_SR2_values = 'S'
  []
[]

[Tensors]
  [Ee]
    type = Python
    expr = 'SR2.fill(0.09, 0.04, -0.02)'
  []
  [S]
    type = Python
    expr = 'SR2.fill(13.2692, 9.4231, 4.8077)'
  []
[]

[Models]
  [model]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
[]
