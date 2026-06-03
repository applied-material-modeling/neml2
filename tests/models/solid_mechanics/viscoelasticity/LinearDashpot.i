# Translated from tests/unit/models/solid_mechanics/viscoelasticity/LinearDashpot.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'stress'
    input_SR2_values = 'S'
    output_SR2_names = 'viscous_strain_rate'
    output_SR2_values = 'Ev_dot'
  []
[]

[Tensors]
  [S]
    type = Python
    expr = 'SR2.fill(100.0, -50.0, -50.0, 20.0, -10.0, 5.0)'
  []
  [Ev_dot]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.02, -0.01, 0.005)'
  []
[]

[Models]
  [model]
    type = LinearDashpot
    viscosity = 1000
  []
[]
