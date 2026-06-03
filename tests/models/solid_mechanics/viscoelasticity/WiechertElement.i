# Translated from tests/unit/models/solid_mechanics/viscoelasticity/WiechertElement.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain viscous_strain_1 viscous_strain_2'
    input_SR2_values = 'E Ev1 Ev2'
    output_SR2_names = 'stress viscous_strain_1_rate viscous_strain_2_rate'
    output_SR2_values = 'S Ev1_dot Ev2_dot'
  []
[]

[Tensors]
  [E]
    type = Python
    expr = 'SR2.fill(0.01, -0.005, -0.005, 0.002, -0.001, 0.0005)'
  []
  [Ev1]
    type = Python
    expr = 'SR2.fill(0.002, -0.001, -0.001, 0.0004, -0.0002, 0.0001)'
  []
  [Ev2]
    type = Python
    expr = 'SR2.fill(0.001, -0.0005, -0.0005, 0.0002, -0.0001, 0.00005)'
  []
  # Einf = 1000, E1 = 500, eta1 = 100, E2 = 200, eta2 = 50
  # S = (Einf + E1 + E2) * E - E1 * Ev1 - E2 * Ev2 = 1700 * E - 500 * Ev1 - 200 * Ev2
  [S]
    type = Python
    expr = 'SR2.fill(15.8, -7.9, -7.9, 3.16, -1.58, 0.79)'
  []
  # Ev1_dot = E1 * (E - Ev1) / eta1 = 500 * (E - Ev1) / 100 = 5 * (E - Ev1)
  [Ev1_dot]
    type = Python
    expr = 'SR2.fill(0.04, -0.02, -0.02, 0.008, -0.004, 0.002)'
  []
  # Ev2_dot = E2 * (E - Ev2) / eta2 = 200 * (E - Ev2) / 50 = 4 * (E - Ev2)
  [Ev2_dot]
    type = Python
    expr = 'SR2.fill(0.036, -0.018, -0.018, 0.0072, -0.0036, 0.0018)'
  []
[]

[Models]
  [model]
    type = WiechertElement
    equilibrium_modulus = 1000
    modulus_1 = 500
    viscosity_1 = 100
    modulus_2 = 200
    viscosity_2 = 50
  []
[]
