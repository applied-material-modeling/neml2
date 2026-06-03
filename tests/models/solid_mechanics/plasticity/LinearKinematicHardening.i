# Translated from tests/unit/models/solid_mechanics/plasticity/LinearKinematicHardening.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'kinematic_plastic_strain'
    input_SR2_values = 'Kp'
    output_SR2_names = 'back_stress'
    output_SR2_values = 'X'
  []
[]

[Tensors]
  [Kp]
    type = Python
    expr = 'SR2.fill(0.05, -0.01, 0.02, 0.04, 0.03, -0.06)'
  []
  [X]
    type = Python
    expr = 'SR2.fill(50.0, -10.0, 20.0, 40.0, 30.0, -60.0)'
  []
[]

[Models]
  [model]
    type = LinearKinematicHardening
    hardening_modulus = 1000
  []
[]
