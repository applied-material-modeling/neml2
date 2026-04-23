[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'H'
    input_Scalar_values = '1000'
    input_SR2_names = 'Kp'
    input_SR2_values = 'Kp'
    output_SR2_names = 'X'
    output_SR2_values = 'X'
  []
[]

[Tensors]
  [Kp]
    type = FillSR2
    values = '0.05 -0.01 0.02 0.04 0.03 -0.06'
  []
  [X]
    type = FillSR2
    values = '50 -10 20 40 30 -60'
  []
[]

[Models]
  [model0]
    type = LinearKinematicHardening
    hardening_modulus = 'H'
    kinematic_plastic_strain = 'Kp'
    back_stress = 'X'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
