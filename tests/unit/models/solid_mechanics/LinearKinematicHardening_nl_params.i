[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/H'
    input_Scalar_values = '1000'
    input_SR2_names = 'state/internal/Kp'
    input_SR2_values = 'Kp'
    output_SR2_names = 'state/internal/X'
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
    hardening_modulus = 'state/H'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
