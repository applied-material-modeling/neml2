[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/gamma_rate'
    input_Scalar_values = '0.0015'
    input_SR2_names = 'state/internal/NX'
    input_SR2_values = 'NX'
    output_SR2_names = 'state/internal/Kp_rate'
    output_SR2_values = 'Kp_rate'
  []
[]

[Tensors]
  [NX]
    type = FillSR2
    values = '0.3482 -0.3482 0 -0.087045 -0.087045 -0.78333'
  []
  [Kp_rate]
    type = FillSR2
    values = '-0.0005223 0.0005223 0 0.0001305675 0.0001305675 0.001174995'
  []
[]

[Models]
  [model]
    type = AssociativeKinematicPlasticHardening
  []
[]
