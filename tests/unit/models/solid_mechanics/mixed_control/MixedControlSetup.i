[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    batch_shape = '()'
    input_symr2_names = 'state/mixed_state forces/fixed_values forces/control'
    input_symr2_values = 'mvals vals control'
    output_symr2_names = 'state/S forces/E'
    output_symr2_values = 'stress strain'
  []
[]

[Tensors]
  [control]
    type = FillSR2
    values = '1.0 0.0 0.0 1.0 1.0 0.0'
  []
  [vals]
    type = FillSR2
    values = '-50.0 0.1 0.15 -25.0 30.0 -0.05'
  []
  [mvals]
    type = FillSR2
    values = '0.1 100.0 20.0 -0.05 -0.025 50.0'
  []
  [stress]
    type = FillSR2
    values = '-50.0 100.0 20.0 -25.0 30.0 50.0'
  []
  [strain]
    type = FillSR2
    values = '0.1 0.1 0.15 -0.05 -0.025 -0.05'
  []
[]

[Models]
  [model]
    type = MixedControlSetup
  []
[]
