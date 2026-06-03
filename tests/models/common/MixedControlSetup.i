# Translated from tests/unit/models/common/MixedControlSetup.i (FillSR2 6-value -> SR2.fill).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'x_below x_above control'
    input_SR2_values = 'mvals vals control'
    output_SR2_names = 'y z'
    output_SR2_values = 'stress strain'
  []
[]

[Tensors]
  [control]
    type = Python
    expr = 'SR2.fill(1.0, 0.0, 0.0, 1.0, 1.0, 0.0)'
  []
  [vals]
    type = Python
    expr = 'SR2.fill(-50.0, 0.1, 0.15, -25.0, 30.0, -0.05)'
  []
  [mvals]
    type = Python
    expr = 'SR2.fill(0.1, 100.0, 20.0, -0.05, -0.025, 50.0)'
  []
  [stress]
    type = Python
    expr = 'SR2.fill(-50.0, 100.0, 20.0, -25.0, 30.0, 50.0)'
  []
  [strain]
    type = Python
    expr = 'SR2.fill(0.1, 0.1, 0.15, -0.05, -0.025, -0.05)'
  []
[]

[Models]
  [model]
    type = MixedControlSetup
  []
[]
