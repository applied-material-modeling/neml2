[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'phase_fraction volume_fraction_change'
    input_Scalar_values = '0.7 0.3'
    output_SR2_names = 'eigenstrain'
    output_SR2_values = 'EV_correct'
  []
[]

[Tensors]
  [EV_correct]
    type = FillSR2
    values = '0.21 0.21 0.21 0 0 0'
  []
[]

[Models]
  [model]
    type = PhaseTransformationEigenstrain
    volume_fraction_change = 'volume_fraction_change'
    phase_fraction = 'phase_fraction'
  []
[]
