[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/f state/dv'
    input_Scalar_values = '0.7 0.3'
    output_SR2_names = 'forces/Eg'
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
    volume_fraction_change = 'state/dv'
    phase_fraction = 'state/f'
  []
[]
