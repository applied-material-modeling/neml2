[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/V state/Vref'
    input_Scalar_values = '412 211'
    output_SR2_names = 'forces/Eg'
    output_SR2_values = 'EV_correct'
  []
[]

[Tensors]
  [EV_correct]
    type = FillSR2
    values = '0.2498894057 0.2498894057 0.2498894057 0 0 0'
  []
[]

[Models]
  [model]
    type = VolumeChangeEigenstrainStateful
    reference_volume = 'state/Vref'
    volume = 'state/V'
    eigenstrain = 'forces/Eg'
  []
[]
