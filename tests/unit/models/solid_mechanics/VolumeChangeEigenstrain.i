[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/V'
    input_Scalar_values = '412'
    output_SR2_names = 'forces/Eg'
    output_SR2_values = 'EV_correct'
    check_second_derivatives = true
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
    type = VolumeChangeEigenstrain
    reference_volume = 211
    volume = 'state/V'
    eigenstrain = 'forces/Eg'
  []
[]
