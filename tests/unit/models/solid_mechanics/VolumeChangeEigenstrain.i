[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'volume'
    input_Scalar_values = '412'
    output_SR2_names = 'eigenstrain'
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
    type = VolumeChangeEigenstrain
    reference_volume = 211
    volume = 'volume'
    eigenstrain = 'eigenstrain'
  []
[]
