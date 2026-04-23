[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature'
    input_Scalar_values = '400'
    output_SR2_names = 'eigenstrain'
    output_SR2_values = 'Eg_correct'
  []
[]

[Tensors]
  [Eg_correct]
    type = FillSR2
    values = '1e-3 1e-3 1e-3 0 0 0'
  []
[]

[Models]
  [model]
    type = ThermalEigenstrain
    reference_temperature = 300
    CTE = 1e-5
  []
[]
