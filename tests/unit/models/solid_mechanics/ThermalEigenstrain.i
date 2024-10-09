[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'forces/T'
    input_scalar_values = '400'
    output_symr2_names = 'forces/Eg'
    output_symr2_values = 'Eg_correct'
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
