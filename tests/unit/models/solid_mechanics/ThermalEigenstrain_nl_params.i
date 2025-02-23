[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'forces/T state/alpha'
    input_Scalar_values = '400 1.0e-5'
    output_SR2_names = 'forces/Eg'
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
  [model0]
    type = ThermalEigenstrain
    reference_temperature = 300
    CTE = 'state/alpha'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
