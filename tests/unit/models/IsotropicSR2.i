[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/s'
    input_Scalar_values = '1.5'
    output_SR2_names = 'state/A'
    output_SR2_values = 'A'
  []
[]

[Tensors]
  [A]
    type = FillSR2
    values = '1.5 1.5 1.5'
  []
[]

[Models]
  [model0]
    type = IsotropicSR2
    factor = 'state/s'
    output = 'state/A'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
