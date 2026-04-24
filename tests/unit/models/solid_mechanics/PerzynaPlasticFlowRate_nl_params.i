[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'yield_function eta n'
    input_Scalar_values = '50 150 6'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = '0.0013717421124828527'
  []
[]

[Models]
  [model0]
    type = PerzynaPlasticFlowRate
    reference_stress = 'eta'
    exponent = 'n'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
