[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'yield_function'
    input_Scalar_values = '50'
    output_Scalar_names = 'flow_rate'
    output_Scalar_values = '0.0013717421124828527'
  []
[]

[Models]
  [model]
    type = PerzynaPlasticFlowRate
    reference_stress = 150
    exponent = 6
  []
[]
