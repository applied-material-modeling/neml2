[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/fp'
    input_scalar_values = '50'
    output_scalar_names = 'state/internal/gamma_rate'
    output_scalar_values = '0.0013717421124828527'
  []
[]

[Models]
  [model]
    type = PerzynaPlasticFlowRate
    reference_stress = 150
    exponent = 6
  []
[]
