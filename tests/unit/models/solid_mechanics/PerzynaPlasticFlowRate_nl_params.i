[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/fp params/eta params/n'
    input_scalar_values = '50 150 6'
    output_scalar_names = 'state/internal/gamma_rate'
    output_scalar_values = '0.0013717421124828527'
  []
[]

[Models]
  [eta]
    type = ScalarInputParameter
    from = 'params/eta'
  []
  [n]
    type = ScalarInputParameter
    from = 'params/n'
  []
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
