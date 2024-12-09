[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/gamma_rate state/internal/Nk'
    input_scalar_values = '0.0015 -1'
    output_scalar_names = 'state/internal/ep_rate'
    output_scalar_values = '0.0015'
  []
[]

[Models]
  [model]
    type = AssociativeIsotropicPlasticHardening
  []
[]
