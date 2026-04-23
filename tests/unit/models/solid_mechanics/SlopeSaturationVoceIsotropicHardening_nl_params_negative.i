[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'flow_rate R theta0 isotropic_hardening'
    input_Scalar_values = '0.1 -100.0 110 50.0'
    output_Scalar_names = 'isotropic_hardening_rate'
    output_Scalar_values = '-16.5'
  []
[]

[Models]
  [model0]
    type = SlopeSaturationVoceIsotropicHardening
    saturated_hardening = 'R'
    initial_hardening_rate = 'theta0'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
