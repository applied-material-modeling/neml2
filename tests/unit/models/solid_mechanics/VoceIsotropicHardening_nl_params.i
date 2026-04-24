[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'equivalent_plastic_strain R d'
    input_Scalar_values = '0.1 100.0 1.1'
    output_Scalar_names = 'isotropic_hardening'
    output_Scalar_values = '10.416586470347177'
  []
[]

[Models]
  [model0]
    type = VoceIsotropicHardening
    saturated_hardening = 'R'
    saturation_rate = 'd'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
