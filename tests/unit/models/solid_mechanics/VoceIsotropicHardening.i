[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'equivalent_plastic_strain'
    input_Scalar_values = '0.1'
    output_Scalar_names = 'isotropic_hardening'
    output_Scalar_values = '10.416586470347177'
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = VoceIsotropicHardening
    saturated_hardening = 100
    saturation_rate = 1.1
  []
[]
