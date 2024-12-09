[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/ep'
    input_scalar_values = '0.1'
    output_scalar_names = 'state/internal/k'
    output_scalar_values = '10.416586470347177'
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
