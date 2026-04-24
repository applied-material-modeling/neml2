[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'equivalent_plastic_strain'
    input_Scalar_values = '0.1'
    output_Scalar_names = 'isotropic_hardening'
    output_Scalar_values = '100'
  []
[]

[Models]
  [model]
    type = LinearIsotropicHardening
    hardening_modulus = 1000
  []
[]
