[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'flow_rate isotropic_hardening_direction'
    input_Scalar_values = '0.0015 -1'
    output_Scalar_names = 'equivalent_plastic_strain_rate'
    output_Scalar_values = '0.0015'
  []
[]

[Models]
  [model]
    type = AssociativeIsotropicPlasticHardening
  []
[]
