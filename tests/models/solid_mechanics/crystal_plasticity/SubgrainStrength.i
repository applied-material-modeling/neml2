[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'subgrain_size'
    input_Scalar_values = '4.0'
    output_Scalar_names = 'subgrain_strength'
    output_Scalar_values = '10000.0'
  []
[]

[Models]
  [model]
    type = SubgrainStrength
    alpha = 0.25
    shear_modulus = 80000.0
    burgers_vector = 2.0
  []
[]
