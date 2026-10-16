[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'subgrain_size temperature'
    input_Scalar_values = '2.0 4.0'
    output_Scalar_names = 'subgrain_size_rate'
    output_Scalar_values = '6.0'
  []
[]

[Models]
  [model]
    type = SubgrainCoarsening
    coarsening_coefficient = 3.0
    shear_modulus = 1.0
    burgers_vector = 2.0
    boltzmann_constant = 0.5
  []
[]
