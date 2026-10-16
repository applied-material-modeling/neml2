[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'precipitate_radius precipitate_density'
    input_Scalar_values = '2.0 4.0'
    output_Scalar_names = 'precipitate_strength'
    output_Scalar_values = '5.0'
  []
[]

[Models]
  [model]
    type = PrecipitateStrength
    alpha = 0.4
    shear_modulus = 100.0
    burgers_vector = 2.0
  []
[]
