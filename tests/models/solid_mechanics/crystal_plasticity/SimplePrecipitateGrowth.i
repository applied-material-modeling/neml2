[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'precipitate_radius precipitate_density'
    input_Scalar_values = '2.0 4.0'
    output_Scalar_names = 'precipitate_radius_rate precipitate_density_rate'
    output_Scalar_values = '0.125 -0.1875'
  []
[]

[Models]
  [model]
    type = SimplePrecipitateGrowth
    coarsening_coefficient = 1.5
  []
[]
