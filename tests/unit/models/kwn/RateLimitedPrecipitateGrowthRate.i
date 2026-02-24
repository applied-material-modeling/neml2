[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/x_inf'
    input_Scalar_values = 'x_inf'
    output_Scalar_names = 'state/R_dot'
    output_Scalar_values = 'R_dot'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [R]
    type = Scalar
    values = '1 2 4'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [x_inf]
    type = Scalar
    values = '0.25'
  []
  [R_dot]
    type = Scalar
    values = '0.42857142857142855 0.21428571428571427 0.10714285714285714'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = RateLimitedPrecipitateGrowthRate
    radius = 'R'
    current_concentration = 'state/x_inf'
    equilibrium_concentration = 0.1
    concentration_difference = 0.7
    diffusivity = 2.0
    growth_rate = 'state/R_dot'
  []
[]
