[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R state/x_inf'
    input_Scalar_values = 'R x_inf'
    output_Scalar_names = 'state/R_dot'
    output_Scalar_values = 'R_dot'
    input_with_intrsc_intmd_dims = 'state/R'
    input_intrsc_intmd_dims = '1'
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
    radius = 'state/R'
    current_concentration = 'state/x_inf'
    equilibrium_concentration = 0.1
    precipitate_concentration = 0.8
    diffusivity = 2.0
    growth_rate = 'state/R_dot'
  []
[]
