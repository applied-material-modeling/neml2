[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R state/sum state/dg state/T'
    input_Scalar_values = 'R sum dg T'
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
  [sum]
    type = Scalar
    values = '0.5'
  []
  [dg]
    type = Scalar
    values = '2.0'
  []
  [T]
    type = Scalar
    values = '400.0'
  []
  [R_dot]
    type = Scalar
    values = '0.00125 0.000625 0.0003125'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Models]
  [model]
    type = SFFKGPrecipitationGrowthRate
    radius = 'state/R'
    projected_diffusivity_sum = 'state/sum'
    gibbs_free_energy_difference = 'state/dg'
    temperature = 'state/T'
    gas_constant = 8.0
    growth_rate = 'state/R_dot'
  []
[]
