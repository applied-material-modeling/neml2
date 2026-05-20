[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/n'
    input_Scalar_values = 'n'
    output_Scalar_names = 'state/f'
    output_Scalar_values = 'f'
    input_with_intrsc_intmd_dims = 'state/n'
    input_intrsc_intmd_dims = '1'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [R]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [n]
    type = Scalar
    values = '10 20 30'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [f]
    type = Scalar
    values = '4105.014400690663'
  []
[]

[Models]
  [model]
    type = PrecipitateVolumeFraction
    radius = 'R'
    number_density = 'state/n'
    volume_fraction = 'state/f'
  []
[]
