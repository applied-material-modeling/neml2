[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/ws'
    input_Scalar_values = 'ws'
    output_Scalar_names = 'state/alpha'
    output_Scalar_values = 'alpha'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [ws]
    type = Scalar
    values = '0.03 0.75 0.22'
    batch_shape = '(3)'
  []
  [alpha]
    type = Scalar
    values = '1.175 5.375 2.28333333'
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = ThermalDecompositionConversionDegree
    initial_precursor_mass_fraction = 0.4
    reaction_yield = 0.7

    precursor_mass_fraction = 'state/ws'
    conversion_degree = 'state/alpha'
  []
[]
