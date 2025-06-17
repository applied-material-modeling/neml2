[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/alpha'
    input_Scalar_values = 'alpha'
    output_Scalar_names = 'state/out'
    output_Scalar_values = 'out'
    check_AD_parameter_derivatives = false
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [alpha]
    type = Scalar
    values = '0.03 0.75 0.9'
    batch_shape = '(3)'
  []
  [out]
    type = Scalar
    values = '4.592994468e-5 0.429675464 0.69373113'
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = AvramiErofeevNucleation
    reaction_coef = 0.7
    reaction_order = 2.75
    conversion_degree = 'state/alpha'
    reaction_rate = 'state/out'
  []
[]
