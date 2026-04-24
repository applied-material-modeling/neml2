[Tensors]
  [T_vals]
    type = Scalar
    values = '0 100 200'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
  [c_A_vals]
    type = Scalar
    values = '1 2 3'
    batch_shape = '(3)'
    intermediate_dimension = 1
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'T A B'
    input_Scalar_values = '100 0.5 1.5'
    output_Scalar_names = 'C'
    output_Scalar_values = '9.25'
    check_AD_parameter_derivatives = false
  []
[]

[Models]
  [c_A]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_vals'
    ordinate = 'c_A_vals'
  []
  [model0]
    type = ScalarLinearCombination
    from = 'A B'
    to = 'C'
    weights = 'c_A 5.5'
    weight_as_parameter = 'true true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
