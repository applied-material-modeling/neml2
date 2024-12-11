[Tensors]
  [X_values]
    type = Scalar
    values = '0.0 1.0 2.0 3.0'
    batch_shape = '(4)'
  []
  [Y_values]
    type = Scalar
    values = '2.0 -1.0 5.0 10.0'
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = ScalarLinearInterpolation
    argument = 'forces/inp'
    abscissa = 'X_values'
    ordinate = 'Y_values'
  []
  [model2]
    type = ScalarLinearCombination
    from_var = 'forces/x1 forces/x2 forces/x3 forces/x4'
    to_var = 'forces/y'
    coefficients = '0.1 0.2 0.3 0.4'
    coefficients_as_parameters = true
  []
[]
