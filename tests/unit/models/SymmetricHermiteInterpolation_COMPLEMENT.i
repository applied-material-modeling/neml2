[Tensors]
  [foo]
    type = Scalar
    values = '-0.5 0.01 0.15 0.6 0.95 1.05 2'
    batch_shape = '(7)'
  []
  [bar]
    type = Scalar
    values = '1.0 1.0 0.972 0.0 0.784 0.972 1.0'
    batch_shape = '(7)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/foo'
    input_Scalar_values = 'foo'
    output_Scalar_names = 'state/bar'
    output_Scalar_values = 'bar'
    derivative_rel_tol = 0
    derivative_abs_tol = 1e-3
    check_second_derivatives = true
  []
[]

[Models]
  [model]
    type = SymmetricHermiteInterpolation
    argument = 'state/foo'
    value = 'state/bar'
    lower_bound = '0.1'
    upper_bound = '1.1'
    complement_condition = true
  []
[]
