[Tensors]
  [foo]
    type = Scalar
    values = '-0.5 0.01 0.02 0.5 0.95 1.01 2'
    batch_shape = '(7)'
  []
  [bar]
    type = Scalar
    values = '1.0 0.896 0.648 0 0 0 0'
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
  []
[]

[Models]
  [model]
    type = HermiteSmoothStep
    argument = 'state/foo'
    value = 'state/bar'
    lower_bound = '0'
    upper_bound = '0.05'
    complement = true
  []
[]
