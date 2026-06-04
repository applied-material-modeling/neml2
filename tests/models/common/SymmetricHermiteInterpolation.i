# Translated from tests/unit/models/common/SymmetricHermiteInterpolation.i.
[Tensors]
  [foo]
    type = Python
    expr = 'Scalar([-0.5, 0.01, 0.15, 0.6, 0.95, 1.05, 2.0])'
  []
  [bar]
    type = Python
    expr = 'Scalar([0.0, 0.0, 0.056, 2.0, 0.432, 0.056, 0.0])'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo'
    input_Scalar_values = 'foo'
    output_Scalar_names = 'bar'
    output_Scalar_values = 'bar'
    derivative_rel_tol = 0
    derivative_abs_tol = 1e-3
  []
[]

[Models]
  [model]
    type = SymmetricHermiteInterpolation
    argument = 'foo'
    output = 'bar'
    lower_bound = '0.1'
    upper_bound = '1.1'
  []
[]
