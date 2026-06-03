# Translated from tests/unit/models/common/ScalarMultiplication1.i.
# Basic multiplication: scaling * A * B * C * D = 1.5 * 3.1 * 2.5 * 4.6 * 7.8 = 417.105.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'A B C D'
    input_Scalar_values = '3.1 2.5 4.6 7.8'
    output_Scalar_names = 'E'
    output_Scalar_values = '417.105'
    value_rel_tol = 1e-5
    value_abs_tol = 1e-5
  []
[]

[Models]
  [model]
    type = ScalarMultiplication
    from = 'A B C D'
    to = 'E'
    scaling = 1.5
  []
[]
