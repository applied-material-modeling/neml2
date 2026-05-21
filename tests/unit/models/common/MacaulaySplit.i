# neml2
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'x'
    input_Scalar_values = '0.7'
    output_Scalar_names = 'pos neg'
    output_Scalar_values = '0.7 0.0'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = MacaulaySplit
    from = 'x'
    to_positive = 'pos'
    to_negative = 'neg'
  []
[]
