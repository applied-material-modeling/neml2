# Scalar-valued CopyVariable smoke test (no C++ Catch2 fixture exists for
# this template instantiation; pattern mirrors ScalarConstantParameter.i and
# the ComposedModel4.i use of CopyScalar).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_Scalar_names = 'src'
    input_Scalar_values = 'T'
    output_Scalar_names = 'dst'
    output_Scalar_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopyScalar
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'Scalar(torch.tensor(2.5, dtype=torch.float64))'
  []
[]
