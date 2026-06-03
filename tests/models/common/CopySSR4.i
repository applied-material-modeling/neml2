# SSR4-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors SSR4ConstantParameter.i).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_SSR4_names = 'src'
    input_SSR4_values = 'T'
    output_SSR4_names = 'dst'
    output_SSR4_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopySSR4
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SSR4(torch.eye(6, dtype=torch.float64))'
  []
[]
