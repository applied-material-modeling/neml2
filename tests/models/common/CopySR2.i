# SR2-valued CopyVariable smoke test (no C++ Catch2 fixture exists for this
# template instantiation; pattern mirrors SR2ConstantParameter.i; the 6-value
# FillSR2 convention scales the shear slots by sqrt(2) into Mandel).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'copy'
    input_SR2_names = 'src'
    input_SR2_values = 'T'
    output_SR2_names = 'dst'
    output_SR2_values = 'T'
  []
[]

[Models]
  [copy]
    type = CopySR2
    from = 'src'
    to = 'dst'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SR2.fill(-1, -4, 7, -1, 9, 1)'
  []
[]
