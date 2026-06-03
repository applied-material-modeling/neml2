# SR2-valued InputParameter smoke test (no C++ Catch2 fixture; pattern
# mirrors ScalarInputParameter.i). FillSR2 6-value -> SR2.fill (shear slots
# scaled by sqrt(2) into Mandel internally).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ip'
    input_SR2_names = 'variable'
    input_SR2_values = 'T'
    output_SR2_names = 'parameter'
    output_SR2_values = 'T'
  []
[]

[Models]
  [ip]
    type = SR2InputParameter
  []
[]

[Tensors]
  [T]
    type = Python
    expr = 'SR2.fill(-1, -4, 7, -1, 9, 1)'
  []
[]
