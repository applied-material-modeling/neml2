# Default weights (all 1) and zero offset: C = A + B for full R2 tensors.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'A B'
    input_R2_values = 'foo bar'
    output_R2_names = 'C'
    output_R2_values = 'baz'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'R2.fill(1, 2, 3, 4, 5, 6, 7, 8, 9)'
  []
  [bar]
    type = Python
    expr = 'R2.fill(9, 8, 7, 6, 5, 4, 3, 2, 1)'
  []
  [baz]
    type = Python
    expr = 'R2.fill(10, 10, 10, 10, 10, 10, 10, 10, 10)'
  []
[]

[Models]
  [model]
    type = R2LinearCombination
    from = 'A B'
    to = 'C'
  []
[]
