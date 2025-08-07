[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/S_mean'
    output_SR2_values = 'S_mean'
  []
[]

[Tensors]
  [S]
    type = SR2
    values = "1 2 3 4 5 6
              2 3 4 4 5 6
              3 4 5 4 5 6
              4 5 6 4 5 6
              5 6 7 4 5 6
              6 7 8 4 5 6
              -1 -2 -3 -4 -5 -6
              -2 -3 -4 -4 -5 -6
              -3 -4 -5 -4 -5 -6
              -4 -5 -6 -4 -5 -6
              -5 -6 -7 -4 -5 -6
              -6 -7 -8 -4 -5 -6"
    batch_shape = '(4,3)'
    intermediate_dimension = 2
  []
  [S_mean]
    type = SR2
    values = "2 3 4 4 5 6
              5 6 7 4 5 6
              -2 -3 -4 -4 -5 -6
              -5 -6 -7 -4 -5 -6"
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = SR2IntermediateMean
    from = 'state/S'
    to = 'state/S_mean'
    dim = 1
  []
[]
