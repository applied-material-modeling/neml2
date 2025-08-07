[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/S'
    input_SR2_values = 'S'
    output_SR2_names = 'state/S_sum'
    output_SR2_values = 'S_sum'
    check_derivatives = false
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
  []
  [S_sum]
    type = SR2
    values = "6   9  12  12  15  18
              15  18  21  12  15  18
              -6  -9 -12 -12 -15 -18
              -15 -18 -21 -12 -15 -18"
    batch_shape = '(4)'
  []
[]

[Models]
  [model]
    type = SR2DynamicSum
    from = 'state/S'
    to = 'state/S_sum'
    dim = 1
  []
[]
