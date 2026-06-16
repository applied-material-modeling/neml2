# Translated from tests/unit/models/common/SR2IntermediateSum.i.
# Input S: dynamic batch (4,), sub batch (3,) -- 12 raw-Mandel SR2 rows.
# Output S_sum: dynamic batch (4,) -- sum over the trailing sub-batch axis.
# C++ `type = SR2 values = ...` stores raw Mandel components directly (no sqrt(2)
# scaling), so we feed the same numbers via SR2(torch.tensor(...)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'S'
    input_SR2_values = 'S'
    output_SR2_names = 'S_sum'
    output_SR2_values = 'S_sum'
  []
[]

[Tensors]
  [S]
    type = Python
    expr = 'SR2(torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                [2.0, 3.0, 4.0, 4.0, 5.0, 6.0],
                                [3.0, 4.0, 5.0, 4.0, 5.0, 6.0]],
                               [[4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
                                [5.0, 6.0, 7.0, 4.0, 5.0, 6.0],
                                [6.0, 7.0, 8.0, 4.0, 5.0, 6.0]],
                               [[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                                [-2.0, -3.0, -4.0, -4.0, -5.0, -6.0],
                                [-3.0, -4.0, -5.0, -4.0, -5.0, -6.0]],
                               [[-4.0, -5.0, -6.0, -4.0, -5.0, -6.0],
                                [-5.0, -6.0, -7.0, -4.0, -5.0, -6.0],
                                [-6.0, -7.0, -8.0, -4.0, -5.0, -6.0]]]),
                sub_batch_ndim=1)'
  []
  [S_sum]
    type = Python
    expr = 'SR2(torch.tensor([[6.0, 9.0, 12.0, 12.0, 15.0, 18.0],
                               [15.0, 18.0, 21.0, 12.0, 15.0, 18.0],
                               [-6.0, -9.0, -12.0, -12.0, -15.0, -18.0],
                               [-15.0, -18.0, -21.0, -12.0, -15.0, -18.0]]))'
  []
[]

[Models]
  [model]
    type = SR2IntermediateSum
    from = 'S'
    to = 'S_sum'
    reduces = ''  # unlabelled sub-batch in this fixture
  []
[]
