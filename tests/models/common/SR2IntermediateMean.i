# Translated from tests/unit/models/common/SR2IntermediateMean.i.
# Input S: dynamic batch (4,), sub batch (3,) -- 12 raw-Mandel SR2 rows.
# Output S_mean: dynamic batch (4,) -- mean over the trailing sub-batch axis.
# C++ `type = SR2 values = ...` stores raw Mandel components directly (no sqrt(2)
# scaling), so we feed the same numbers via SR2(torch.tensor(...)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'S'
    input_SR2_values = 'S'
    output_SR2_names = 'S_mean'
    output_SR2_values = 'S_mean'
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
  [S_mean]
    type = Python
    expr = 'SR2(torch.tensor([[2.0, 3.0, 4.0, 4.0, 5.0, 6.0],
                               [5.0, 6.0, 7.0, 4.0, 5.0, 6.0],
                               [-2.0, -3.0, -4.0, -4.0, -5.0, -6.0],
                               [-5.0, -6.0, -7.0, -4.0, -5.0, -6.0]]))'
  []
[]

[Models]
  [model]
    type = SR2IntermediateMean
    from = 'S'
    to = 'S_mean'
    reduces = ''  # unlabelled sub-batch in this fixture
  []
[]
