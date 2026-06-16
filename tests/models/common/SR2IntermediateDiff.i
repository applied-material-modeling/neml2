# Translated from tests/unit/models/common/SR2IntermediateDiff.i pattern.
# Input S: dynamic batch (4,), sub batch (3,) -- 12 raw-Mandel SR2 rows.
# Output S_diff: dynamic batch (4,), sub batch (2,) -- first-order finite
# difference along the trailing sub-batch axis (dim=-1, n=1).
# C++ `type = SR2 values = ...` stores raw Mandel components directly (no
# sqrt(2) scaling), so we feed the same numbers via SR2(torch.tensor(...)).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'S'
    input_SR2_values = 'S'
    output_SR2_names = 'S_diff'
    output_SR2_values = 'S_diff'
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
  [S_diff]
    type = Python
    expr = 'SR2(torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
                               [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]],
                               [[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]],
                               [[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0]]]),
                sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = SR2IntermediateDiff
    from = 'S'
    to = 'S_diff'
    dim = -1
    reduces = ''  # unlabelled sub-batch in this fixture
    n = 1
  []
[]
