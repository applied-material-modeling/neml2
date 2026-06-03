# Translated from tests/unit/models/common/SR2DynamicSum.i.
# Input S: dynamic batch (4, 3), no sub-batch -- 12 raw-Mandel SR2 rows.
# Output S_sum: dynamic batch (4,) -- sum over the trailing dynamic axis.
# C++ `type = SR2 values = ...` stores raw Mandel components directly (no sqrt(2)
# scaling), so we feed the same numbers via SR2(torch.tensor(...)).
# C++ uses `dim = 1`; we use the equivalent `dim = -1` so the model's chain-rule
# action (which assumes the leading-K tangent convention for non-negative dims)
# stays in its negative-dim branch under the native ModelUnitTest harness, where
# input tangents carry the bare input shape rather than a K-prepended one.
# Output values are identical -- dim=1 and dim=-1 select the same axis for a
# 2-D dynamic batch shape (4, 3).
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
                                [-6.0, -7.0, -8.0, -4.0, -5.0, -6.0]]]))'
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
    type = SR2DynamicSum
    from = 'S'
    to = 'S_sum'
    dim = -1
  []
[]
