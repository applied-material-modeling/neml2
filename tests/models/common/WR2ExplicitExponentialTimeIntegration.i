# Translated from tests/unit/models/common/WR2ExplicitExponentialTimeIntegration.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_WR2_names = 'foo_rate'
    input_WR2_values = 'w'
    input_MRP_names = 'foo~1'
    input_MRP_values = 'old_foo'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_MRP_names = 'foo'
    output_MRP_values = 'foo'
  []
[]

[Models]
  [model]
    type = WR2ExplicitExponentialTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = "MRP(torch.tensor([0.03789409, -0.01908914, 0.02832582]))"
  []
  [old_foo]
    type = Python
    expr = "MRP(torch.tensor([0.03739906, -0.01994617, 0.02991925]))"
  []
  [w]
    type = Python
    expr = "WR2(torch.tensor([0.01, 0.02, -0.03]))"
  []
[]
