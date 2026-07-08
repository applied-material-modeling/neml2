# Translated from tests/unit/models/common/WR2ImplicitExponentialTimeIntegration.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_WR2_names = 'foo_rate'
    input_WR2_values = 'w'
    input_MRP_names = 'foo foo~1'
    input_MRP_values = 'foo old_foo'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_MRP_names = 'foo_residual'
    output_MRP_values = 'res'
    value_rel_tol = 1e-4
  []
[]

[Models]
  [model]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]

[Tensors]
  [res]
    type = Python
    expr = 'MRP(torch.tensor([-0.032903, -0.005864, 0.006609], dtype=torch.float64))'
  []
  [foo]
    type = Python
    expr = 'MRP(torch.tensor([0.00499066, -0.0249533, 0.03493462], dtype=torch.float64))'
  []
  [old_foo]
    type = Python
    expr = 'MRP(torch.tensor([0.03739906, -0.01994617, 0.02991925], dtype=torch.float64))'
  []
  [w]
    type = Python
    expr = 'WR2(torch.tensor([0.01, 0.02, -0.03], dtype=torch.float64))'
  []
[]
