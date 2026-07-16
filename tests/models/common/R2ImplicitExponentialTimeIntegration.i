# R2 counterpart of WR2ImplicitExponentialTimeIntegration.i: exponential-map
# integration of a full second-order tensor, r = s - exp(dt*s_rate) @ s_n.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'foo_rate foo foo~1'
    input_R2_values = 'w foo old_foo'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_R2_names = 'foo_residual'
    output_R2_values = 'res'
    value_rel_tol = 1e-4
  []
[]

[Models]
  [model]
    type = R2ImplicitExponentialTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]

[Tensors]
  [res]
    type = Python
    expr = 'R2(torch.tensor([[ 0.00103698, -0.00201191,  0.00291373],
                             [-0.00001654,  0.00002999, -0.00200579],
                             [-0.00000395, -0.00001206,  0.00008897]], dtype=torch.float64))'
  []
  [foo]
    type = Python
    expr = 'R2(torch.tensor([[ 1.001, 0.012, -0.003],
                             [-0.002, 1.0,    0.021],
                             [ 0.006, -0.004, 1.0  ]], dtype=torch.float64))'
  []
  [old_foo]
    type = Python
    expr = 'R2(torch.tensor([[1.0,   0.01, 0.0 ],
                             [0.0,   1.0,  0.02],
                             [0.005, 0.0,  1.0 ]], dtype=torch.float64))'
  []
  [w]
    type = Python
    expr = 'R2(torch.tensor([[ 0.0,   0.02, -0.03 ],
                             [-0.01,  0.0,   0.015],
                             [ 0.005, -0.02, 0.0  ]], dtype=torch.float64))'
  []
[]
