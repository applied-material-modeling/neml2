# R2 instantiation of VariableRate (no Catch2 fixture exists for the R2
# template; pattern mirrors CopyR2.i for tensor literals).
# dt = 0.2; foo~1 = 0 so rate = foo / 0.2 componentwise.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_R2_names = 'foo foo~1'
    input_R2_values = 'foo foo_n'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_R2_names = 'foo_rate'
    output_R2_values = 'foo_rate'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'R2(torch.tensor([[0.1, 0.04, 0.07], [-0.02, 0.2, 0.05], [0.03, 0.06, 0.3]], dtype=torch.float64))'
  []
  [foo_n]
    type = Python
    expr = 'R2(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64))'
  []
  [foo_rate]
    type = Python
    expr = 'R2(torch.tensor([[0.5, 0.2, 0.35], [-0.1, 1.0, 0.25], [0.15, 0.3, 1.5]], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = R2VariableRate
    variable = 'foo'
    time = 't'
  []
[]
