# Vec instantiation of VariableRate (no Catch2 fixture exists for the Vec
# template; pattern mirrors CopyVec.i for tensor literals).
# dt = 0.2; foo~1 = 0 so rate = foo / 0.2 componentwise.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'foo foo~1'
    input_Vec_values = 'foo foo_n'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_Vec_names = 'foo_rate'
    output_Vec_values = 'foo_rate'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'Vec(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))'
  []
  [foo_n]
    type = Python
    expr = 'Vec(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  [foo_rate]
    type = Python
    expr = 'Vec(torch.tensor([0.5, -1.0, 1.5], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = VecVariableRate
    variable = 'foo'
    time = 't'
  []
[]
