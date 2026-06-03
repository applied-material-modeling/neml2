# SR2 instantiation of VariableRate (no Catch2 fixture exists for the SR2
# template; pattern mirrors SR2BackwardEulerTimeIntegration.i).
# dt = t - t~1 = 0.2; foo~1 = 0, so rate = foo / 0.2 componentwise. Since
# SR2.fill(a, b, c, d, e, f) applies sqrt(2) Mandel scaling uniformly to the
# three shear slots, dividing the raw fill arguments by dt gives the same
# scaling on the rate output.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'foo foo~1'
    input_SR2_values = 'foo foo_n'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_SR2_names = 'foo_rate'
    output_SR2_values = 'foo_rate'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'SR2.fill(0.1, 0.2, 0.3, 0.04, 0.05, 0.06)'
  []
  [foo_n]
    type = Python
    expr = 'SR2.fill(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)'
  []
  [foo_rate]
    type = Python
    expr = 'SR2.fill(0.5, 1.0, 1.5, 0.2, 0.25, 0.3)'
  []
[]

[Models]
  [model]
    type = SR2VariableRate
    variable = 'foo'
    time = 't'
  []
[]
