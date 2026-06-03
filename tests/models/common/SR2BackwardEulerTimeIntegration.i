# Translated by analogy from tests/unit/models/common/ScalarBackwardEulerTimeIntegration.i,
# extended to the SR2 instantiation. dt = t - t~1 = 0.2; residual = foo - foo~1 - dt * foo_rate.
# SR2.fill(a,b,c,d,e,f) applies sqrt(2) Mandel scaling to (d,e,f); since the residual is linear
# in the six fill arguments, the residual SR2 is the per-component combination of the inputs.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'foo_rate foo foo~1'
    input_SR2_values = 'foo_rate foo foo_n'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    output_SR2_names = 'foo_residual'
    output_SR2_values = 'foo_residual'
  []
[]

[Tensors]
  [foo]
    type = Python
    expr = 'SR2.fill(1.1, 2.2, 3.3, 0.4, 0.5, 0.6)'
  []
  [foo_n]
    type = Python
    expr = 'SR2.fill(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)'
  []
  [foo_rate]
    type = Python
    expr = 'SR2.fill(-0.3, -0.6, -0.9, 0.1, 0.2, 0.3)'
  []
  [foo_residual]
    type = Python
    expr = 'SR2.fill(1.16, 2.32, 3.48, 0.38, 0.46, 0.54)'
  []
[]

[Models]
  [model]
    type = SR2BackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]
