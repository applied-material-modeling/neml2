# Mirrors tests/unit/models/common/ScalarForwardEulerTimeIntegration.i, lifted
# to SR2. dt = t - t~1 = 1.3 - 1.1 = 0.2, so foo = foo~1 + 0.2 * foo_rate.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 't t~1'
    input_Scalar_values = '1.3 1.1'
    input_SR2_names = 'foo_rate foo~1'
    input_SR2_values = 'rate prev'
    output_SR2_names = 'foo'
    output_SR2_values = 'expected'
  []
[]

[Tensors]
  [rate]
    type = Python
    expr = 'SR2.fill(-0.3, 0.5, -1.0, 2.0, -1.5, 0.5)'
  []
  [prev]
    type = Python
    expr = 'SR2.fill(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)'
  []
  [expected]
    type = Python
    expr = 'SR2.fill(0.94, 2.1, 2.8, 4.4, 4.7, 6.1)'
  []
[]

[Models]
  [model]
    type = SR2ForwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
[]
