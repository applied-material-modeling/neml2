[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/sy'
    input_Scalar_values = '50'
    input_SR2_names = 'state/internal/M'
    input_SR2_values = 'M'
    output_Scalar_names = 'state/internal/fp'
    output_Scalar_values = '99.8876'
    check_second_derivatives = true
    derivative_abs_tol = 1e-06
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '40 120 80 10 10 90'
  []
[]

[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 'state/sy'
  []
  [model]
    type = ComposedModel
    models = 'vonmises yield'
  []
[]
