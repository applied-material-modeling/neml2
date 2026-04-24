[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'M'
    input_SR2_values = 'M'
    output_Scalar_names = 'fp'
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
    tensor = 'M'
    invariant = 's'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    effective_stress = 's'
    yield_function = 'fp'
  []
  [model]
    type = ComposedModel
    models = 'vonmises yield'
  []
[]
