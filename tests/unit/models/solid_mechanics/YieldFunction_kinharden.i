[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'M X'
    input_SR2_values = 'M X'
    output_Scalar_names = 'fp'
    output_Scalar_values = '99.8876'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '100 110 100 50 40 30'
  []
  [X]
    type = FillSR2
    values = '60 -10 20 40 30 -60'
  []
[]

[Models]
  [overstress]
    type = SR2LinearCombination
    to = 'O'
    from = 'M X'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
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
    models = 'overstress vonmises yield'
  []
[]
