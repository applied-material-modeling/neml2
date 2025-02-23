[Tensors]
  [foo]
    type = FillSR2
    values = '1 2 3 4 5 6'
  []
  [bar]
    type = FillSR2
    values = '-1 -4 7 -1 9 1'
  []
  [baz]
    type = FillSR2
    values = '3 10 -11 6 -13 4'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/c_A state/c_B'
    input_Scalar_values = '1.0 -2.0'
    input_SR2_names = 'state/A state/substate/B'
    input_SR2_values = 'foo bar'
    output_SR2_names = 'state/outsub/C'
    output_SR2_values = 'baz'
  []
[]

[Models]
  [model0]
    type = SR2LinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
    coefficients = 'state/c_A state/c_B'
    coefficient_as_parameter = 'true true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
