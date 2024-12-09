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
    values = '0 -2 10 3 14 7'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_symr2_names = 'state/A state/substate/B'
    input_symr2_values = 'foo bar'
    output_symr2_names = 'state/outsub/C'
    output_symr2_values = 'baz'
  []
[]

[Models]
  [model]
    type = SR2LinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
  []
[]
