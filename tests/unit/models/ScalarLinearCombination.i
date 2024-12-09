[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/A state/substate/B'
    input_scalar_values = '3 2'
    output_scalar_names = 'state/outsub/C'
    output_scalar_values = '5'
  []
[]

[Models]
  [model]
    type = ScalarLinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
  []
[]
