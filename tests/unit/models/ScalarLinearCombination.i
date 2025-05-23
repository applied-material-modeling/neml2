[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/substate/B'
    input_Scalar_values = '3 2'
    output_Scalar_names = 'state/outsub/C'
    output_Scalar_values = '7'
  []
[]

[Models]
  [model]
    type = ScalarLinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
    constant_coefficient = 2.0
  []
[]
