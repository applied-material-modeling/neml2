[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/substate/B state/c_A state/c_B'
    input_Scalar_values = '3 2 1 2'
    output_Scalar_names = 'state/outsub/C'
    output_Scalar_values = '7'
  []
[]

[Models]
  [model0]
    type = ScalarLinearCombination
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
