[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/A state/substate/B state/c'
    input_Scalar_values = '3 2 1'
    output_Scalar_names = 'state/outsub/C'
    output_Scalar_values = '6'
  []
[]

[Models]
  [model0]
    type = ScalarLinearCombination
    from_var = 'state/A state/substate/B'
    to_var = 'state/outsub/C'
    coefficients = 'state/c state/c'
    coefficient_as_parameter = 'true true'
    constant_coefficient = 'state/c'
    constant_coefficient_as_parameter = 'true'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
