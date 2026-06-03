# Port of ../../neml2-ref/tests/unit/models/common/ComposedModel6.i
#
# Nested ComposedModel: BC composes B and C; ABC composes A and BC. Pins
# that the dependency resolver flattens the nested composition correctly
# and that an inner ComposedModel is itself composable.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'ABC'
    input_Scalar_names = 'v'
    input_Scalar_values = '3'
    output_Scalar_names = 'x_residual v_residual'
    output_Scalar_values = '3 6'
  []
[]

[Models]
  [A]
    type = CopyScalar
    from = 'v'
    to = 'a'
  []
  [B]
    type = CopyScalar
    from = 'v'
    to = 'x_residual'
  []
  [C]
    type = ScalarLinearCombination
    from = 'a v'
    to = 'v_residual'
  []
  [BC]
    type = ComposedModel
    models = 'B C'
  []
  [ABC]
    type = ComposedModel
    models = 'A BC'
  []
[]
