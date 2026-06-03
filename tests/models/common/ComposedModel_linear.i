# Port of ../../neml2-ref/tests/unit/models/common/ComposedModel1.i
#
# Smoke test for the basic ComposedModel pattern: two
# ScalarBackwardEulerTimeIntegration children produce ``foo_residual`` and
# ``bar_residual``, which a ScalarLinearCombination sums into
# ``foo_bar_residual``. ModelUnitTest pins the forward output and exercises
# the dependency resolver's linear-chain wiring.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'foo bar foo_rate bar_rate foo~1 bar~1 t t~1'
    input_Scalar_values = '2 -1 5 -3 0 0 1.3 1.1'
    output_Scalar_names = 'foo_bar_residual'
    output_Scalar_values = '0.6'
  []
[]

[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'foo'
    time = 't'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'bar'
    time = 't'
  []
  [residual_sum]
    type = ScalarLinearCombination
    from = 'foo_residual bar_residual'
    to = 'foo_bar_residual'
  []
  [model]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum'
  []
[]
