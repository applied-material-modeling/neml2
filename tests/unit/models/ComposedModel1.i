[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/foo state/bar state/foo_rate state/bar_rate old_state/foo old_state/bar forces/t old_forces/t'
    input_scalar_values = '2 -1 5 -3 0 0 1.3 1.1'
    output_scalar_names = 'residual/foo_bar'
    output_scalar_values = '0.6'
  []
[]

[Models]
  [integrate_foo]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/foo'
  []
  [integrate_bar]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/bar'
  []
  [residual_sum]
    type = ScalarLinearCombination
    from_var = 'residual/foo residual/bar'
    to_var = 'residual/foo_bar'
  []
  [model]
    type = ComposedModel
    models = 'integrate_foo integrate_bar residual_sum'
  []
[]
