[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/se state/internal/sp state/internal/f state/internal/k params/sy params/q1 params/q2 params/q3'
    input_scalar_values = '70 30 0.1 20 50 1.5 1.0 2.25'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '0.28441415168201506'
    check_second_derivatives = true
    second_derivative_abs_tol = 1e-3
  []
[]

[Models]
  [sy]
    type = ScalarInputParameter
    from = 'params/sy'
  []
  [q1]
    type = ScalarInputParameter
    from = 'params/q1'
  []
  [q2]
    type = ScalarInputParameter
    from = 'params/q2'
  []
  [q3]
    type = ScalarInputParameter
    from = 'params/q3'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 'sy'
    q1 = 'q1'
    q2 = 'q2'
    q3 = 'q3'
    isotropic_hardening = 'state/internal/k'
  []
  [model]
    type = ComposedModel
    models = 'yield'
  []
[]
