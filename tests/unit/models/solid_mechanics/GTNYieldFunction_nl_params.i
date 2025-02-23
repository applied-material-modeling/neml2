[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/se state/internal/sp state/internal/f state/internal/k state/sy state/q1 state/q2 state/q3'
    input_Scalar_values = '70 30 0.1 20 50 1.5 1.0 2.25'
    output_Scalar_names = 'state/internal/fp'
    output_Scalar_values = '0.28441415168201506'
    check_second_derivatives = true
    second_derivative_abs_tol = 1e-3
  []
[]

[Models]
  [model0]
    type = GTNYieldFunction
    yield_stress = 'state/sy'
    q1 = 'state/q1'
    q2 = 'state/q2'
    q3 = 'state/q3'
    isotropic_hardening = 'state/internal/k'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
