[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/se state/internal/sp state/internal/f state/internal/k'
    input_scalar_values = '70 30 0.1 20'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '0.28441415168201506'
    check_second_derivatives = true
    derivative_abs_tol = 1e-06
  []
[]

[Models]
  [model]
    type = GTNYieldFunction
    yield_stress = 50
    q1 = 1.5
    q2 = 1.0
    q3 = 2.25
    isotropic_hardening = 'state/internal/k'
  []
[]
