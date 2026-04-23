[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'flow_invariant poro_invariant void_fraction isotropic_hardening'
    input_Scalar_values = '70 30 0.1 20'
    output_Scalar_names = 'yield_function'
    output_Scalar_values = '0.28441415168201506'
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
    isotropic_hardening = 'isotropic_hardening'
  []
[]
