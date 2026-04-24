[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'flow_invariant poro_invariant void_fraction isotropic_hardening sy q1 q2 q3'
    input_Scalar_values = '70 30 0.1 20 50 1.5 1.0 2.25'
    output_Scalar_names = 'yield_function'
    output_Scalar_values = '0.28441415168201506'
    check_second_derivatives = true
    second_derivative_abs_tol = 1e-3
  []
[]

[Models]
  [model0]
    type = GTNYieldFunction
    yield_stress = 'sy'
    q1 = 'q1'
    q2 = 'q2'
    q3 = 'q3'
    isotropic_hardening = 'isotropic_hardening'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
