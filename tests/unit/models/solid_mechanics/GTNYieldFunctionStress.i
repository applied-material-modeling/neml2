[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'void_fraction isotropic_hardening'
    input_Scalar_values = '0.1 20'
    input_SR2_names = 'M'
    input_SR2_values = 'M'
    output_Scalar_names = 'yield_function'
    output_Scalar_values = '5.898644517958986'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '40 120 80 10 10 90'
  []
[]

[Models]
  [j2]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'M'
    invariant = 'se'
  []
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'M'
    invariant = 'sp'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 50
    q1 = 1.5
    q2 = 1.0
    q3 = 2.25
    flow_invariant = 'se'
    poro_invariant = 'sp'
    isotropic_hardening = 'isotropic_hardening'
    yield_function = 'yield_function'
  []
  [model]
    type = ComposedModel
    models = 'j2 i1 yield'
  []
[]
