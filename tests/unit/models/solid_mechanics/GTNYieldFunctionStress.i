[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/f state/internal/k'
    input_scalar_values = '0.1 20'
    input_symr2_names = 'state/internal/M'
    input_symr2_values = 'M'
    output_scalar_names = 'state/internal/fp'
    output_scalar_values = '5.898644517958986'
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
    tensor = 'state/internal/M'
    invariant = 'state/internal/se'
  []
  [i1]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'state/internal/M'
    invariant = 'state/internal/sp'
  []
  [yield]
    type = GTNYieldFunction
    yield_stress = 50
    q1 = 1.5
    q2 = 1.0
    q3 = 2.25
    isotropic_hardening = 'state/internal/k'
  []
  [model]
    type = ComposedModel
    models = 'j2 i1 yield'
  []
[]
