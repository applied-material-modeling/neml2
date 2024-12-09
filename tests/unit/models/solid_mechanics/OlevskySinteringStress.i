[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/f'
    input_scalar_values = '0.3'
    output_scalar_names = 'state/internal/ss'
    output_scalar_values = '0.27'
    check_second_derivatives = true
    derivative_abs_tol = 1e-06
    parameter_derivative_rel_tol = 1e-3
  []
[]

[Models]
  [model]
    type = OlevskySinteringStress
    surface_tension = 1e-3
    particle_radius = 1e-3
  []
[]
