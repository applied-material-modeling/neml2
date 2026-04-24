[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'void_fraction'
    input_Scalar_values = '0.3'
    output_Scalar_names = 'sintering_stress'
    output_Scalar_values = '0.27'
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
