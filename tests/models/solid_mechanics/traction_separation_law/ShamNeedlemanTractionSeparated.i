# neml2
# Completely separated interface (f = 0.9 >= f_F = 0.8) held open. Both
# tractions vanish: the tensile and shear load paths are gone and the
# compressive penalty is inactive because the jump is positive.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_jump tangential_jump viscous_normal_opening viscous_sliding cavity_fraction'
    input_Scalar_values = '0.02 0.01 0.004 0.002 0.9'
    output_Scalar_names = 'normal_traction shear_traction'
    output_Scalar_values = '0.0 0.0'
    value_abs_tol = 1e-12
  []
[]

[Models]
  [model]
    type = ShamNeedlemanTraction
    normal_stiffness = 1000.0
    shear_stiffness = 800.0
    penalty_stiffness = 2000.0
    failure_fraction = 0.8
  []
[]
