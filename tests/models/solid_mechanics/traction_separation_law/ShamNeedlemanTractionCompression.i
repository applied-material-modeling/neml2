# neml2
# Intact interface in compression: the normal load path switches to the
# contact penalty and drops its dependence on the viscous opening, while the
# shear path is untouched.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_jump tangential_jump viscous_normal_opening viscous_sliding cavity_fraction'
    # Tn = 2000 * (-0.01) = -20.0 (viscous_normal_opening does not enter)
    # Tt =  800 * (0.01 - 0.002) = 6.4
    input_Scalar_values = '-0.01 0.01 0.004 0.002 0.6'
    output_Scalar_names = 'normal_traction shear_traction'
    output_Scalar_values = '-20.0 6.4'
    value_abs_tol = 1e-10
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
