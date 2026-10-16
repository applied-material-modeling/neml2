# neml2
# Intact interface under tension. The springs are in series with the viscous
# openings, so the spring sees the jump minus the viscous part.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_jump tangential_jump viscous_normal_opening viscous_sliding cavity_fraction'
    # Tn = 1000 * (0.02 - 0.004) = 16.0
    # Tt =  800 * (0.01 - 0.002) = 6.4
    input_Scalar_values = '0.02 0.01 0.004 0.002 0.6'
    output_Scalar_names = 'normal_traction shear_traction'
    output_Scalar_values = '16.0 6.4'
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
