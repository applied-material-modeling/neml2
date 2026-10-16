# neml2
# Separated interface driven back into contact. The penalty must survive
# separation -- a failed boundary still cannot interpenetrate -- while the
# shear traction stays zero.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'normal_jump tangential_jump viscous_normal_opening viscous_sliding cavity_fraction'
    # Tn = 2000 * (-0.01) = -20.0, Tt = 0
    input_Scalar_values = '-0.01 0.01 0.004 0.002 0.9'
    output_Scalar_names = 'normal_traction shear_traction'
    output_Scalar_values = '-20.0 0.0'
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
