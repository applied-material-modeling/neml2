# Mixed-mode derivative test for BiLinearMixedModeTraction.
# Input is away from all degenerate points:
#   delta_s > 0  (avoids cusp in d(delta_m)/d(delta_s) at delta_s=0)
#   d_old = 0, d_trial >> h  (avoids kink in max(d_trial, d_old) at d_trial=d_old)
# This exercises the new d(damage)/d(delta_old) and d(traction)/d(delta_old) Jacobians.
# derivative_rel_tol = 1e-4 accounts for O(h) forward-difference error on the large
# second derivative of d_bilinear (~3000) at this input point.

[Tensors]
  [delta]
    type = Vec
    values = '0.015 0.01 0.005'
  []
  [T_expected]
    type = Vec
    values = '80.147397211404737 53.431598140936494 26.715799070468247'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'forces/displacement_jump old_forces/displacement_jump'
    input_Vec_values = 'delta delta'
    input_Scalar_names = 'old_state/damage forces/t old_forces/t'
    input_Scalar_values = '0.0 1.0 0.0'
    output_Vec_names = 'state/traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'state/damage'
    output_Scalar_values = '0.46568401859063496'
    # check_second_derivatives = false — forward FD truncation on f''' is O(h·|f'''|);
    # the large curvature of d_bilinear (~3000) makes the artifact too large to verify.
    check_second_derivatives = false
    derivative_rel_tol = 1e-4
  []
[]

[Models]
  [model]
    type = BiLinearMixedModeTraction
    penalty_stiffness = 1e4
    mode_I_critical_fracture_energy = 1e3
    mode_II_critical_fracture_energy = 2e3
    normal_strength = 100
    shear_strength = 100
    mixed_mode_exponent = 2.0
    viscosity = 0.0
    alpha = 1e-6
    lag_mode_mixity = true
    lag_displacement_jump = false
    criterion = 'BK'
  []
[]
