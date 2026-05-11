[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'displacement_jump displacement_jump~1'
    input_Vec_values = 'jump jump_old'
    input_Scalar_names = 'damage~1'
    input_Scalar_values = '0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'traction'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.3083126495'
    value_abs_tol = 1e-6
    # Bilinear damage has a stiff slope and the analytic derivative is exact;
    # the FD-vs-analytic gap is dominated by FD truncation against the
    # ds2 = 0 kink in delta_m and the active-split kink at dn = 0.
    derivative_rel_tol = 1e-3
    derivative_abs_tol = 5e-2
  []
[]

[Tensors]
  # The current jump drives mode mixity (lag_mode_mixity = false): beta = 0.02/0.02 = 1.
  [jump]
    type = Vec
    values = '0.02 0.02 0.0'
  []
  # The lagged jump drives delta_m (lag_displacement_jump = true).
  [jump_old]
    type = Vec
    values = '0.01 0.01 0.0'
  []
  # Mixed-mode damage with lag_displacement_jump = true (BK), GIc = GIIc:
  #   beta = 1, beta^2 = 1
  #   delta_n0 = delta_s0 = 0.01
  #   delta_init = delta_n0 * delta_s0 * sqrt(2) / sqrt(delta_s0^2 + delta_n0^2) = 0.01
  #   beta_sq_ratio = 0.5; BK exponent term cancels (GIIc - GIc = 0):
  #   delta_final = 2 / (K * delta_init) * GIc = 0.2
  #   delta_m = sqrt(0.01^2 + 0.01^2) = 0.014142135624 (driven by old jump)
  #   d = 0.2 * (delta_m - 0.01) / (delta_m * 0.19) = 0.3083126495
  #   T_n = T_s1 = (1 - d) * K * 0.02 = 13.833747010
  [traction]
    type = Vec
    values = '13.833747010 13.833747010 0.0'
  []
[]

[Models]
  [model]
    type = BiLinearMixedModeTractionSeparation
    penalty_stiffness = 1000.0
    normal_fracture_energy = 1.0
    shear_fracture_energy = 1.0
    normal_strength = 10.0
    shear_strength = 10.0
    eta = 2.0
    criterion = 'BK'
    lag_displacement_jump = true
  []
[]
