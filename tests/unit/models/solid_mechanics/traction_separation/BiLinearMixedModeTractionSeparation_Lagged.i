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
    output_Scalar_values = '0.4232258065'
    value_abs_tol = 1e-6
    # Bilinear damage has a stiff slope (K=1000 with significant softening) and the analytic
    # derivative is exact. The FD-vs-analytic gap is dominated by truncation against the
    # ds2 = 0 kink in delta_m, where the second-order FD term is O(K * h / delta_m).
    derivative_rel_tol = 1e-3
    derivative_abs_tol = 5e-2
  []
[]

[Tensors]
  [jump]
    type = Vec
    values = '0.02 0.01 0.0'
  []
  # The lagged jump drives mode mixity: beta = 0.01/0.005 = 2 (mode II dominated).
  [jump_old]
    type = Vec
    values = '0.005 0.01 0.0'
  []
  # Mixed-mode damage with lag_mode_mixity = true (BK):
  #   beta = 2, beta^2 = 4
  #   delta_n0 = 0.01, delta_s0 = 0.015
  #   delta_init = 0.01*0.015*sqrt(5) / sqrt(0.015^2 + 4*0.01^2) = 0.0134164079
  #   delta_final = 2/(K*delta_init) * (GIc + (GIIc-GIc)*(4/5)^2) = 0.2444767655
  #   delta_m = sqrt(0.02^2 + 0.01^2) = 0.0223606798 (driven by current jump)
  #   d = 0.2444767655 * (delta_m - delta_init) /
  #       (delta_m * (delta_final - delta_init)) = 0.4232258065
  #   T_n = (1-d) * K * 0.02 = 11.5354838710,  T_s1 = (1-d)*K*0.01 = 5.7677419355
  [traction]
    type = Vec
    values = '11.5354838710 5.7677419355 0.0'
  []
[]

[Models]
  [model]
    type = BiLinearMixedModeTractionSeparation
    penalty_stiffness = 1000.0
    normal_fracture_energy = 1.0
    shear_fracture_energy = 2.0
    normal_strength = 10.0
    shear_strength = 15.0
    eta = 2.0
    criterion = 'BK'
    lag_mode_mixity = true
  []
[]
