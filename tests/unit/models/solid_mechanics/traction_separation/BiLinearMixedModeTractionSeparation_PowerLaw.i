[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'displacement_jump displacement_jump~1'
    input_Vec_values = 'jump zero_jump'
    input_Scalar_names = 'damage~1'
    input_Scalar_values = '0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'traction'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.5506295093'
    value_abs_tol = 1e-6
    # Bilinear damage has a stiff slope and the analytic derivative is exact,
    # so the FD-vs-analytic gap is dominated by FD truncation error at h=1e-6.
    derivative_rel_tol = 1e-3
    derivative_abs_tol = 1e-2
  []
[]

[Tensors]
  [jump]
    type = Vec
    values = '0.02 0.01 0.0'
  []
  [zero_jump]
    type = Vec
    values = '0.0 0.0 0.0'
  []
  # Mixed-mode opening with the POWER_LAW criterion (GIc != GIIc, asymmetric strength):
  #   delta_n0 = N/K = 0.01, delta_s0 = S/K = 0.015
  #   beta = 0.5, beta_sq = 0.25
  #   delta_init = delta_n0 * delta_s0 * sqrt(1+0.25) /
  #                sqrt(delta_s0^2 + 0.25*delta_n0^2) = 0.0106066017
  #   Gc_mixed = (1/GIc)^eta + (beta_sq/GIIc)^eta = 1 + 0.015625 = 1.015625
  #   delta_final = (2 + 2*0.25)/(K*delta_init) * Gc_mixed^(-1/eta) = 0.2338821385
  #   delta_m = sqrt(0.02^2 + 0.01^2) = 0.0223606798
  #   d = delta_final * (delta_m - delta_init) /
  #       (delta_m * (delta_final - delta_init)) = 0.5506295093
  [traction]
    type = Vec
    values = '8.9874098149 4.4937049075 0.0'
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
    criterion = 'POWER_LAW'
  []
[]
