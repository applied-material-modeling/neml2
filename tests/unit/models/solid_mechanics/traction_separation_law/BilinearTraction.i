[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dm di df dn_pos dn_neg ds1 ds2 damage~1'
    # δ_m = 0.5, δ_init = 0.1, δ_final = 1.0, d_old = 0.0
    # d_trial = 1.0 * (0.5 - 0.1) / (0.5 * 0.9) = 0.8888888889
    # damage = max(d_trial, d_old) = 0.8888888889
    # Loading: dn_pos = 0.02, dn_neg = 0, ds1 = 0.01, ds2 = -0.01
    # T_n  = K(1-d) δ_n^+ + K δ_n^- = 1000*0.1111111111*0.02 + 0 = 2.222222222
    # T_s1 = K(1-d) δ_s1 = 1000*0.1111111111*0.01 = 1.111111111
    # T_s2 = K(1-d) δ_s2 = 1000*0.1111111111*(-0.01) = -1.111111111
    input_Scalar_values = '0.5 0.1 1.0 0.02 0.0 0.01 -0.01 0.0'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.8888888889'
    value_abs_tol = 1e-6
    # Damage chain has stiff slope; FD truncation against the bilinear interior dominates.
    derivative_abs_tol = 1e-3
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [T_expected]
    type = Vec
    values = '2.222222222 1.111111111 -1.111111111'
  []
[]

[Models]
  [model]
    type = BilinearTraction
    separation = 'dm'
    activation = 'di'
    saturation = 'df'
    normal_separation = 'dn_pos'
    normal_penetration = 'dn_neg'
    tangential_separation_1 = 'ds1'
    tangential_separation_2 = 'ds2'
    to = 'traction'
    damage = 'damage'
    penalty_stiffness = 1000.0
  []
[]
