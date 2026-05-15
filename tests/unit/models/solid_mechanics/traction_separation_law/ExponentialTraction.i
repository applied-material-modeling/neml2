[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'delta_eff dn ds1 ds2 damage~1'
    # δ_eff = 0.9, d_old = 0.5
    # d_trial = 1 - exp(-0.9) = 0.5934303403
    # damage = max(d_trial, d_old) = 0.5934303403
    # factor = (1-d) * Gc/δ_0² = exp(-0.9) * 2 = 0.8131393194
    # T_n  = factor * 0.4 = 0.3252557278
    # T_s1 = factor * 0.4 = 0.3252557278
    # T_s2 = factor * 0.7 = 0.5691975236
    input_Scalar_values = '0.9 0.4 0.4 0.7 0.5'
    output_Vec_names = 'traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'damage'
    output_Scalar_values = '0.5934303403'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [T_expected]
    type = Vec
    values = '0.3252557278 0.3252557278 0.5691975236'
  []
[]

[Models]
  [model]
    type = ExponentialTraction
    effective_separation = 'delta_eff'
    normal_separation = 'dn'
    tangential_separation_1 = 'ds1'
    tangential_separation_2 = 'ds2'
    to = 'traction'
    damage = 'damage'
    fracture_toughness = 2.0
    characteristic_length = 1.0
  []
[]
