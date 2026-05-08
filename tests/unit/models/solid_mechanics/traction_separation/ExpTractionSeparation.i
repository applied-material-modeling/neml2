[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'displacement_jump'
    input_Vec_values = 'jump'
    input_Scalar_names = 'effective_displacement_jump_max~1'
    input_Scalar_values = '0.5'
    output_Vec_names = 'traction'
    output_Vec_values = 'traction'
    output_Scalar_names = 'effective_displacement_jump_max'
    output_Scalar_values = '0.9'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [jump]
    type = Vec
    values = '0.4 0.4 0.7'
  []
  # With Gc=2, delta0=1, beta=1, eps=1e-12:
  #   delta_eff_sq = 0.16 + (0.16 + 0.49) = 0.81
  #   delta_eff    = 0.9
  # kappa~1 = 0.5 < 0.9 -> advancing branch -> kappa = 0.9
  # d        = 1 - exp(-0.9)
  # factor   = exp(-0.9) * 2 = 0.4065696597 * 2 = 0.8131393194
  # T_n  = factor * 0.4 = 0.3252557278
  # T_s1 = factor * 0.4 = 0.3252557278
  # T_s2 = factor * 0.7 = 0.5691975236
  [traction]
    type = Vec
    values = '0.3252557278 0.3252557278 0.5691975236'
  []
[]

[Models]
  [model]
    type = ExpTractionSeparation
    fracture_energy = 2.0
    characteristic_length = 1.0
    tangential_weight = 1.0
    regularizer = 1e-12
    irreversible_damage = true
  []
[]
