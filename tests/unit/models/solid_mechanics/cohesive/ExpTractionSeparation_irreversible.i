[Tensors]
  [delta]
    type = Vec
    values = '0.3 0.2 0.1'
  []
  [T_expected]
    type = Vec
    values = '0.441455329405731 0.294303552937154 0.147151776468577'
  []
  [delta_eff_max_expected]
    type = Scalar
    values = '0.5'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'forces/displacement_jump'
    input_Vec_values = 'delta'
    input_Scalar_names = 'old_state/effective_displacement_jump_scalar_max'
    input_Scalar_values = '0.5'
    output_Vec_names = 'state/traction'
    output_Vec_values = 'T_expected'
    output_Scalar_names = 'state/effective_displacement_jump_scalar_max'
    output_Scalar_values = 'delta_eff_max_expected'
    check_second_derivatives = false
  []
[]

[Models]
  [model]
    type = ExpTractionSeparation
    fracture_energy = 1.0
    softening_length_scale = 0.5
    tangential_weight = 0.5
    irreversible_damage = true
  []
[]
