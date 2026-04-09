[Tensors]
  [delta]
    type = Vec
    values = '0.02 0.0 0.0'
  []
  [T_expected]
    type = Vec
    values = '99.94997498749373 0.0 0.0'
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
    output_Scalar_values = '0.5002501250625313'
    check_derivatives = false
    check_second_derivatives = false
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
    lag_mode_mixity = true
    lag_displacement_jump = false
    criterion = 'BK'
  []
[]
