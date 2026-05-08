[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Vec_names = 'displacement_jump'
    input_Vec_values = 'jump'
    output_Vec_names = 'traction'
    output_Vec_values = 'traction'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [jump]
    type = Vec
    values = '0.5 0.5 0.5'
  []
  # With delta_u0_n = delta_u0_t = 1, Tmax_n = Tmax_t = 1:
  #   b_n  = 0.5
  #   b_s1 = 0.5/sqrt(2) = sqrt(2)/4
  #   b_s2 = sqrt(2)/4
  #   x    = 0.5 + 0.125 + 0.125 = 0.75
  # T_n  = e          * 0.5         * exp(-0.75) = 0.5 * exp(0.25)  ≈ 0.6420127083
  # T_s1 = sqrt(2e)   * sqrt(2)/4   * exp(-0.75) = 0.5 * exp(-0.25) ≈ 0.3894003915
  # T_s2 = same as T_s1
  [traction]
    type = Vec
    values = '0.6420127083 0.3894003915 0.3894003915'
  []
[]

[Models]
  [model]
    type = SalehaniIrani3DCTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    maximum_normal_traction = 1.0
    maximum_shear_traction = 1.0
  []
[]
