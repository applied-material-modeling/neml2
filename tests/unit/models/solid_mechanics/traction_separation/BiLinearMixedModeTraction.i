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
    output_Scalar_values = '0.7487630162'
    value_abs_tol = 1e-6
    derivative_abs_tol = 1e-2
  []
[]

[Tensors]
  [jump_old]
    type = Vec
    values = '0.01 0.01 0.0'
  []
  [jump]
    type = Vec
    values = '0.02 0.02 0.02'
  []
  # Mixed-mode opening with GIc = GIIc (BK exponent term cancels):
  #   delta_n0 = N/K = 0.01, delta_s0 = S/K = 0.01
  #   beta ~ 1, delta_init ~ 0.01, delta_final = 2 GIc/(K delta_n0) = 0.2
  #   delta_m = sqrt(3) * 0.02 = 0.034641016
  #   d = 0.2*(delta_m - 0.01)/(delta_m*0.19) = 0.7487630162
  #   T_n = T_t1 = T_t2 = (1 - d)*K*0.02 = 5.0247397
  [traction]
    type = Vec
    values = '5.0247396757 5.0247396757 5.0247396757'
  []
[]

[Models]
  [model]
    type = BiLinearMixedModeTraction
    penalty_stiffness = 1000.0
    normal_fracture_energy = 1.0
    shear_fracture_energy = 1.0
    normal_strength = 10.0
    shear_strength = 10.0
    eta = 2.0
    alpha = 1e-8
  []
[]
