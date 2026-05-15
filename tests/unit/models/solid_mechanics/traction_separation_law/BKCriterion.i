[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'beta delta_init dn_pos'
    input_Scalar_values = 'beta_val 0.01 0.5'
    output_Scalar_names = 'delta_final'
    # β² = 2; β²/(1+β²) = 2/3; (2/3)² = 4/9; GIIc=GIc=1 -> term cancels -> term = GIc = 1.
    # delta_final = 2/(K * delta_init) * term = 2/(1000 * 0.01) * 1 = 0.2
    output_Scalar_values = '0.2'
    # The d(delta_final)/d(delta_init) derivative has magnitude ~20 (delta_final / delta_init,
    # both small), so FD truncation at h=1e-6 leaves ~1e-3 residual.
    derivative_abs_tol = 1e-3
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [beta_val]
    type = Scalar
    values = '1.41421356237'
  []
[]

[Models]
  [model]
    type = BKCriterion
    mixity = 'beta'
    initiation = 'delta_init'
    normal = 'dn_pos'
    to = 'delta_final'
    penalty_stiffness = 1000.0
    normal_fracture_toughness = 1.0
    shear_fracture_toughness = 1.0
    shear_strength = 10.0
    eta = 2.0
  []
[]
