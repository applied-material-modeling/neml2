[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'beta delta_init dn_pos'
    input_Scalar_values = 'beta_val 0.0106066017 0.5'
    output_Scalar_names = 'delta_final'
    # β = 0.5, β² = 0.25; GIc=1, GIIc=2, η=2.
    # Gc_mixed = (1/1)² + (0.25/2)² = 1 + 0.015625 = 1.015625
    # Gc_mixed^(-1/η) = 1/sqrt(1.015625) = 0.99229558
    # delta_final = (2 + 2*0.25)/(1000*0.0106066017) * 0.99229558
    #             = 2.5/10.6066017 * 0.99229558 = 0.2338821385
    output_Scalar_values = '0.2338821385'
    # The d(delta_final)/d(delta_init) derivative has magnitude ~22 (delta_final / delta_init),
    # so FD truncation at h=1e-6 leaves ~1e-3 residual.
    derivative_abs_tol = 1e-3
    derivative_rel_tol = 1e-3
  []
[]

[Tensors]
  [beta_val]
    type = Scalar
    values = '0.5'
  []
[]

[Models]
  [model]
    type = PowerLawCriterion
    mixity = 'beta'
    initiation = 'delta_init'
    normal = 'dn_pos'
    to = 'delta_final'
    penalty_stiffness = 1000.0
    normal_fracture_toughness = 1.0
    shear_fracture_toughness = 2.0
    shear_strength = 15.0
    eta = 2.0
  []
[]
