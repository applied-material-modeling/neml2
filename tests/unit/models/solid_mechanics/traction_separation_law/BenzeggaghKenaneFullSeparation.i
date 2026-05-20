# neml2
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'mode_mixity critical_separation normal_separation'
    input_Scalar_values = 'beta_val 0.01 0.5'
    output_Scalar_names = 'full_separation'
    # β² = 2; β²/(1+β²) = 2/3; (2/3)² = 4/9; GIIc=GIc=1 -> term cancels -> term = GIc = 1.
    # delta_f = 2/(K * delta_c) * term = 2/(1000 * 0.01) * 1 = 0.2
    output_Scalar_values = '0.2'
    # The d(delta_f)/d(delta_c) derivative has magnitude ~20 (delta_f / delta_c, both small),
    # so FD truncation at h=1e-6 leaves ~1e-3 residual.
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
  [bk]
    type = BenzeggaghKenaneFullSeparation
    mode_mixity = 'mode_mixity'
    critical_separation = 'critical_separation'
    penalty_stiffness = 1000.0
    mode_I_fracture_toughness = 1.0
    mode_II_fracture_toughness = 1.0
    shear_strength = 10.0
    eta = 2.0
  []
  [model]
    type = ComposedModel
    models = 'bk'
  []
[]
