# neml2
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'mode_mixity critical_separation normal_separation'
    input_Scalar_values = 'beta_val 0.0106066017 0.5'
    output_Scalar_names = 'full_separation'
    # β = 0.5, β² = 0.25; GIc=1, GIIc=2, η=2.
    # Gc_mixed = (1/1)² + (0.25/2)² = 1 + 0.015625 = 1.015625
    # Gc_mixed^(-1/η) = 1/sqrt(1.015625) = 0.99229558
    # delta_f = (2 + 2*0.25)/(1000*0.0106066017) * 0.99229558
    #         = 2.5/10.6066017 * 0.99229558 = 0.2338821385
    output_Scalar_values = '0.2338821385'
    # The d(delta_f)/d(delta_c) derivative has magnitude ~22 (delta_f / delta_c), so FD
    # truncation at h=1e-6 leaves ~1e-3 residual.
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
  [pl]
    type = PowerLawFullSeparation
    mode_mixity = 'mode_mixity'
    critical_separation = 'critical_separation'
    penalty_stiffness = 1000.0
    mode_I_fracture_toughness = 1.0
    mode_II_fracture_toughness = 2.0
    shear_strength = 15.0
    eta = 2.0
  []
  [model]
    type = ComposedModel
    models = 'pl'
  []
[]
