# neml2
# Translated from tests/unit/models/solid_mechanics/traction_separation_law/BenzeggaghKenaneFullSeparation.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'mode_mixity critical_separation normal_separation'
    input_Scalar_values = 'beta_val 0.01 0.5'
    output_Scalar_names = 'full_separation'
    # beta = sqrt(2); beta^2 = 2; beta^2/(1+beta^2) = 2/3; (2/3)^2 = 4/9;
    # GIIc = GIc = 1 -> (GIIc - GIc) term cancels -> term = GIc = 1.
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
    type = Python
    expr = 'Scalar(torch.tensor(1.41421356237))'
  []
[]

[Models]
  [model]
    type = BenzeggaghKenaneFullSeparation
    mode_mixity = 'mode_mixity'
    critical_separation = 'critical_separation'
    penalty_stiffness = 1000.0
    mode_I_fracture_toughness = 1.0
    mode_II_fracture_toughness = 1.0
    shear_strength = 10.0
    eta = 2.0
  []
[]
