# Translated from tests/unit/models/solid_mechanics/traction_separation_law/CamanhoDavilaCriticalSeparation.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'mode_mixity normal_separation'
    # beta = sqrt(2), normal_separation = 0.5 (any positive value triggers the opening branch)
    input_Scalar_values = 'beta_val 0.5'
    output_Scalar_names = 'critical_separation'
    # K=1000, N=10, S=10 -> delta_n0 = delta_s0 = 0.01.
    # beta^2 = 2 -> delta_mixed = sqrt(0.0003) = 0.0173205081
    # delta_c = 0.01 * 0.01 * sqrt(3) / 0.0173205081 = 0.0100000000
    output_Scalar_values = '0.01'
    derivative_abs_tol = 1e-6
  []
[]

[Tensors]
  [beta_val]
    type = Python
    expr = 'Scalar(torch.tensor(1.41421356237))'
  []
[]

[Models]
  [camanho]
    type = CamanhoDavilaCriticalSeparation
    mode_mixity = 'mode_mixity'
    penalty_stiffness = 1000.0
    normal_strength = 10.0
    shear_strength = 10.0
  []
  [model]
    type = ComposedModel
    models = 'camanho'
  []
[]
