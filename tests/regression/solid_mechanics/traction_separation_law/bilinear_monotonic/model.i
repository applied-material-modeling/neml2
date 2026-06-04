# neml2
# Bilinear mixed-mode TSL with BK propagation criterion under a monotonic
# mixed-mode opening ramp. The jump is taken from below delta_init through
# the softening regime up to past delta_final, exercising the elastic,
# softening, and fully-degraded branches in one trajectory.
[Tensors]
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.0, 40)'
  []
  [jumps]
    type = Python
    expr = 'Vec(torch.linspace(0.0, 1.0, 40, dtype=torch.float64).reshape(40, 1) * torch.tensor([0.05, 0.04, 0.03], dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_Vec_names = 'separation'
    force_Vec_values = 'jumps'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [decompose]
    type = VecComponents
    from = 'separation'
    to = 'signed_normal_separation tangential_separation_1 tangential_separation_2'
  []
  [tangential_separation]
    type = ScalarPNorm
    from = 'tangential_separation_1 tangential_separation_2'
    to = 'tangential_separation'
    exponent = 2.0
  []
  [macaulay_n]
    type = MacaulaySplit
    from = 'signed_normal_separation'
    to_positive = 'normal_separation'
    to_negative = 'normal_penetration'
  []
  [mode_mixity]
    type = ModeMixity
  []
  [critical_separation]
    type = CamanhoDavilaCriticalSeparation
    mode_mixity = 'mode_mixity'
    penalty_stiffness = 1000.0
    normal_strength = 10.0
    shear_strength = 10.0
  []
  [full_separation]
    type = BenzeggaghKenaneFullSeparation
    mode_mixity = 'mode_mixity'
    critical_separation = 'critical_separation'
    penalty_stiffness = 1000.0
    mode_I_fracture_toughness = 1.0
    mode_II_fracture_toughness = 1.0
    shear_strength = 10.0
    eta = 2.0
  []
  [effective_separation]
    type = ScalarPNorm
    from = 'normal_separation tangential_separation'
    to = 'effective_separation'
    exponent = 2.0
  []
  [traction]
    type = BilinearTraction
    critical_separation = 'critical_separation'
    full_separation = 'full_separation'
    normal_penetration = 'normal_penetration'
    penalty_stiffness = 1000.0
  []
  [model]
    type = ComposedModel
    models = 'decompose tangential_separation macaulay_n effective_separation traction'
    additional_outputs = 'damage'
  []
[]
