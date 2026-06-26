# neml2
# Bilinear mixed-mode TSL under a load-unload-reload schedule.
# Mixed-mode jump ramps up into softening, partially unloads, then reloads
# past the previous peak. Damage must be irreversible (non-decreasing) so
# the unloading limb returns elastically along the secant of the current
# damaged stiffness, and reloading resumes the softening branch only after
# the effective separation passes its previous maximum.
[Tensors]
  [times]
    type = Python
    expr = 'Scalar.linspace(0, 1, 30)'
  []
  [jumps]
    type = Python
    expr = 'Vec(torch.tensor([[0.00333, 0.00333, 0.00333], [0.00667, 0.00667, 0.00667], [0.01, 0.01, 0.01], [0.01333, 0.01333, 0.01333], [0.01667, 0.01667, 0.01667], [0.02, 0.02, 0.02], [0.02333, 0.02333, 0.02333], [0.02667, 0.02667, 0.02667], [0.03, 0.03, 0.03], [0.03333, 0.03333, 0.03333], [0.03667, 0.03667, 0.03667], [0.04, 0.04, 0.04], [0.04333, 0.04333, 0.04333], [0.04667, 0.04667, 0.04667], [0.05, 0.05, 0.05], [0.04625, 0.04625, 0.04625], [0.0425, 0.0425, 0.0425], [0.03875, 0.03875, 0.03875], [0.035, 0.035, 0.035], [0.03125, 0.03125, 0.03125], [0.0275, 0.0275, 0.0275], [0.02375, 0.02375, 0.02375], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.04, 0.04, 0.04], [0.05, 0.05, 0.05], [0.06, 0.06, 0.06], [0.07, 0.07, 0.07], [0.08, 0.08, 0.08], [0.09, 0.09, 0.09]], dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Vec_names = 'separation'
    prescribed_Vec_values = 'jumps'
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
