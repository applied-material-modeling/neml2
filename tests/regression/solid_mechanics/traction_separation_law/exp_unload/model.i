# Exponential damage TSL under a load-unload-reload schedule.
# Mixed-mode jump ramps up, partially unloads, then reloads past the previous peak.
# Damage must be frozen on unloading (irreversible_damage = true) and resume only
# when the effective separation exceeds its historical maximum.
[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 30
  []
  [jumps]
    type = Vec
    batch_shape = '(30)'
    values = '
      0.2 0.1 0   0.4 0.2 0   0.6 0.3 0   0.8 0.4 0   1.0 0.5 0
      1.2 0.6 0   1.4 0.7 0   1.6 0.8 0   1.8 0.9 0   2.0 1.0 0
      2.2 1.1 0   2.4 1.2 0   2.6 1.3 0   2.8 1.4 0   3.0 1.5 0
      2.775 1.3875 0   2.55 1.275 0   2.325 1.1625 0   2.1 1.05 0
      1.875 0.9375 0   1.65 0.825 0   1.425 0.7125 0   1.2 0.6 0
      1.67143 0.835714 0   2.14286 1.07143 0   2.61429 1.30714 0
      3.08571 1.54286 0   3.55714 1.77857 0   4.02857 2.01429 0
      4.5 2.25 0
    '
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_Vec_names = 'displacement_jump'
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
    from = 'displacement_jump'
    to = 'delta_n delta_s1 delta_s2'
  []
  [delta_eff]
    # Weighted Euclidean norm: sqrt(δ_n² + β_w (δ_s1² + δ_s2²) + ε), β_w = 1.0
    type = ScalarPNorm
    from = 'delta_n delta_s1 delta_s2'
    to = 'delta_eff'
    exponent = 2.0
    weights = '1.0 1.0 1.0'
  []
  [traction]
    type = ExponentialTraction
    effective_separation = 'delta_eff'
    normal_separation = 'delta_n'
    tangential_separation_1 = 'delta_s1'
    tangential_separation_2 = 'delta_s2'
    to = 'traction'
    damage = 'damage'
    fracture_toughness = 2.0
    characteristic_length = 1.0
  []
  [model]
    type = ComposedModel
    models = 'decompose delta_eff traction'
  []
[]
