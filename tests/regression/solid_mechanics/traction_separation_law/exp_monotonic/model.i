# Exponential damage TSL under a monotonic mixed-mode opening ramp.
# Damage grows monotonically with the effective separation; traction
# rises elastically then softens exponentially.
[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 40
  []
  [zero_jump]
    type = Vec
    values = '0.0 0.0 0.0'
  []
  [max_jump]
    type = Vec
    values = '3.0 1.5 0.0'
  []
  [jumps]
    type = LinspaceVec
    start = zero_jump
    end = max_jump
    nstep = 40
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
