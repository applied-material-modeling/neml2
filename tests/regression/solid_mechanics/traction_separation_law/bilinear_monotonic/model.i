# Bilinear mixed-mode TSL with BK propagation criterion under a monotonic
# mixed-mode opening ramp. The jump is taken from below delta_init through
# the softening regime up to past delta_final, exercising the elastic,
# softening, and fully-degraded branches in one trajectory.
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
    values = '0.05 0.04 0.03'
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
  [delta_s]
    type = ScalarPNorm
    from = 'delta_s1 delta_s2'
    to = 'delta_s'
    exponent = 2.0
  []
  [macaulay_n]
    type = MacaulaySplit
    from = 'delta_n'
    to_positive = 'delta_n_pos'
    to_negative = 'delta_n_neg'
  []
  [beta]
    type = ModeMixity
    normal = 'delta_n_pos'
    tangential = 'delta_s'
    to = 'beta'
  []
  [delta_init]
    type = CamanhoDavilaInitiation
    mixity = 'beta'
    normal = 'delta_n_pos'
    to = 'delta_init'
    penalty_stiffness = 1000.0
    normal_strength = 10.0
    shear_strength = 10.0
  []
  [delta_final]
    type = BKCriterion
    mixity = 'beta'
    initiation = 'delta_init'
    normal = 'delta_n_pos'
    to = 'delta_final'
    penalty_stiffness = 1000.0
    normal_fracture_toughness = 1.0
    shear_fracture_toughness = 1.0
    shear_strength = 10.0
    eta = 2.0
  []
  [delta_m]
    type = ScalarPNorm
    from = 'delta_n_pos delta_s'
    to = 'delta_m'
    exponent = 2.0
  []
  [traction]
    type = BilinearTraction
    separation = 'delta_m'
    activation = 'delta_init'
    saturation = 'delta_final'
    normal_separation = 'delta_n_pos'
    normal_penetration = 'delta_n_neg'
    tangential_separation_1 = 'delta_s1'
    tangential_separation_2 = 'delta_s2'
    to = 'traction'
    damage = 'damage'
    penalty_stiffness = 1000.0
  []
  [model]
    type = ComposedModel
    models = 'decompose delta_s macaulay_n beta delta_init
              delta_final delta_m traction'
    additional_outputs = 'damage'
  []
[]
