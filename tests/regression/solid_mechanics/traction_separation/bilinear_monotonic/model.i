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
  [model]
    type = BiLinearMixedModeTractionSeparation
    penalty_stiffness = 1000.0
    normal_fracture_energy = 1.0
    shear_fracture_energy = 1.0
    normal_strength = 10.0
    shear_strength = 10.0
    eta = 2.0
  []
[]
