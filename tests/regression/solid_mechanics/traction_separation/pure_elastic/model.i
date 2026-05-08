# Linear-elastic traction-separation under a monotonic mixed-mode jump ramp.
# Traction is a constant orthotropic stiffness times the displacement jump:
# T = diag(K_n, K_t, K_t) * delta. No internal state.
[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 30
  []
  [zero_jump]
    type = Vec
    values = '0.0 0.0 0.0'
  []
  [max_jump]
    type = Vec
    values = '0.05 0.02 -0.01'
  []
  [jumps]
    type = LinspaceVec
    start = zero_jump
    end = max_jump
    nstep = 30
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
    type = PureElasticTractionSeparation
    normal_stiffness = 1000.0
    tangential_stiffness = 500.0
  []
[]
