# Linear-elastic traction-separation under a monotonic mixed-mode jump ramp.
# Composed from VecComponents + OrthotropicLinearTraction.
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
    to = 'normal_separation tangential_separation_1 tangential_separation_2'
  []
  [linear_traction]
    type = OrthotropicLinearTraction
    normal_stiffness = 1000.0
    tangential_stiffness = 500.0
  []
  [model]
    type = ComposedModel
    models = 'decompose linear_traction'
  []
[]
