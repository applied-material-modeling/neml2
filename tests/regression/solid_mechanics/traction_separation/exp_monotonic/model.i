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
  [model]
    type = ExpTractionSeparation
    fracture_energy = 2.0
    characteristic_length = 1.0
    tangential_weight = 1.0
    regularizer = 1e-12
    irreversible_damage = true
  []
[]
