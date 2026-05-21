# neml2
# 3D exponential cohesive law (Salehani-Irani) under a monotonic mixed-mode opening ramp.
# Traction grows, peaks near the characteristic length, then decays exponentially to zero.
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
  [traction]
    type = SalehaniIraniTraction
    normal_characteristic_length = 1.0
    tangential_characteristic_length = 1.0
    normal_strength = 1.0
    shear_strength = 1.0
  []
  [model]
    type = ComposedModel
    models = 'decompose traction'
  []
[]
