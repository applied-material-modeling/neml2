# neml2
# 3D exponential cohesive law (Salehani-Irani) under a monotonic mixed-mode opening ramp.
# Traction grows, peaks near the characteristic length, then decays exponentially to zero.
[Tensors]
  [delta_u0_n]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [delta_u0_t]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [times]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(1.0).dynamic_batch, 40)'
  []
  [jumps]
    type = Python
    expr = 'Vec(torch.linspace(0.0, 1.0, 40, dtype=torch.float64).unsqueeze(-1) * torch.tensor([3.0, 1.5, 0.0], dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Vec_names = 'separation'
    prescribed_Vec_values = 'jumps'
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
  [damage_envelope]
    type = SalehaniIraniDamage
    normal_characteristic_length = 'delta_u0_n'
    tangential_characteristic_length = 'delta_u0_t'
  []
  [damage_update]
    type = BackwardEulerViscousDamage
  []
  [traction]
    type = SalehaniIraniTraction
    normal_characteristic_length = 'delta_u0_n'
    tangential_characteristic_length = 'delta_u0_t'
    normal_strength = 1.0
    shear_strength = 1.0
  []
  [model]
    type = ComposedModel
    models = 'decompose damage_envelope damage_update traction'
    additional_outputs = 'damage'
  []
[]
