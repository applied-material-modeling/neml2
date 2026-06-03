# neml2
# Linear-elastic traction-separation under a monotonic mixed-mode jump ramp.
# Composed from VecComponents + OrthotropicLinearTraction.
# T = diag(K_n, K_t, K_t) * delta. No internal state.
[Tensors]
  [times]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.0, 30)'
  []
  [jumps]
    type = Python
    expr = 'Vec(torch.linspace(0.0, 1.0, 30, dtype=torch.float64).reshape(30, 1) * torch.tensor([0.05, 0.02, -0.01], dtype=torch.float64))'
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
