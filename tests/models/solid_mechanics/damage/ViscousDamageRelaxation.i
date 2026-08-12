# ModelUnitTest for ViscousDamageRelaxation:
#   loading:   omega = (omega_prev + mu*dt*target) / (1 + mu*dt)
#   unloading: omega = omega_prev
#
# Pins two points: (a) mid-loading step under viscous relaxation, and
# (b) the mu*dt -> infty rate-independent limit (omega -> target).
# Only case (a) is asserted here; the rate-independent limit is exercised
# by the composed simo_ju_viscous.i verification separately.
#
# Chosen point: target = 0.7, omega_prev = 0.3, dt = t - t_prev = 0.5,
# mu_visc = 2.0. Loading (target > omega_prev), so
#   mu*dt = 2 * 0.5 = 1.0
#   omega = (0.3 + 1.0 * 0.7) / (1 + 1.0) = 1.0 / 2 = 0.5

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'target omega~1 t t~1'
    input_Scalar_values = 'tgt omega_prev t_now t_prev'
    output_Scalar_names = 'omega'
    output_Scalar_values = 'omega_expected'
  []
[]

[Tensors]
  [tgt]
    type = Python
    expr = 'Scalar(torch.tensor([0.7], dtype=torch.float64))'
  []
  [omega_prev]
    type = Python
    expr = 'Scalar(torch.tensor([0.3], dtype=torch.float64))'
  []
  [t_now]
    type = Python
    expr = 'Scalar(torch.tensor([1.5], dtype=torch.float64))'
  []
  [t_prev]
    type = Python
    expr = 'Scalar(torch.tensor([1.0], dtype=torch.float64))'
  []
  [omega_expected]
    type = Python
    expr = 'Scalar(torch.tensor([0.5], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = ViscousDamageRelaxation
    target = 'target'
    omega  = 'omega'
    mu_visc = 2.0
  []
[]
