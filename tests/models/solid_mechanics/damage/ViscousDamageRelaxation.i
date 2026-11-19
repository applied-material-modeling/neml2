# ModelUnitTest for ViscousDamageRelaxation:
#   loading   (target > omega_prev): omega = (omega_prev + mu*dt*target) / (1 + mu*dt)
#   unloading (target <= omega_prev): omega = omega_prev      (damage never heals)
#
# Two batch entries pin both branches. Shared: mu_visc = 2.0, t = 1.5,
# t~1 = 1.0, so dt = 0.5 and mu*dt = 1.0.
#
# Entry 0 -- loading. target = 0.7 > omega_prev = 0.3:
#   omega = (0.3 + 1.0 * 0.7) / (1 + 1.0) = 1.0 / 2 = 0.5
#
# Entry 1 -- unloading. target = 0.1 < omega_prev = 0.6, so damage freezes:
#   omega = omega_prev = 0.6
# The values are picked so the two branches are distinguishable: evaluating
# the loading formula here would give (0.6 + 0.1)/2 = 0.35, not 0.6, so an
# inverted gate cannot pass.
#
# The mu*dt -> infinity rate-independent limit is exercised separately by the
# composed regression cases under tests/regression/.../simo_ju/.

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
    expr = 'Scalar(torch.tensor([0.7, 0.1], dtype=torch.float64))'
  []
  [omega_prev]
    type = Python
    expr = 'Scalar(torch.tensor([0.3, 0.6], dtype=torch.float64))'
  []
  [t_now]
    type = Python
    expr = 'Scalar(torch.tensor([1.5, 1.5], dtype=torch.float64))'
  []
  [t_prev]
    type = Python
    expr = 'Scalar(torch.tensor([1.0, 1.0], dtype=torch.float64))'
  []
  [omega_expected]
    type = Python
    expr = 'Scalar(torch.tensor([0.5, 0.6], dtype=torch.float64))'
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
