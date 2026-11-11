# PowerLawSlipRule with gamma0 and n promoted to model inputs, so the
# promoted-parameter derivative actions are exercised:
#   d gamma_dot / d gamma0 = |tau/tau_bar|^(n-1) * tau/tau_bar
#   d gamma_dot / d n      = gamma_dot * ln|tau/tau_bar|   (0 at tau = 0)
#
# gamma0 and n are plain-batch (one value per batch entry) while the resolved
# shears and strengths are per-slip, so the exponent must broadcast over the
# slip sub-batch axis. Each batch entry includes tau = 0, where the n
# derivative takes its zero limit, and entry 1 zeroes gamma0 (the gamma0
# derivative stays finite there).
#
# gamma_dot = gamma0 * |tau/tau_bar|^(n-1) * tau/tau_bar, by hand:
#   entry 0  n = 3,   gamma0 = 1e-3: [-1.728e-3, 0, 7.8717201166e-5, 1.423828125e-3]
#   entry 1  n = 6.5, gamma0 = 0:    [0, 0, 0, 0]
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'resolved_shears slip_strengths gamma0_in n_in'
    input_Scalar_values = 'tau tau_bar gamma0_vals n_vals'
    output_Scalar_names = 'slip_rates'
    output_Scalar_values = 'rates'
  []
[]

[Models]
  [model]
    type = PowerLawSlipRule
    gamma0 = 'gamma0_in'
    n = 'n_in'
  []
[]

[Tensors]
  [tau]
    type = Python
    expr = "Scalar(torch.tensor([[-60.0, 0.0, 30.0, 90.0], [-20.0, 10.0, 0.0, 150.0]], dtype=torch.float64), sub_batch_ndim=1)"
  []
  [tau_bar]
    type = Python
    expr = "Scalar(torch.tensor([[50.0, 60.0, 70.0, 80.0], [100.0, 100.0, 100.0, 100.0]], dtype=torch.float64), sub_batch_ndim=1)"
  []
  [gamma0_vals]
    type = Python
    expr = "Scalar(torch.tensor([1e-3, 0.0], dtype=torch.float64))"
  []
  [n_vals]
    type = Python
    expr = "Scalar(torch.tensor([3.0, 6.5], dtype=torch.float64))"
  []
  [rates]
    type = Python
    expr = "Scalar(torch.tensor([[-1.728e-3, 0.0, 7.8717201166e-5, 1.423828125e-3], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float64), sub_batch_ndim=1)"
  []
[]
