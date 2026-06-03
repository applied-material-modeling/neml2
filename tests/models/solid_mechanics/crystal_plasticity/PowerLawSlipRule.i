# Translated from tests/unit/models/.../PowerLawSlipRule.i. LinspaceScalar with
# group='intermediate' -> Scalar(torch.linspace(...)).sub_batch.retag(1); the
# expected rates are a 12-entry per-slip Scalar (intermediate_dimension=1).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'resolved_shears slip_strengths'
    input_Scalar_values = 'tau tau_bar'
    output_Scalar_names = 'slip_rates'
    output_Scalar_values = 'rates'
    value_rel_tol = 1e-4
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Tensors]
  [tau]
    type = Python
    expr = 'Scalar.linspace(-100, 200, 12).sub_batch.retag(1)'
  []
  [tau_bar]
    type = Python
    expr = 'Scalar.linspace(50, 250, 12).sub_batch.retag(1)'
  []
  [rates]
    type = Python
    expr = 'Scalar(torch.tensor([-3.4297e-02, -1.3898e-03, -3.7875e-05, -1.3357e-07, 1.7191e-09, 9.9957e-07, 9.3434e-06, 3.3176e-05, 7.6855e-05, 1.4079e-04, 2.2299e-04, 3.2045e-04]), sub_batch_ndim=1)'
  []
[]

[Models]
  [model]
    type = PowerLawSlipRule
    n = 5.1
    gamma0 = 1e-3
  []
[]
