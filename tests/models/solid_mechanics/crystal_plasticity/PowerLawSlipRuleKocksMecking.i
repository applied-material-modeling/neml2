# Temperature-dependent slip rule, wired as in the lumped_km_hardening
# regression: KocksMeckingRateSensitivity n(T) -> SmoothCap -> PowerLawSlipRule
# n. The ModelUnitTest derivative check takes the temperature directional
# derivative of the slip rates through the whole chain (KM, the cap, and the
# slip rule's promoted-n action) and compares it with the autograd JVP.
#
# n = -mu b^3 / (k T A) with mu = 4e4, b = 2.48e-7, k = 1.38064e-20, A = -4.91:
#   T = 300, 900, 1500  ->  n = 30.0007, 10.0002, 6.0001
#   capped (maximum 15, exponent 4)  ->  14.7744, 9.5595, 5.9623
# gamma_dot = 0.2 |tau/100|^(n_eff-1) tau/100 for tau = -90, 40, 110.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'temperature resolved_shears slip_strengths'
    input_Scalar_values = 'T tau tau_bar'
    output_Scalar_names = 'slip_rates'
    output_Scalar_values = 'rates'
    value_rel_tol = 1e-8
  []
[]

[Models]
  [rate_sensitivity]
    type = KocksMeckingRateSensitivity
    temperature = 'temperature'
    rate_sensitivity = 'rate_sensitivity'
    A = -4.91
    shear_modulus = 4.0e4
    k = 1.38064e-20
    b = 2.48e-7
  []
  [cap]
    type = SmoothCap
    argument = 'rate_sensitivity'
    value = 'capped_rate_sensitivity'
    maximum = 15.0
    exponent = 4.0
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 'capped_rate_sensitivity'
    gamma0 = 0.2
  []
  [model]
    type = ComposedModel
    models = 'rate_sensitivity cap slip_rule'
  []
[]

[Tensors]
  [T]
    type = Python
    expr = "Scalar(torch.tensor([300.0, 900.0, 1500.0], dtype=torch.float64))"
  []
  [tau]
    type = Python
    expr = "Scalar(torch.tensor([[-90.0, 40.0, 110.0]] * 3, dtype=torch.float64), sub_batch_ndim=1)"
  []
  [tau_bar]
    type = Python
    expr = "Scalar(torch.full((3, 3), 100.0, dtype=torch.float64), sub_batch_ndim=1)"
  []
  [rates]
    type = Python
    expr = "Scalar(torch.tensor([[-0.0421687617006516, 2.640632244702938e-07, 0.8176770209148808], [-0.07304815840077188, 3.13982287418732e-05, 0.49742220970284107], [-0.10671072600449547, 0.0008479584719673482, 0.35304286865700124]], dtype=torch.float64), sub_batch_ndim=1)"
  []
[]
