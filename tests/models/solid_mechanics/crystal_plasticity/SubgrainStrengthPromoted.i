# SubgrainStrength with all three coefficients promoted to model inputs, so
# the promoted-parameter derivative actions are exercised.
#
# The batch sweeps a zero through each coefficient in turn. A zero coefficient
# is a valid way to switch this contribution off, and the derivative stays
# finite there: d(tau)/d(alpha) = mu*b/lambda is 100 in the first entry even
# though tau itself is 0.
#
# tau = alpha*mu*b/lambda, computed by hand:
#   0.00 * 100 * 2 / 2 =  0
#   0.25 *   0 * 2 / 2 =  0
#   0.25 * 100 * 0 / 2 =  0
#   0.25 *  80 * 2 / 4 = 10
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'subgrain_size alpha_in mu_in b_in'
    input_Scalar_values = 'lambda_vals alpha_vals mu_vals b_vals'
    output_Scalar_names = 'subgrain_strength'
    output_Scalar_values = 'tau_correct'
  []
[]

[Models]
  [model]
    type = SubgrainStrength
    alpha = 'alpha_in'
    shear_modulus = 'mu_in'
    burgers_vector = 'b_in'
  []
[]

[Tensors]
  [lambda_vals]
    type = Python
    expr = "Scalar(torch.tensor([2.0, 2.0, 2.0, 4.0], dtype=torch.float64))"
  []
  [alpha_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.25, 0.25, 0.25], dtype=torch.float64))"
  []
  [mu_vals]
    type = Python
    expr = "Scalar(torch.tensor([100.0, 0.0, 100.0, 80.0], dtype=torch.float64))"
  []
  [b_vals]
    type = Python
    expr = "Scalar(torch.tensor([2.0, 2.0, 0.0, 2.0], dtype=torch.float64))"
  []
  [tau_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.0, 0.0, 10.0], dtype=torch.float64))"
  []
[]
