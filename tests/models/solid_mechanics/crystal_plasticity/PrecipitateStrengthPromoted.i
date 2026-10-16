# PrecipitateStrength with all three coefficients promoted to model inputs, so
# the promoted-parameter derivative actions are exercised.
#
# The batch sweeps a zero through each coefficient in turn. A zero coefficient
# is a valid way to switch this contribution off, and the derivative stays
# finite there: d(tau)/d(alpha) = mu*b/(2*R*N) is 12.5 in the first entry even
# though tau itself is 0.
#
# tau = alpha*mu*b/(2*R*N) with R = 2 and N = 4, computed by hand:
#   0.0 * 100 * 2 / 16 = 0
#   0.4 *   0 * 2 / 16 = 0
#   0.4 * 100 * 0 / 16 = 0
#   0.4 * 100 * 2 / 16 = 5
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'precipitate_radius precipitate_density alpha_in mu_in b_in'
    input_Scalar_values = 'R_vals N_vals alpha_vals mu_vals b_vals'
    output_Scalar_names = 'precipitate_strength'
    output_Scalar_values = 'tau_correct'
  []
[]

[Models]
  [model]
    type = PrecipitateStrength
    alpha = 'alpha_in'
    shear_modulus = 'mu_in'
    burgers_vector = 'b_in'
  []
[]

[Tensors]
  [R_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 2.0, dtype=torch.float64))"
  []
  [N_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 4.0, dtype=torch.float64))"
  []
  [alpha_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.4, 0.4, 0.4], dtype=torch.float64))"
  []
  [mu_vals]
    type = Python
    expr = "Scalar(torch.tensor([100.0, 0.0, 100.0, 100.0], dtype=torch.float64))"
  []
  [b_vals]
    type = Python
    expr = "Scalar(torch.tensor([2.0, 2.0, 0.0, 2.0], dtype=torch.float64))"
  []
  [tau_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.0, 0.0, 5.0], dtype=torch.float64))"
  []
[]
