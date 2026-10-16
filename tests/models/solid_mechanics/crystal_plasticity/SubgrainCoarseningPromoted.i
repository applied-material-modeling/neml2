# SubgrainCoarsening with all three coefficients promoted to model inputs, so
# the promoted-parameter derivative actions are exercised.
#
# The batch sweeps a zero through each coefficient in turn. A zero coefficient
# is a valid way to switch coarsening off, and the derivative stays finite
# there: d(lambda_dot)/d(K) = mu*b^3/(k*T*lambda) is 2 in the first entry even
# though the rate itself is 0.
#
# lambda_dot = mu*b^3*K/(k*T*lambda) with k = 0.5, T = 4 and lambda = 2,
# computed by hand (the common factor 1/(k*T*lambda) is 1/4):
#   1 * 2^3 * 0.0 / 4 = 0
#   0 * 2^3 * 3.0 / 4 = 0
#   1 * 0^3 * 3.0 / 4 = 0
#   1 * 2^3 * 3.0 / 4 = 6
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'subgrain_size temperature K_in mu_in b_in'
    input_Scalar_values = 'lambda_vals T_vals K_vals mu_vals b_vals'
    output_Scalar_names = 'subgrain_size_rate'
    output_Scalar_values = 'rate_correct'
  []
[]

[Models]
  [model]
    type = SubgrainCoarsening
    coarsening_coefficient = 'K_in'
    shear_modulus = 'mu_in'
    burgers_vector = 'b_in'
    boltzmann_constant = 0.5
  []
[]

[Tensors]
  [lambda_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 2.0, dtype=torch.float64))"
  []
  [T_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 4.0, dtype=torch.float64))"
  []
  [K_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 3.0, 3.0, 3.0], dtype=torch.float64))"
  []
  [mu_vals]
    type = Python
    expr = "Scalar(torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64))"
  []
  [b_vals]
    type = Python
    expr = "Scalar(torch.tensor([2.0, 2.0, 0.0, 2.0], dtype=torch.float64))"
  []
  [rate_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.0, 0.0, 6.0], dtype=torch.float64))"
  []
[]
