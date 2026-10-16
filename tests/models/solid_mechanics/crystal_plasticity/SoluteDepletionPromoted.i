# SoluteDepletion with all three coefficients promoted to model inputs, so the
# promoted-parameter derivative actions are exercised, and with the batch
# covering both branches of the Macaulay bracket plus its endpoint.
#
# c_dot = K*<c0 - c>^(1/3)*(c - ceq), with ceq = 0.1 throughout:
#   entry 0  active branch,  c = 0.2 < c0 = 8.2:  0.5 * 8^(1/3) * 0.1 = 0.1
#   entry 1  inactive,       c = 1.0 > c0 = 0.5:  bracket is 0        = 0
#   entry 2  endpoint,       c = 0.5 = c0 = 0.5:  bracket is 0        = 0
#   entry 3  K = 0, active branch                                     = 0
#
# Entries 1 and 2 are the regression: the cube root of a zero bracket has an
# unbounded derivative, so differentiating the primal there used to produce a
# nan rather than the zero this model defines. Entry 3 pins the finite
# d(c_dot)/d(K) = <c0 - c>^(1/3)*(c - ceq) = 0.2 at a zeroed coefficient.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'solute_concentration K_in c0_in ceq_in'
    input_Scalar_values = 'c_vals K_vals c0_vals ceq_vals'
    output_Scalar_names = 'solute_concentration_rate'
    output_Scalar_values = 'rate_correct'
  []
[]

[Models]
  [model]
    type = SoluteDepletion
    depletion_coefficient = 'K_in'
    initial_concentration = 'c0_in'
    equilibrium_concentration = 'ceq_in'
  []
[]

[Tensors]
  [c_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.2, 1.0, 0.5, 0.2], dtype=torch.float64))"
  []
  [K_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.5, 0.5, 0.5, 0.0], dtype=torch.float64))"
  []
  [c0_vals]
    type = Python
    expr = "Scalar(torch.tensor([8.2, 0.5, 0.5, 8.2], dtype=torch.float64))"
  []
  [ceq_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 0.1, dtype=torch.float64))"
  []
  [rate_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.1, 0.0, 0.0, 0.0], dtype=torch.float64))"
  []
[]
