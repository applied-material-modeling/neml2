# SoluteDepletion with all four coefficients promoted to model inputs, so the
# promoted-parameter derivative actions are exercised, and with the batch
# covering both sides of the Macaulay bracket plus its kink.
#
# c_dot = -K*(<c0 - c> + delta)^(1/3)*(c - ceq), with ceq = 0.1 throughout:
#   entry 0  c < c0:   c = 0.2, c0 = 8.2, delta = 19:     -0.5 * 27^(1/3)  * 0.1 = -0.15
#   entry 1  c > c0:   c = 1.0, c0 = 0.5, delta = 8:      -0.5 * 8^(1/3)   * 0.9 = -0.9
#   entry 2  c = c0:   c = 0.5, c0 = 0.5, delta = 0.125:  -0.5 * 0.125^(1/3) * 0.4 = -0.1
#   entry 3  K = 0,    otherwise as entry 0:                                    = 0
#
# Entry 2 is the physical initial condition: the seed keeps the rate nonzero
# there, so the state can leave c0, and the derivatives stay finite. Entry 1
# is an overshoot reachable mid-Newton. Entry 3 pins the finite
# d(c_dot)/d(K) = -(<c0 - c> + delta)^(1/3)*(c - ceq) = -0.3 at a zeroed
# coefficient.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'solute_concentration K_in c0_in ceq_in delta_in'
    input_Scalar_values = 'c_vals K_vals c0_vals ceq_vals delta_vals'
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
    seed_concentration = 'delta_in'
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
  [delta_vals]
    type = Python
    expr = "Scalar(torch.tensor([19.0, 8.0, 0.125, 19.0], dtype=torch.float64))"
  []
  [rate_correct]
    type = Python
    expr = "Scalar(torch.tensor([-0.15, -0.9, -0.1, 0.0], dtype=torch.float64))"
  []
[]
