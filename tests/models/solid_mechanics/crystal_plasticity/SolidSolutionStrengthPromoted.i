# SolidSolutionStrength with all three coefficients promoted to model inputs,
# so the promoted-parameter derivative actions are exercised.
#
# The batch sweeps a zero through each coefficient in turn. A zero coefficient
# is a valid way to switch this contribution off, and the derivative stays
# finite there: d(tau)/d(alpha) = mu*eps^(3/2)*c^(1/2) is 400 in the first
# entry even though tau itself is 0. The zero misfit strain in the third entry
# also pins the right derivative d(tau)/d(eps) -> 0 rather than inf*0.
#
# tau = alpha*mu*eps^(3/2)*c^(1/2) with c = 0.25, computed by hand:
#   0.0 * 100 * 4^1.5 * 0.5 =   0
#   0.5 *   0 * 4^1.5 * 0.5 =   0
#   0.5 * 100 * 0^1.5 * 0.5 =   0
#   0.5 * 100 *     8 * 0.5 = 200
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'solute_concentration alpha_in mu_in eps_in'
    input_Scalar_values = 'c_vals alpha_vals mu_vals eps_vals'
    output_Scalar_names = 'solid_solution_strength'
    output_Scalar_values = 'tau_correct'
  []
[]

[Models]
  [model]
    type = SolidSolutionStrength
    alpha = 'alpha_in'
    shear_modulus = 'mu_in'
    misfit_strain = 'eps_in'
  []
[]

[Tensors]
  [c_vals]
    type = Python
    expr = "Scalar(torch.full((4,), 0.25, dtype=torch.float64))"
  []
  [alpha_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.5, 0.5, 0.5], dtype=torch.float64))"
  []
  [mu_vals]
    type = Python
    expr = "Scalar(torch.tensor([100.0, 0.0, 100.0, 100.0], dtype=torch.float64))"
  []
  [eps_vals]
    type = Python
    expr = "Scalar(torch.tensor([4.0, 4.0, 0.0, 4.0], dtype=torch.float64))"
  []
  [tau_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 0.0, 0.0, 200.0], dtype=torch.float64))"
  []
[]
