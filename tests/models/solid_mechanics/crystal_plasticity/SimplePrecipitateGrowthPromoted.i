# SimplePrecipitateGrowth with the coarsening coefficient promoted to a model
# input, so the promoted-parameter derivative actions on both rates are
# exercised.
#
# A zero K is a valid way to switch coarsening off; both rates then vanish but
# their K derivatives stay finite, 1/(3*R^2) = 1/12 and -1/R^3 = -1/8 at R = 2.
#
# With R = 2 and N = 1, computed by hand:
#   R_dot = K/(3*R^2) = K/12  ->  [0.0, 1.0]
#   N_dot = -K/R^3    = -K/8  ->  [0.0, -1.5]
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'precipitate_radius precipitate_density K_in'
    input_Scalar_values = 'R_vals N_vals K_vals'
    output_Scalar_names = 'precipitate_radius_rate precipitate_density_rate'
    output_Scalar_values = 'R_dot_correct N_dot_correct'
  []
[]

[Models]
  [model]
    type = SimplePrecipitateGrowth
    coarsening_coefficient = 'K_in'
  []
[]

[Tensors]
  [R_vals]
    type = Python
    expr = "Scalar(torch.full((2,), 2.0, dtype=torch.float64))"
  []
  [N_vals]
    type = Python
    expr = "Scalar(torch.full((2,), 1.0, dtype=torch.float64))"
  []
  [K_vals]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 12.0], dtype=torch.float64))"
  []
  [R_dot_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, 1.0], dtype=torch.float64))"
  []
  [N_dot_correct]
    type = Python
    expr = "Scalar(torch.tensor([0.0, -1.5], dtype=torch.float64))"
  []
[]
