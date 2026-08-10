# Smooth branch of PerzynaPlasticFlowRateResidual: r = eta * gdot^(1/n) - <f>.
# Three batch entries, chosen to pin the three behaviours that matter:
#   [0] f > 0, gdot exactly (f/eta)^n -- the consistency point, residual 0. This
#       is the property the inverted form has to satisfy: the same root as the
#       explicit PerzynaPlasticFlowRate, whose own unit test uses eta = 150,
#       exponent = 6, f = 50 and gets gdot = 0.0013717421124828527.
#   [1] f > 0, gdot below its root -- residual negative, pushing gdot up.
#   [2] f < 0 (elastic) -- the Macaulay bracket zeroes the driving term, so the
#       residual is eta * gdot^(1/n) alone and drives gdot to zero.
# All three sit far above the default cutoff, so this exercises the fractional
# power; PerzynaPlasticFlowRateResidual_regularized.i covers the tangent-line
# branch below it.
[Tensors]
  [f]
    type = Python
    expr = 'Scalar(torch.tensor([50.0, 50.0, -20.0], dtype=torch.float64))'
  []
  [gdot]
    type = Python
    expr = 'Scalar(torch.tensor([(1.0 / 3.0) ** 6, 1.0e-3, 1.0e-6], dtype=torch.float64))'
  []
  [resid]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, -2.5658350974743058, 15.0], dtype=torch.float64))'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'yield_function flow_rate'
    input_Scalar_values = 'f gdot'
    output_Scalar_names = 'flow_rate_residual'
    output_Scalar_values = 'resid'
  []
[]

[Models]
  [model]
    type = PerzynaPlasticFlowRateResidual
    reference_stress = 150
    exponent = 6
  []
[]
