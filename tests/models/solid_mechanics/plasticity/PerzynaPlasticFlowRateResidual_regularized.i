# Regularized branch of PerzynaPlasticFlowRateResidual: below `cutoff` the
# fractional power hands over to its tangent line at the cutoff, so the residual
# stays C1 with a bounded derivative through zero.
#
# `cutoff` is deliberately 1e-3 here, not the 1e-20 default. At the default the
# tangent-line slope is eta/(n*cutoff) * (cutoff/1)^(1/n) ~ 1e15, which no
# finite-difference derivative check could resolve; 1e-3 puts the same branch at
# a slope of ~7.9e3 and makes it verifiable. The physical default is exercised
# by the sibling smooth-branch test and by the regression scenarios.
#
# Two batch entries:
#   [0] 0 < gdot < cutoff, f > 0 -- the tangent-line branch under load.
#   [1] gdot < 0, f < 0 -- a negative flow rate is unphysical but Newton can
#       propose one, and the linear extension has to stay finite and keep
#       pushing back toward zero rather than producing NaN from a fractional
#       power of a negative number.
[Tensors]
  [f]
    type = Python
    expr = 'Scalar(torch.tensor([30.0, -5.0], dtype=torch.float64))'
  []
  [gdot]
    type = Python
    expr = 'Scalar(torch.tensor([2.0e-4, -2.0e-4], dtype=torch.float64))'
  []
  [resid]
    type = Python
    expr = 'Scalar(torch.tensor([11.109609582188931, 37.947331922020552], dtype=torch.float64))'
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
    cutoff = 1e-3
  []
[]
