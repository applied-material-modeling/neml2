# rho_dot = (k1*sqrt(rho) - k2*rho) * sum|gamma_dot|, with k1 = 2, k2 = 0.5:
#   rho     = [1,    4,   9,    16]
#   hard    = [1.5,  2,   1.5,  0 ]
#   sum     = [0.1,  0.2, 0.3,  0.4]
#   rho_dot = [0.15, 0.4, 0.45, 0 ]
# The density is lumped, so neither input carries a slip-system sub-batch axis.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'dislocation_density sum_slip_rates'
    input_Scalar_values = 'rho gamma_sum'
    output_Scalar_names = 'dislocation_density_rate'
    output_Scalar_values = 'rho_dot'
  []
[]

[Tensors]
  [rho]
    type = Python
    expr = 'Scalar([1.0, 4.0, 9.0, 16.0])'
  []
  [gamma_sum]
    type = Python
    expr = 'Scalar([0.1, 0.2, 0.3, 0.4])'
  []
  [rho_dot]
    type = Python
    expr = 'Scalar([0.15, 0.4, 0.45, 0.0])'
  []
[]

[Models]
  [model]
    type = LumpedForestDislocationEvolution
    k1 = 2.0
    k2 = 0.5
  []
[]
