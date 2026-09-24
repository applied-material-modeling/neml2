# ModelUnitTest for WeibullDamage:
#   D = 1 - exp[ -(<r - Y_in>_+ / (p1 * Y_in))^p2 ]
#
# Two batch entries pin both branches of the Macaulay gate.
#
# Entry 0 -- damage active (r > Y_in). r = 1953.125 J/m^3, Y_in = 300 J/m^3,
# p1 = 5, p2 = 2. This is the (E=2.5 GPa, eps=1.25 permille) case:
#   psi_0  = 0.5 * E * eps^2  = 1953.125 J/m^3
#   arg    = (1953.125 - 300) / (5 * 300) = 1.102083
#   arg^p2 = 1.102083^2                   = 1.214588
#   D      = 1 - exp(-1.214588)           = 0.703168
#
# Entry 1 -- below threshold (r < Y_in). r = 200 < 300, so the Macaulay
# bracket zeros the argument and D = 1 - exp(0) = 0 exactly. Chosen well
# below Y_in rather than at r = Y_in: the Macaulay derivative is ambiguous
# exactly at the corner, and this fixture also checks the JVP.

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'r'
    input_Scalar_values = 'r_val'
    output_Scalar_names = 'D'
    output_Scalar_values = 'D_expected'
  []
[]

[Tensors]
  [r_val]
    type = Python
    expr = 'Scalar(torch.tensor([1953.125, 200.0], dtype=torch.float64))'
  []
  [D_expected]
    type = Python
    expr = 'Scalar(torch.tensor([0.7031680196428838, 0.0], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = WeibullDamage
    r    = 'r'
    D    = 'D'
    Y_in = 300.0
    p1   = 5.0
    p2   = 2.0
  []
[]
