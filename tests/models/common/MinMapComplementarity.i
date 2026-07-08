# Hard-switch (minimum-map) counterpart of common/FBComplementarity.i.
# Uses the production sign config (a: LE, b: GE): r = min(-a, b).
# At a=3.1, b=2.5 -> min(-3.1, 2.5) = -3.1 (a-branch active, far from the kink
# so the finite-difference Jacobian check sees the exact analytic dr/da=-1).
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'a b'
    input_Scalar_values = '3.1 2.5'
    output_Scalar_names = 'complementarity'
    output_Scalar_values = '-3.1'
  []
[]

[Models]
  [model]
    type = MinMapComplementarity
    a_inequality = 'LE'
    b_inequality = 'GE'
  []
[]
