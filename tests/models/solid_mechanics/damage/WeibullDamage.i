# ModelUnitTest for WeibullDamage:
#   D = 1 - exp[ -(<r - Y_in>_+ / (p1 * Y_in))^p2 ]
#
# Pins the per-class forward output at one active-damage point. A regression
# here would catch a sign flip in the Macaulay clamp, an off-by-one in the
# power exponent, or a mis-wiring of the parameter promotion path.
#
# Chosen point: r = 1953.125 J/m^3, Y_in = 300 J/m^3, p1 = 5, p2 = 2.
# This is exactly the (E=2.5 GPa, eps=1.25 permille, p1=5, p2=2) case
# machine-checked in the Phase 1 Python reference sanity script:
#   psi_0 = 0.5 * E * eps^2 = 1953.125 J/m^3
#   arg   = (1953.125 - 300) / (5 * 300) = 1.102083
#   arg^p2 = 1.102083^2      = 1.214588
#   D     = 1 - exp(-1.214588) = 0.703168

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
  # 1-element batch (batch_shape = (1,), base_shape = ()) rather than fully
  # 0-D — sidesteps a reshape() edge case in the request_AD reverse-blocks
  # helper (neml2/models/input_ad.py:188) that mishandles the zero-batch,
  # zero-base scalar-to-scalar case.
  [r_val]
    type = Python
    expr = 'Scalar(torch.tensor([1953.125], dtype=torch.float64))'
  []
  [D_expected]
    type = Python
    expr = 'Scalar(torch.tensor([0.7031680196428838], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = WeibullDamage
    r = 'r'
    D = 'D'
    Y_in = 300.0
    p1 = 5.0
    p2 = 2.0
  []
[]
