# ModelUnitTest for WeibullDamage at p2 < 1 -- the inactive-branch AD guard.
#
# Regression fixture for a real defect: with p2 < 1 the derivative of
# arg^p2 is p2 * arg^(p2-1), which DIVERGES as arg -> 0. Below the damage
# threshold arg is exactly 0, so reverse mode formed inf * 0 and the JVP came
# back nan (and inf exactly at r == Y_in) while the primal was a perfectly
# healthy D = 0. Only p2 >= 1 was previously pinned, so nothing caught it --
# and the sweeps in tests/verification/.../simo_ju/ go down to p2 = 0.01.
#
# WeibullDamage now guards the inactive branch with a double `where`, so the
# power is never differentiated at the singular point.
#
# This fixture uses p2 = 0.01 (the most extreme value the sweeps use) and
# checks BOTH branches. ModelUnitTest verifies the JVP against autograd on
# every entry, so a return of the nan/inf behaviour fails here.
#
#   Y_in = 300, p1 = 1, p2 = 0.01
#
# Entry 0 -- below onset, r = 200 < Y_in:
#   <r - Y_in>_+ = 0  =>  D = 0 exactly, and dD/dr must be FINITE (0).
#
# Entry 1 -- above onset, r = 400:
#   arg    = (400 - 300) / (1 * 300) = 0.333333...
#   arg^p2 = 0.333333^0.01           = 0.98908...
#   D      = 1 - exp(-0.98908...)    = 0.62807907115555340
#
# Note on the kink: the guard's comparison is strict, so at r == Y_in exactly
# the tangent is the left derivative (0) rather than a half-weighted value.
# Both are valid subgradients where the function has a corner. That point is
# deliberately not pinned here -- it is measure-zero and the convention is an
# implementation detail, not a contract.

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
    expr = 'Scalar(torch.tensor([200.0, 400.0], dtype=torch.float64))'
  []
  [D_expected]
    type = Python
    expr = 'Scalar(torch.tensor([0.0, 0.6280790711555534], dtype=torch.float64))'
  []
[]

[Models]
  [model]
    type = WeibullDamage
    r    = 'r'
    D    = 'D'
    Y_in = 300.0
    p1   = 1.0
    p2   = 0.01
  []
[]
