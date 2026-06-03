# Translated from tests/unit/models/common/SR2Invariant_VONMISES.i.
# The native SR2Invariant only supports invariant_type = VONMISES, so the
# I1 / I2 / EFFECTIVE_STRAIN sibling fixtures have no native counterpart yet.
# C++ FillSR2 '1 2 3 4 5 6' builds A = [[1,6,5],[6,2,4],[5,4,3]] with the three
# shear slots scaled by sqrt(2) into Mandel; SR2.fill(...) reproduces that.
# VONMISES(A) = sqrt(234) ~= 15.2970585408.
[Tensors]
  [foo]
    type = Python
    expr = 'SR2.fill(1, 2, 3, 4, 5, 6)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'internal/O'
    input_SR2_values = 'foo'
    output_Scalar_names = 'internal/VM'
    output_Scalar_values = '15.2970585408'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'internal/O'
    invariant = 'internal/VM'
  []
[]
