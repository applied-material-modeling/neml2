# Translated from tests/unit/models/common/SR2Invariant_I1.i.
# C++ FillSR2 '1 2 3 4 5 6' builds A = [[1,6,5],[6,2,4],[5,4,3]] with the three
# shear slots scaled by sqrt(2) into Mandel; SR2.fill(...) reproduces that.
# I1(A) = tr(A) = 1 + 2 + 3 = 6.
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
    output_Scalar_names = 'internal/I1'
    output_Scalar_values = '6'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = SR2Invariant
    invariant_type = 'I1'
    tensor = 'internal/O'
    invariant = 'internal/I1'
  []
[]
