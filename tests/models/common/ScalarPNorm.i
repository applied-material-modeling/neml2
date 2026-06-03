# Translated from tests/unit/models/common/ScalarPNorm.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'x1 x2 x3'
    # Weighted 2-norm with weights = '1 0.5 2':
    #   y = sqrt(1*4^2 + 0.5*4^2 + 2*0^2 + eps) = sqrt(16 + 8) = sqrt(24)
    #     = 4.898979485566...
    input_Scalar_values = '4.0 4.0 0.0'
    output_Scalar_names = 'y'
    output_Scalar_values = '4.898979485566356'
    derivative_abs_tol = 1e-6
  []
[]

[Models]
  [model]
    type = ScalarPNorm
    from = 'x1 x2 x3'
    to = 'y'
    exponent = 2.0
    weights = '1.0 0.5 2.0'
  []
[]
