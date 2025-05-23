
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'cracked'
    input_Scalar_names = 'state/d'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'state/alpha'
    output_Scalar_values = '0.619369'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
  []
[]


[Models]
  [cracked]
    type = CrackGeometricFunctionAT2
    phase = 'state/d'
    crack = 'state/alpha'
  []
[]