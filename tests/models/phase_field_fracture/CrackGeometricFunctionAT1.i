# Translated from tests/unit/models/phase_field_fracture/CrackGeometricFunctionAT1.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'cracked'
    input_Scalar_names = 'd'
    input_Scalar_values = '0.787'
    output_Scalar_names = 'alpha'
    output_Scalar_values = '0.787'
    derivative_abs_tol = 1e-06
  []
[]

[Models]
  [cracked]
    type = CrackGeometricFunctionAT1
    phase = 'd'
    crack = 'alpha'
  []
[]
