# Translated from tests/unit/models/solid_mechanics/kinematics/ThermalDeformationJacobian.i.
# The C++ fixture declares output_R2_*, but the model output is a Scalar
# (Variable<Scalar> & _J), so the native translation uses output_Scalar_*.
# Numerical value: J = 1 + 1e-5 * (400 - 300) = 1.001.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'T'
    input_Scalar_values = '400'
    output_Scalar_names = 'J'
    output_Scalar_values = '1.001'
    derivative_rel_tol = 1e-04
  []
[]

[Models]
  [model]
    type = ThermalDeformationJacobian
    reference_temperature = 300
    temperature = 'T'
    CTE = 1e-5
    jacobian = 'J'
  []
[]
