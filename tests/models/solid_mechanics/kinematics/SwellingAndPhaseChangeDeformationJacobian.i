# Translated from tests/unit/models/solid_mechanics/kinematics/SwellingAndPhaseChangeDeformationJacobian.i.
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'fluid_fraction c'
    input_Scalar_values = '0.9 0.3'
    output_Scalar_names = 'jacobian'
    output_Scalar_values = '1.0153'
    derivative_rel_tol = 1e-04
  []
[]

[Models]
  [model]
    type = SwellingAndPhaseChangeDeformationJacobian
    phase_fraction = 'c'
    swelling_coefficient = 1e-2
    reference_volume_difference = 0.02
    jacobian = 'jacobian'
    fluid_fraction = 'fluid_fraction'
  []
[]
