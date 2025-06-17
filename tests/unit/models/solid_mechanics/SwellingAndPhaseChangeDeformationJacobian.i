[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/vf state/c'
    input_Scalar_values = '0.9 0.3'
    output_Scalar_names = 'state/J'
    output_Scalar_values = 1.0153
    parameter_derivative_rel_tol = 1e-04
  []
[]

[Models]
  [J]
    type = SwellingAndPhaseChangeDeformationJacobian
    phase_fraction = 'state/c'
    swelling_coefficient = 1e-2
    reference_volume_difference = 0.02
    jacobian = 'state/J'
    fluid_fraction = 'state/vf'
  []
  [model]
    type = ComposedModel
    models = 'J'
  []
[]
