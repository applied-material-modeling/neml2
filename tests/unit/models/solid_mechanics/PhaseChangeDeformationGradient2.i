[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/vf state/c'
    input_Scalar_values = '0.9 0.3'
    output_R2_names = 'state/F'
    output_R2_values = 'Ft'
    parameter_derivative_rel_tol = 1e-04
  []
[]

[Tensors]
  [Ft]
    type = FillR2
    values = '0.9949514088 0.0   0.0
              0.0   0.9949514088 0.0
              0.0   0.0   0.9949514088'
  []
[]

[Models]
  [model]
    type = PhaseChangeDeformationGradient
    phase_fraction = 'state/c'
    CPE = 1e-2
    CPC = 0.02
    deformation_gradient = 'state/F'
    fluid_fraction = 'state/vf'
    inverse_condition = true
  []
[]
