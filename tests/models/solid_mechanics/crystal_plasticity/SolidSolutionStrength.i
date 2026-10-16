[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'solute_concentration'
    input_Scalar_values = '0.25'
    output_Scalar_names = 'solid_solution_strength'
    output_Scalar_values = '200.0'
  []
[]

[Models]
  [model]
    type = SolidSolutionStrength
    alpha = 0.5
    shear_modulus = 100.0
    misfit_strain = 4.0
  []
[]
