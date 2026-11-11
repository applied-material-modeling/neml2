# c_dot = -K*(<c0 - c> + delta)^(1/3)*(c - ceq) = -0.5 * (8 + 19)^(1/3) * 0.1 = -0.15
[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'solute_concentration'
    input_Scalar_values = '0.2'
    output_Scalar_names = 'solute_concentration_rate'
    output_Scalar_values = '-0.15'
  []
[]

[Models]
  [model]
    type = SoluteDepletion
    depletion_coefficient = 0.5
    initial_concentration = 8.2
    equilibrium_concentration = 0.1
    seed_concentration = 19.0
  []
[]
