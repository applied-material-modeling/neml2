[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'stress'
    input_SR2_values = 'S'
    output_SR2_names = 'viscous_strain_rate'
    output_SR2_values = 'Ev_dot'
  []
[]

[Tensors]
  [S]
    type = FillSR2
    values = '100 -50 -50 20 -10 5'
  []
  [Ev_dot]
    type = FillSR2
    values = '0.1 -0.05 -0.05 0.02 -0.01 0.005'
  []
[]

[Models]
  [model]
    type = MaxwellViscoelasticity
    viscosity = 1000
  []
[]
