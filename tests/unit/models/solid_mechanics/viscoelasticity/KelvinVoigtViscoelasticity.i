[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain strain_rate'
    input_SR2_values = 'E E_dot'
    output_SR2_names = 'stress'
    output_SR2_values = 'S'
  []
[]

[Tensors]
  [E]
    type = FillSR2
    values = '0.01 -0.005 -0.005 0.002 -0.001 0.0005'
  []
  [E_dot]
    type = FillSR2
    values = '0.001 -0.0005 -0.0005 0.0002 -0.0001 0.00005'
  []
  # S = K * E + eta * E_dot = 1000 * E + 500 * E_dot
  [S]
    type = FillSR2
    values = '10.5 -5.25 -5.25 2.1 -1.05 0.525'
  []
[]

[Models]
  [model]
    type = KelvinVoigtViscoelasticity
    modulus = 1000
    viscosity = 500
  []
[]
