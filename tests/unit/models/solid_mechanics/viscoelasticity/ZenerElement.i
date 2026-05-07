[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain viscous_strain'
    input_SR2_values = 'E Ev'
    output_SR2_names = 'stress viscous_strain_rate'
    output_SR2_values = 'S Ev_dot'
  []
[]

[Tensors]
  [E]
    type = FillSR2
    values = '0.01 -0.005 -0.005 0.002 -0.001 0.0005'
  []
  [Ev]
    type = FillSR2
    values = '0.002 -0.001 -0.001 0.0004 -0.0002 0.0001'
  []
  # Einf = 1000, EM = 500, etaM = 100
  # S = (Einf + EM) * E - EM * Ev = 1500 * E - 500 * Ev
  # Row by row:
  # 1500 * 0.01 - 500 * 0.002 = 15 - 1 = 14
  # 1500 * -0.005 - 500 * -0.001 = -7.5 + 0.5 = -7
  # 1500 * -0.005 - 500 * -0.001 = -7
  # 1500 * 0.002 - 500 * 0.0004 = 3 - 0.2 = 2.8
  # 1500 * -0.001 - 500 * -0.0002 = -1.5 + 0.1 = -1.4
  # 1500 * 0.0005 - 500 * 0.0001 = 0.75 - 0.05 = 0.7
  [S]
    type = FillSR2
    values = '14 -7 -7 2.8 -1.4 0.7'
  []
  # Ev_dot = EM * (E - Ev) / etaM = 500 * (E - Ev) / 100 = 5 * (E - Ev)
  # Row by row:
  # 5 * (0.01 - 0.002) = 0.04
  # 5 * (-0.005 - -0.001) = -0.02
  # 5 * (-0.005 - -0.001) = -0.02
  # 5 * (0.002 - 0.0004) = 0.008
  # 5 * (-0.001 - -0.0002) = -0.004
  # 5 * (0.0005 - 0.0001) = 0.002
  [Ev_dot]
    type = FillSR2
    values = '0.04 -0.02 -0.02 0.008 -0.004 0.002'
  []
[]

[Models]
  [model]
    type = ZenerElement
    equilibrium_modulus = 1000
    maxwell_modulus = 500
    maxwell_viscosity = 100
  []
[]
