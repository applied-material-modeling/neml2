[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'strain maxwell_viscous_strain kelvin_voigt_strain'
    input_SR2_values = 'E EvM EK'
    output_SR2_names = 'stress maxwell_viscous_strain_rate kelvin_voigt_strain_rate'
    output_SR2_values = 'S EvM_dot EK_dot'
  []
[]

[Tensors]
  [E]
    type = FillSR2
    values = '0.01 -0.005 -0.005 0.002 -0.001 0.0005'
  []
  [EvM]
    type = FillSR2
    values = '0.002 -0.001 -0.001 0.0004 -0.0002 0.0001'
  []
  [EK]
    type = FillSR2
    values = '0.001 -0.0005 -0.0005 0.0002 -0.0001 0.00005'
  []
  # EM = 1000, etaM = 200, EK = 500, etaK = 100
  # S = EM * (E - EvM - EK) = 1000 * (E - EvM - EK)
  # Row by row:
  # 1000 * (0.01 - 0.002 - 0.001) = 1000 * 0.007 = 7
  # 1000 * (-0.005 - -0.001 - -0.0005) = 1000 * -0.0035 = -3.5
  # same: -3.5
  # 1000 * (0.002 - 0.0004 - 0.0002) = 1000 * 0.0014 = 1.4
  # 1000 * (-0.001 - -0.0002 - -0.0001) = 1000 * -0.0007 = -0.7
  # 1000 * (0.0005 - 0.0001 - 0.00005) = 1000 * 0.00035 = 0.35
  [S]
    type = FillSR2
    values = '7 -3.5 -3.5 1.4 -0.7 0.35'
  []
  # EvM_dot = S / etaM = S / 200
  [EvM_dot]
    type = FillSR2
    values = '0.035 -0.0175 -0.0175 0.007 -0.0035 0.00175'
  []
  # EK_dot = (S - EK_param * EK) / etaK = (S - 500 * EK) / 100
  # Row by row:
  # (7 - 500 * 0.001) / 100 = (7 - 0.5) / 100 = 0.065
  # (-3.5 - 500 * -0.0005) / 100 = (-3.5 + 0.25) / 100 = -0.0325
  # same: -0.0325
  # (1.4 - 500 * 0.0002) / 100 = (1.4 - 0.1) / 100 = 0.013
  # (-0.7 - 500 * -0.0001) / 100 = (-0.7 + 0.05) / 100 = -0.0065
  # (0.35 - 500 * 0.00005) / 100 = (0.35 - 0.025) / 100 = 0.00325
  [EK_dot]
    type = FillSR2
    values = '0.065 -0.0325 -0.0325 0.013 -0.0065 0.00325'
  []
[]

[Models]
  [model]
    type = BurgersViscoelasticity
    maxwell_modulus = 1000
    maxwell_viscosity = 200
    kelvin_modulus = 500
    kelvin_viscosity = 100
  []
[]
