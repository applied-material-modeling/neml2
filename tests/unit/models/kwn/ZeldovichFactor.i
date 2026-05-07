[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R_crit state/T'
    input_Scalar_values = 'R_crit T'
    output_Scalar_names = 'state/Z'
    output_Scalar_values = 'Z'
    check_AD_parameter_derivatives = true
  []
[]

[Tensors]
  [R_crit]
    type = Scalar
    values = '2.0'
  []
  [gamma]
    type = Scalar
    values = '4.0'
  []
  [T]
    type = Scalar
    values = '1.0'
  []
  [Z]
    type = Scalar
    values = '0.07957747154594767'
  []
[]

[Models]
  [model]
    type = ZeldovichFactor
    critical_radius = 'state/R_crit'
    surface_energy = 'gamma'
    temperature = 'state/T'
    molar_volume = 2.0
    avogadro_number = 2.0
    boltzmann_constant = 1.0
    zeldovich_factor = 'state/Z'
  []
[]
