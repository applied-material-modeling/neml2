[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/R_crit state/sum'
    input_Scalar_values = 'R_crit sum'
    output_Scalar_names = 'state/beta'
    output_Scalar_values = 'beta'
    check_AD_parameter_derivatives = true
  []
[]

[Tensors]
  [R_crit]
    type = Scalar
    values = '2.0'
  []
  [sum]
    type = Scalar
    values = '4.0'
  []
  [beta]
    type = Scalar
    values = '12.566370614359172'
  []
[]

[Models]
  [model]
    type = KineticFactor
    critical_radius = 'state/R_crit'
    projected_diffusivity_sum = 'state/sum'
    molar_volume = 1.0
    avogadro_number = 1.0
    kinetic_factor = 'state/beta'
  []
[]
