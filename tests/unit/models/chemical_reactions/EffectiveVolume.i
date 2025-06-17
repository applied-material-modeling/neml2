[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/wb state/ws state/wp state/wg state/phiop'
    input_Scalar_values = 'wb ws wp wg phiop'
    output_Scalar_names = 'state/V'
    output_Scalar_values = 'Vref'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [wb]
    type = Scalar
    values = "0.1 0.7 0.2"
    batch_shape = '(3)'
  []
  [ws]
    type = Scalar
    values = "0.5 0.6 0.2"
    batch_shape = '(3)'
  []
  [wp]
    type = Scalar
    values = "0.1 0.22 0.4"
    batch_shape = '(3)'
  []
  [wg]
    type = Scalar
    values = "0.15 0.23 0.33"
    batch_shape = '(3)'
  []
  [phiop]
    type = Scalar
    values = "0.9 0.01 0.55"
    batch_shape = '(3)'
  []
  [Vref]
    type = Scalar
    values = "0.6024819194 0.09441082427 0.2818082603"
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = EffectiveVolume
    reference_mass = 4.1
    open_volume_fraction = 'state/phiop'
    mass_fractions = 'state/wb state/ws state/wp state/wg'
    densities = '1123 576 988 11'
    composite_volume = 'state/V'
  []
[]
