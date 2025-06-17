[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/phi'
    input_Scalar_values = 'phi'
    output_Scalar_names = 'state/K'
    output_Scalar_values = 'K'
    check_AD_parameter_derivatives = false
    derivative_abs_tol = 1e-7
  []
[]

[Tensors]
  [phi]
    type = Scalar
    values = "0.01 0.65 0.98"
    batch_shape = '(3)'
  []
  [phio]
    type = Scalar
    values = "0 0.35 1"
    batch_shape = '(3)'
  []
  [m]
    type = Scalar
    values = "0 0.4 1.8"
    batch_shape = '(3)'
  []
  [K]
    type = Scalar
    values = "0.07596492249 6.305733751 0.0"
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = KozenyCarmanPermeability
    reference_permeability = 3
    reference_porosity = 'phio'
    reference_power = 'm'
    power = 0.8
    porosity = 'state/phi'
    permeability = 'state/K'
  []
[]
