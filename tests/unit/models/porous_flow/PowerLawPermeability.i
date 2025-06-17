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
    values = '0.01 0.65 0.98'
    batch_shape = '(3)'
  []
  [phi0]
    type = Scalar
    values = '0.71 0.35 1'
    batch_shape = '(3)'
  []
  [p]
    type = Scalar
    values = '0 0.4 1.8'
    batch_shape = '(3)'
  []
  [K]
    type = Scalar
    values = '3 3.842902621 2.892865159'
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = PowerLawPermeability
    reference_permeability = 3
    reference_porosity = 'phi0'
    exponent = 'p'
    porosity = 'state/phi'
    permeability = 'state/K'
  []
[]
