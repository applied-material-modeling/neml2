[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/f1 state/f2'
    input_Scalar_values = 'f1 f2'
    output_Scalar_names = 'state/x'
    output_Scalar_values = 'x'
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [f1]
    type = Scalar
    values = '0.1 0.2 0.3'
    batch_shape = '(3)'
  []
  [f2]
    type = Scalar
    values = '0.05 0.1 0.2'
    batch_shape = '(3)'
  []
  [x]
    type = Scalar
    values = '0.16470588235294118 0.04285714285714286 -0.22'
    batch_shape = '(3)'
  []
[]

[Models]
  [model]
    type = CurrentConcentration
    initial_concentration = 0.25
    precipitate_volume_fractions = 'state/f1 state/f2'
    precipitate_concentrations = '0.8 0.6'
    current_concentration = 'state/x'
  []
[]
