[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_hardening_rate'
    output_Tensor_values = 'rate'
    input_Tensor_names = 'state/internal/slip_hardening state/internal/slip_rates'
    input_Tensor_values = 'tau_bar slip'
    input_Scalar_names = 'state/theta0 state/tau_f'
    input_Scalar_values = '200.0 60.0'
  []
[]

[Tensors]
  [a]
    type = Scalar
    values = '1.2'
  []
  [sdirs]
    type = FillMillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = FillMillerIndex
    values = '1 1 1'
  []
  [tau_bar]
    type = Tensor
    values = '40 40 40 40 40 40 40 40 40 40 40 40'
    base_shape = '(12)'
  []
  [slip]
    type = Tensor
    values = '0.1 -0.1 0.1 0.1 -0.1 0.1 0.1 -0.1 0.1 0.1 -0.1 0.1'
    base_shape = '(12)'
  []
  [rate]
    type = Tensor
    values = '6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667 6.666666666666667'
    base_shape = '(12)'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = "a"
    slip_directions = "sdirs"
    slip_planes = "splanes"
  []
[]

[Models]
  [model0]
    type = VocePerSlipHardeningRule
    initial_slope = 'state/theta0'
    saturated_hardening = 'state/tau_f'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
