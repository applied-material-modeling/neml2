[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_hardening_rate'
    output_Tensor_values = 'rate'
    input_Tensor_names = 'state/internal/slip_hardening state/internal/slip_rates'
    input_Tensor_values = 'tau_bar slip'
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
    values = '6.666666666666667 -33.3333333333333 6.666666666666667  6.666666666666667 -33.3333333333333 6.666666666666667 6.666666666666667 -33.3333333333333 6.666666666666667 6.666666666666667 -33.3333333333333 6.666666666666667'
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
  [model]
    type = FAPerSlipHardeningRule
    initial_slope = 200.0
    saturated_hardening = 60.0
  []
[]
