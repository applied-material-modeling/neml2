[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Tensor_names = 'state/internal/slip_rates'
    input_Tensor_values = 'rates'
    output_Scalar_names = 'state/internal/sum_slip_rates'
    output_Scalar_values = 'sum'
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
  [sum]
    type = Scalar
    values = '1.91'
  []
  [rates]
    type = Tensor
    values = '-0.2 -0.15 -0.1 -0.05 0.01 0.05 0.1 0.15 0.2 0.25 0.30 0.35'
    base_shape = '(12)'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [model]
    type = SumSlipRates
  []
[]
