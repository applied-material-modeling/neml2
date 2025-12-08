[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'state/internal/slip_rates'
    input_Scalar_values = 'rates'
    output_Scalar_names = 'state/internal/sum_slip_rates'
    output_Scalar_values = '1.91'
    input_with_intrsc_intmd_dims = 'state/internal/slip_rates'
    input_intrsc_intmd_dims = '1'
  []
[]

[Tensors]
  [sdirs]
    type = MillerIndex
    values = '1 1 0'
  []
  [splanes]
    type = MillerIndex
    values = '1 1 1'
  []
  [rates]
    type = Scalar
    values = '-0.2 -0.15 -0.1 -0.05 0.01 0.05 0.1 0.15 0.2 0.25 0.30 0.35'
    batch_shape = '(12)'
    intermediate_dimension = 1
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 1.2
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [model]
    type = SumSlipRates
  []
[]
