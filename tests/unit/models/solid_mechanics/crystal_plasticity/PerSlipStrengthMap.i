[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_strengths'
    output_Tensor_values = 'strengths'
    input_Tensor_names = 'state/internal/slip_hardening'
    input_Tensor_values = 'hardening'
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
  [hardening]
    type = Tensor
    values = '100.0 100.0 100.0 100.0 100 100 100 100 100 100 100 100'
    base_shape = '(12)'
  []
  [strengths]
    type = Tensor
    values = '150 150 150 150 150 150 150 150 150 150 150 150'
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
    type = PerSlipStrengthMap
    constant_strength = 50.0
  []
[]
