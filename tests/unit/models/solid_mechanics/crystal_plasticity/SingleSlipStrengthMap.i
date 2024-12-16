[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_strengths'
    output_Tensor_values = 'strengths'
    input_Scalar_names = 'state/internal/slip_hardening'
    input_Scalar_values = 'hardening'
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
    type = Scalar
    values = '100.0'
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
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
[]
