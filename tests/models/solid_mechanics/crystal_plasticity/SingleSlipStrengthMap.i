# FCC <110>{111} geometry via a [Data] CubicCrystal block (the C++ fixture fakes
# the per-slip dimension with intmd_expand and no [Data]; native needs the real
# geometry). slip_strength_i = slip_hardening + constant_strength, per slip.
[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = '1.0'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Tensors]
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 0.0]))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1.0, 1.0, 1.0]))'
  []
  [hardening]
    type = Python
    expr = 'Scalar(7.5)'
  []
  [strengths]
    type = Python
    expr = 'Scalar(torch.full((12,), 57.5), sub_batch_ndim=1)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'slip_hardening'
    input_Scalar_values = 'hardening'
    output_Scalar_names = 'slip_strengths'
    output_Scalar_values = 'strengths'
  []
[]

[Models]
  [model]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
[]
