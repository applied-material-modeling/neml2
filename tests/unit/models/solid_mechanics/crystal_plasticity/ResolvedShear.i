[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/resolved_shears'
    output_Tensor_values = 'shears'
    input_SR2_names = 'state/internal/cauchy_stress'
    input_SR2_values = 'stress'
    input_Rot_names = 'state/orientation'
    input_Rot_values = 'R'
    derivative_rel_tol = 1e-4
    derivative_abs_tol = 5e-6
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
  [shears]
    type = Tensor
    values = '-60.31676597889275 100.51708798107992 -57.7177866110632 40.24316689472251 -85.95238135907564 -23.079480904888687 -60.273921086357404 2.59897936782955 25.635615380182898  123.59656888596861 -34.638305706174506 126.19554825379818'
    base_shape = '(12)'
  []
  [stress]
    type = FillSR2
    values = '100.0 -50.0 25.0 30.0 -75.0 125.0'
  []
  [R]
    type = FillRot
    values = '0.00499066 -0.0249533 0.03493462'
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
  [euler_rodrigues]
    type = RotationMatrix
    from = 'state/orientation'
    to = 'state/orientation_matrix'
  []
  [tau]
    type = ResolvedShear
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues tau'
  []
[]
