# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/ResolvedShear.i.
# FCC <110>{111} via a [Data] CubicCrystal. The C++ fixture composes
# RotationMatrix -> ResolvedShear and drives with a MRP; commit 7ad06e395
# fixed the ComposedModel<->ModelUnitTest wrapper-boundary limitation, so we
# can mirror that composition directly here.
#
# Expected per-slip `shears` (12-entry Scalar with sub_batch_ndim=1) carries
# over verbatim from the C++ fixture's Tensor base_shape=(12) reference values;
# they are independent of how `orientation_matrix` is supplied.
[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = '1.2'
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
  [stress]
    type = Python
    expr = 'SR2.fill(100.0, -50.0, 25.0, 30.0, -75.0, 125.0)'
  []
  [R]
    type = Python
    expr = 'MRP(torch.tensor([0.00499066, -0.0249533, 0.03493462], dtype=torch.float64))'
  []
  [shears]
    type = Python
    expr = 'Scalar(torch.tensor([-60.31676597889275, 100.51708798107992, -57.7177866110632, 40.24316689472251, -85.95238135907564, -23.079480904888687, -60.273921086357404, 2.59897936782955, 25.635615380182898, 123.59656888596861, -34.638305706174506, 126.19554825379818]), sub_batch_ndim=1)'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'stress'
    input_SR2_values = 'stress'
    input_MRP_names = 'orientation'
    input_MRP_values = 'R'
    output_Scalar_names = 'resolved_shears'
    output_Scalar_values = 'shears'
    derivative_rel_tol = 1e-4
    derivative_abs_tol = 5e-6
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [tau]
    type = ResolvedShear
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues tau'
  []
[]
