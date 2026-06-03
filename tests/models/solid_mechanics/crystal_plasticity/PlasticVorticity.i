# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/PlasticVorticity.i.
# FCC <110>{111} via a [Data] CubicCrystal. The C++ fixture composes
# RotationMatrix -> PlasticVorticity and drives with a Rot; commit 7ad06e395
# fixed the ComposedModel<->ModelUnitTest wrapper-boundary limitation, so we
# can mirror that composition directly here.
#
# Slip rates are a 12-entry per-slip Scalar: LinspaceScalar(-0.1, 0.2, 12) with
# group='intermediate' -> Scalar(torch.linspace(...)).with_sub_batch(1).
#
# Expected output wp comes verbatim from the C++ fixture's FillWR2 3-value form
# (axial-vector packing, no Mandel scaling).
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
  [R]
    type = Python
    expr = 'Rot(torch.tensor([0.00499066, -0.0249533, 0.03493462], dtype=torch.float64))'
  []
  [gamma]
    type = Python
    expr = 'Scalar(torch.linspace(-0.1, 0.2, 12)).with_sub_batch(1)'
  []
  [wp]
    type = Python
    expr = 'WR2(torch.tensor([-0.09829713, -0.01517324, 0.09810889]))'
  []
[]

[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Rot_names = 'orientation'
    input_Rot_values = 'R'
    input_Scalar_names = 'slip_rates'
    input_Scalar_values = 'gamma'
    output_WR2_names = 'plastic_vorticity'
    output_WR2_values = 'wp'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [vorticity]
    type = PlasticVorticity
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues vorticity'
  []
[]
