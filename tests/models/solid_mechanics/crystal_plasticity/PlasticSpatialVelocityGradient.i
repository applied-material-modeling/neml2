# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/PlasticSpatialVelocityGradient.i.
# FCC <110>{111} via a [Data] CubicCrystal. The C++ fixture composes
# RotationMatrix -> PlasticSpatialVelocityGradient and drives with a Rot;
# commit 7ad06e395 fixed the ComposedModel<->ModelUnitTest wrapper-boundary
# limitation, so we can mirror that composition directly here.
#
# Slip rates are a 12-entry per-slip Scalar (all 1's, mirroring the C++
# LinspaceScalar(start=1, end=1, nstep=12) with group='intermediate'); the
# C++ fixture's batch_shape=(3) on gamma is dropped because the lp is
# independent of any outer batch when gamma is uniform.
#
# Expected output lp comes verbatim from the C++ fixture's FillR2 9-value
# row-major form.
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
    expr = 'Rot(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  [gamma]
    type = Python
    expr = 'Scalar(torch.ones(12, dtype=torch.float64)).with_sub_batch(1)'
  []
  [lp]
    type = Python
    expr = 'R2(torch.tensor([[0.0, -1.6330, -1.6330], [0.0, 0.0, 0.0], [-1.6330, 0.0, 0.0]], dtype=torch.float64))'
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
    output_R2_names = 'plastic_spatial_velocity_gradient'
    output_R2_values = 'lp'
    value_rel_tol = 0
    value_abs_tol = 1e-3
    derivative_rel_tol = 0
    derivative_abs_tol = 1e-3
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [lp_model]
    type = PlasticSpatialVelocityGradient
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues lp_model'
  []
[]
