# Translated from tests/unit/models/solid_mechanics/crystal_plasticity/PlasticDeformationRate.i.
# FCC <110>{111} via a [Data] CubicCrystal. The C++ fixture composes
# RotationMatrix -> PlasticDeformationRate and drives with a Rot; commit 7ad06e395
# fixed the ComposedModel<->ModelUnitTest wrapper-boundary limitation, so we
# can mirror that composition directly here.
#
# Slip rates are a 12-entry per-slip Scalar: LinspaceScalar(-0.1, 0.2, 12) with
# group='intermediate' -> Scalar(torch.linspace(...)).sub_batch.retag(1).
#
# Expected output dp comes verbatim from the C++ fixture's FillSR2 6-value form,
# which scales the 3 shear slots by √2 into Mandel via SR2.fill (per add-unit).
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
    expr = 'Scalar.linspace(-0.1, 0.2, 12).sub_batch.retag(1)'
  []
  [dp]
    type = Python
    expr = 'SR2.fill(0.0546068, -0.0421977, -0.0124091, 0.123251, -0.0935403, 0.0278809)'
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
    output_SR2_names = 'plastic_deformation_rate'
    output_SR2_values = 'dp'
    derivative_rel_tol = 0
    derivative_abs_tol = 1e-5
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [dp_model]
    type = PlasticDeformationRate
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues dp_model'
  []
[]
