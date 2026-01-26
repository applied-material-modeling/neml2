[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_R2_names = 'state/internal/plastic_spatial_velocity_gradient'
    output_R2_values = 'lp'
    input_Rot_names = 'state/orientation'
    input_Rot_values = 'R'
    input_Scalar_names = 'state/internal/slip_rates'
    input_Scalar_values = 'gamma'
    derivative_rel_tol = 0
    derivative_abs_tol = 1e-3
    second_derivative_rel_tol = 0
    second_derivative_abs_tol = 1e-3
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
  [R]
    type = FillRot
    values = '0 0 0'
  []
  [gamma_a]
    type = FullScalar
    value = '1 1 1'
    batch_shape = (3)
  []
  [gamma_b]
    type = FullScalar
    value = '1 1 1'
    batch_shape = (3)
  []
  [gamma]
    type = LinspaceScalar
    start = 'gamma_a'
    end = 'gamma_b'
    nstep = 12
    group = 'intermediate'
  []
  [lp]
    type = FillR2
    values = '0 -1.6330 -1.6330 0 0 0 -1.6330 0 0'
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
  [wp]
    type = PlasticSpatialVelocityGradient
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues wp'
  []
[]
