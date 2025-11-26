[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_WR2_names = 'state/internal/plastic_vorticity'
    output_WR2_values = 'wp'
    input_Rot_names = 'state/orientation'
    input_Rot_values = 'R'
    input_Scalar_names = 'state/internal/slip_rates'
    input_Scalar_values = 'gamma'
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
    second_derivative_rel_tol = 0
    second_derivative_abs_tol = 5e-6
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
    values = '0.00499066 -0.0249533 0.03493462'
  []
  [gamma_a]
    type = FullScalar
    value = '-0.1 -0.1 -0.1'
    batch_shape = (3)
  []
  [gamma_b]
    type = FullScalar
    value = '0.2 0.2 0.2'
    batch_shape = (3)
  []
  [gamma]
    type = LinspaceScalar
    start = 'gamma_a'
    end = 'gamma_b'
    nstep = 12
    group = 'intermediate'
  []
  [wp]
    type = FillWR2
    values = '-0.09829713 -0.01517324 0.09810889'
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
  [vorticity]
    type = PlasticVorticity
  []
  [model]
    type = ComposedModel
    models = 'euler_rodrigues vorticity'
  []
[]
