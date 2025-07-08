[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_rates'
    output_Tensor_values = 'rates'
    input_Tensor_names = 'state/internal/resolved_shears state/internal/slip_strengths'
    input_Tensor_values = 'tau tau_bar'
    input_Scalar_names = 'state/gamma0 state/n'
    input_Scalar_values = '1.0e-3 5.1'
    check_AD_parameter_derivatives = false
    derivative_rel_tol = 0
    derivative_abs_tol = 5e-6
    second_derivative_rel_tol = 0
    second_derivative_abs_tol = 5e-6
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
  [tau]
    type = LinspaceTensor
    start = -100
    end = 200
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10,3)'
  []
  [tau_bar]
    type = LinspaceTensor
    start = 50
    end = 250
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10,3)'
  []
  [rates]
    type = Tensor
    values = '-3.4297e-02 -1.3898e-03 -3.7875e-05 -1.3357e-07 1.7191e-09 9.9957e-07 9.3434e-06 3.3176e-05 7.6855e-05 1.4079e-04 2.2299e-04 3.2045e-04'
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
  [model0]
    type = PowerLawSlipRule
    n = 'state/n'
    gamma0 = 'state/gamma0'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
