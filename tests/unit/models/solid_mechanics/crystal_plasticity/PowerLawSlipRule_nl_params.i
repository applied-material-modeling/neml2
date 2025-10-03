[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    output_Tensor_names = 'state/internal/slip_rates'
    output_Tensor_values = 'rates'
    input_Tensor_names = 'state/internal/resolved_shears state/internal/slip_strengths state/internal/kinematic_hardening state/internal/isotropic_hardening'
    input_Tensor_values = 'tau tau_bar tau_bs tau_iso'
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
  [tau_iso]
    type = LinspaceTensor
    start = 0
    end = 25
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10,3)'
  []
  [tau_bs]
    type = LinspaceTensor
    start = 25
    end = 75
    nstep = 12
    dim = 0
    batch_dim = 0
    batch_expand = '(10,3)'
  []
  [rates]
    type = Tensor
    values = '-0.10702716559171677 -0.00705164298415184 -0.0004869953845627606 -2.3242924726218916e-05 -2.9915361210441915e-07 0 0 9.038072512623785e-09 2.566535783898871e-07 1.4236787819754682e-06 4.274507318768883e-06 9.343423698737752e-06'
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
    kinematic_hardening = 'state/internal/kinematic_hardening'
    isotropic_hardening = 'state/internal/isotropic_hardening'
  []
  [model]
    type = ComposedModel
    models = 'model0'
  []
[]
