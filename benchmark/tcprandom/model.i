# neml2
[Settings]
  example_batch_shape = '(${nbatch},)'
[]

[Tensors]
  # HIT-substituted shim so the verbatim triple-quoted Python blocks below
  # can reference nbatch as a bare identifier. ${...} substitution only
  # works inside single-line single-quoted HIT strings; triple-quoted
  # blocks are passed verbatim to the Python eval namespace.
  [nbatch]
    type = Python
    expr = '${nbatch}'
  []
  # end_time = FullScalar(5000) batched (nbatch,) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.full((nbatch,), 5000.0, dtype=torch.float64))
    '''
  []
  # times = LinspaceScalar(0, end_time, 500) -> shape (500, nbatch)
  [times]
    type = Python
    expr = '''
      Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 500, dtype=torch.float64).unsqueeze(-1)
      )
    '''
  []
  # deformation_rate_single = FillSR2(dxx=0.0, dyy=0.0001, dzz=-0.0001) batched (nbatch,)
  [deformation_rate_single]
    type = Python
    expr = '''
      SR2.fill(0.0, 0.0001, -0.0001, 0.0, 0.0, 0.0).dynamic_batch.expand(nbatch)
    '''
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 500) -> shape (500, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = '''
      SR2(deformation_rate_single.data.unsqueeze(0).expand(500, nbatch, 6).contiguous())
    '''
  []

  # vorticity_single = FillWR2(0, 0, 0) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = '''
      WR2(torch.zeros(nbatch, 3, dtype=torch.float64))
    '''
  []
  # vorticity = LinspaceWR2(w_single, w_single, 500) -> shape (500, nbatch, 3)
  [vorticity]
    type = Python
    expr = '''
      WR2(vorticity_single.data.unsqueeze(0).expand(500, nbatch, 3).contiguous())
    '''
  []

  # Crystal geometry inputs: lattice parameter + slip direction + slip plane
  [a]
    type = Python
    expr = '''
      Scalar(torch.tensor([1.0], dtype=torch.float64))
    '''
  []
  [sdirs]
    type = Python
    expr = '''
      MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))
    '''
  []
  [splanes]
    type = Python
    expr = '''
      MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))
    '''
  []
  # initial_orientation = Orientation(input_type='random', quantity=nbatch, normalize=True)
  # Replicates v2 Rot::rand using Shoemake's uniform unit-quaternion sampler,
  # converts to a rotation matrix, then to a modified-Rodrigues parameter
  # vector, then performs the ``normalize=True`` shadow swap that pushes
  # any |r|>=1 MRP into its shadow ``-r / |r|^2``. The driver's old
  # ``random_seed=25`` option is gone in v3, so the seed is set inline
  # here to keep the scenario deterministic.
  [initial_orientation]
    type = Python
    expr = '''
      gen = torch.Generator(device=torch.get_default_device()).manual_seed(25)
      u = torch.rand((nbatch, 3), dtype=torch.float64, generator=gen)
      two_pi = 2.0 * torch.pi
      w = torch.sqrt(1.0 - u[:, 0]) * torch.sin(two_pi * u[:, 1])
      x = torch.sqrt(1.0 - u[:, 0]) * torch.cos(two_pi * u[:, 1])
      y = torch.sqrt(u[:, 0]) * torch.sin(two_pi * u[:, 2])
      z = torch.sqrt(u[:, 0]) * torch.cos(two_pi * u[:, 2])
      M00 = 1.0 - 2.0 * (y * y + z * z)
      M01 = 2.0 * (x * y - z * w)
      M02 = 2.0 * (x * z + y * w)
      M10 = 2.0 * (x * y + z * w)
      M11 = 1.0 - 2.0 * (x * x + z * z)
      M12 = 2.0 * (y * z - x * w)
      M20 = 2.0 * (x * z - y * w)
      M21 = 2.0 * (y * z + x * w)
      M22 = 1.0 - 2.0 * (x * x + y * y)
      trace = M00 + M11 + M22
      theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
      scale = torch.where(theta == 0, torch.zeros_like(theta), torch.tan(theta / 2.0) / (2.0 * torch.sin(theta)))
      rx = (M21 - M12) * scale
      ry = (M02 - M20) * scale
      rz = (M10 - M01) * scale
      ns = rx * rx + ry * ry + rz * rz
      f = torch.sqrt(ns + 1.0) + 1.0
      r = torch.stack([rx / f, ry / f, rz / f], dim=-1)
      r_ns = (r * r).sum(-1, keepdim=True)
      r_norm = torch.where(r_ns < 1.0, r, -r / r_ns)
      result = MRP(r_norm.contiguous())
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    prescribed_SR2_names = 'deformation_rate'
    prescribed_SR2_values = 'deformation_rate'
    prescribed_WR2_names = 'vorticity'
    prescribed_WR2_values = 'vorticity'
    ic_MRP_names = 'orientation'
    ic_MRP_values = 'initial_orientation'
  []
[]

[Data]
  [crystal_geometry]
    type = CubicCrystal
    lattice_parameter = 'a'
    slip_directions = 'sdirs'
    slip_planes = 'splanes'
  []
[]

[Models]
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [orientation_rate]
    type = OrientationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 6.0
    gamma0 = 1.0
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 30.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 10.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'elastic_strain'
  []
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'euler_rodrigues elasticity orientation_rate resolved_shear elastic_stretch plastic_deformation_rate plastic_spin sum_slip_rates slip_rule slip_strength voce_hardening integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    linesearch_cutback = 2.0
    linesearch_stopping_criteria = 1.0e-3
    max_linesearch_iterations = 5
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [cp_warmup_1]
    type = CrystalPlasticityStrainPredictor
    scale = 0.05
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_MRP = 'orientation'
    unknowns_Scalar = 'slip_hardening'
  []
  [predictor]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [fix_orientation]
    type = FixOrientation
    input = 'orientation'
    output = 'orientation'
  []
  [model_with_stress]
    type = ComposedModel
    # `fix_orientation` declares `priority="high"` on its output, so the
    # resolver lifts the duplicate-provider error against `model`'s
    # `orientation` and runs fix_orientation last -- its shadow-swapped
    # value is the one returned to the driver.
    models = 'model fix_orientation elasticity'
    additional_outputs = 'elastic_strain'
  []
[]
