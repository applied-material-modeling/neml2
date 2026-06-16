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
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, nbatch)
  [times]
    type = Python
    expr = '''
      Scalar(
          end_time.data.unsqueeze(0)
          * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1)
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
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = '''
      SR2(deformation_rate_single.data.unsqueeze(0).expand(100, nbatch, 6).contiguous())
    '''
  []

  # vorticity_single = FillWR2(0, 0, 0) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = '''
      WR2(torch.zeros(nbatch, 3, dtype=torch.float64))
    '''
  []
  # vorticity = LinspaceWR2(w_single, w_single, 100) -> shape (100, nbatch, 3)
  [vorticity]
    type = Python
    expr = '''
      WR2(vorticity_single.data.unsqueeze(0).expand(100, nbatch, 3).contiguous())
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
  # initial_orientation = Orientation(input_type='euler_angles', angle_convention='kocks',
  #                                   angle_type='degrees', values='30 60 45',
  #                                   quantity=nbatch, normalize=True)
  # Replicates v2 Rot::fill_euler_angles (Kocks convention) followed by
  # the ``normalize=True`` shadow swap that pushes any |r|>=1 MRP into its
  # shadow ``-r / |r|^2`` (the same rotation, smaller MRP).
  # The result is a single (3,) Rot expanded to (nbatch, 3).
  [initial_orientation]
    type = Python
    expr = '''
      angles = torch.tensor([30.0, 60.0, 45.0], dtype=torch.float64) * (torch.pi / 180.0)
      a_e, b_e, c_e = angles[0], angles[1], angles[2]
      sa, ca = torch.sin(a_e), torch.cos(a_e)
      sb, cb = torch.sin(b_e), torch.cos(b_e)
      sc, cc = torch.sin(c_e), torch.cos(c_e)
      M = torch.stack([
          torch.stack([-sc * sa - cc * ca * cb,  sc * ca - cc * sa * cb,  cc * sb]),
          torch.stack([ cc * sa - sc * ca * cb, -cc * ca - sc * sa * cb,  sc * sb]),
          torch.stack([ ca * sb,                 sa * sb,                 cb       ]),
      ])
      trace = M[0, 0] + M[1, 1] + M[2, 2]
      theta = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
      scale = torch.where(theta == 0, torch.zeros_like(theta), torch.tan(theta / 2.0) / (2.0 * torch.sin(theta)))
      rx = (M[2, 1] - M[1, 2]) * scale
      ry = (M[0, 2] - M[2, 0]) * scale
      rz = (M[1, 0] - M[0, 1]) * scale
      ns = rx * rx + ry * ry + rz * rz
      f = torch.sqrt(ns + 1.0) + 1.0
      r_single = torch.stack([rx / f, ry / f, rz / f])
      r_ns = (r_single * r_single).sum(-1, keepdim=True)
      r_norm = torch.where(r_ns < 1.0, r_single, -r_single / r_ns)
      result = Rot(r_norm.unsqueeze(0).expand(nbatch, 3).contiguous())
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model_with_stress'
    prescribed_time = 'times'
    force_SR2_names = 'deformation_rate'
    force_SR2_values = 'deformation_rate'
    force_WR2_names = 'vorticity'
    force_WR2_values = 'vorticity'
    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
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
    max_its = 500
    linesearch_cutback = 2.0
    linesearch_stopping_criteria = 1.0e-3
    max_linesearch_iterations = 5
    rel_tol = 1e-4
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
    threshold = 1e-6
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Rot = 'orientation'
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
  [model_with_stress]
    type = ComposedModel
    models = 'model elasticity'
    additional_outputs = 'elastic_strain'
  []
[]
