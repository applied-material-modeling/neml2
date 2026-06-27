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
  # end_time = LinspaceScalar(1, 10, nbatch) -> shape (nbatch,)
  [end_time]
    type = Python
    expr = '''
      Scalar(torch.linspace(1.0, 10.0, nbatch, dtype=torch.float64))
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
  # deformation_rate_single = FillSR2(dxx=0.1, dyy=-0.05, dzz=-0.05) batched (nbatch,)
  [deformation_rate_single]
    type = Python
    expr = '''
      SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(nbatch)
    '''
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, nbatch, 6)
  [deformation_rate]
    type = Python
    expr = '''
      SR2(deformation_rate_single.data.unsqueeze(0).expand(100, nbatch, 6).contiguous())
    '''
  []

  # vorticity_single = FillWR2(w1=0.1, w2=-0.05, w3=-0.05) batched (nbatch,)
  [vorticity_single]
    type = Python
    expr = '''
      WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(nbatch, 3).contiguous())
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

  # Initial orientation = FillRot(R1, R2, R3, method='standard') with
  #   R1 = linspace(0, 0.75, nbatch)
  #   R2 = linspace(0, -0.25, nbatch)
  #   R3 = linspace(-0.1, 0.1, nbatch)
  # Convert standard Rodrigues r_std to modified-Rodrigues parameters via
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (nbatch, 3).
  [initial_orientation]
    type = Python
    expr = '''
      r_std = torch.stack([
          torch.linspace(0.0, 0.75, nbatch, dtype=torch.float64),
          torch.linspace(0.0, -0.25, nbatch, dtype=torch.float64),
          torch.linspace(-0.1, 0.1, nbatch, dtype=torch.float64),
      ], dim=-1)
      result = Rot(r_std / (torch.sqrt((r_std * r_std).sum(-1, keepdim=True) + 1.0) + 1.0))
    '''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'deformation_rate'
    prescribed_SR2_values = 'deformation_rate'
    prescribed_WR2_names = 'vorticity'
    prescribed_WR2_values = 'vorticity'
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
  ############################################################################
  # Sub-system #1: implicit update of elastic_strain + slip_hardening using
  # LAGGED orientation (orientation~1). No cache block needed -- the history
  # input syntax `orientation~1` directly references the previous step's
  # value, mirroring
  # tests/regression/.../single_crystal_decoupled_explicit_orientation/.
  ############################################################################
  [euler_rodrigues]
    type = RotationMatrix
    from = 'orientation~1'
    to = 'orientation_matrix'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [resolved_shear]
    type = ResolvedShear
  []
  [elastic_stretch]
    type = ElasticStrainRate
  []
  [plastic_deformation_rate]
    type = PlasticDeformationRate
  []
  [sum_slip_rates]
    type = SumSlipRates
  []
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = SingleSlipStrengthMap
    constant_strength = 50.0
  []
  [voce_hardening]
    type = VoceSingleSlipHardeningRule
    initial_slope = 500.0
    saturated_hardening = 50.0
  []
  [integrate_slip_hardening]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'slip_hardening'
  []
  [integrate_elastic_strain]
    type = SR2BackwardEulerTimeIntegration
    variable = 'elastic_strain'
  []
  [implicit_rate_1]
    type = ComposedModel
    models = 'euler_rodrigues elasticity resolved_shear
              elastic_stretch plastic_deformation_rate
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain'
  []
[]

[Models]
  ############################################################################
  # Sub-system #2: explicit orientation update using the freshly-updated
  # elastic_strain from subsystem1. Both subsystems share `euler_rodrigues`
  # (computed from orientation~1, the lagged orientation).
  ############################################################################
  [orientation_rate]
    type = OrientationRate
  []
  [plastic_spin]
    type = PlasticVorticity
  []
  [integrate_orientation]
    type = WR2ExplicitExponentialTimeIntegration
    variable = 'orientation'
  []
[]

[EquationSystems]
  [eq_sys_1]
    type = NonlinearSystem
    model = 'implicit_rate_1'
    unknowns = 'elastic_strain slip_hardening'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
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
    scale = 0.1
  []
  [cp_warmup_2]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'slip_hardening'
  []
  [predictor1]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [subsystem1]
    type = ImplicitUpdate
    equation_system = 'eq_sys_1'
    solver = 'newton'
    predictor = 'predictor1'
  []
  [subsystem2]
    type = ComposedModel
    models = 'euler_rodrigues elasticity resolved_shear
              plastic_deformation_rate plastic_spin
              slip_rule slip_strength orientation_rate
              integrate_orientation'
  []

  ############################################################################
  # Sequentially update sub-system #1 and sub-system #2. subsystem1 uses
  # orientation~1 (lagged); subsystem2 outputs the updated orientation.
  # The dependency resolver runs subsystem1 first (subsystem2 depends on
  # elastic_strain + slip_hardening), no priority needed.
  ############################################################################
  [model]
    type = ComposedModel
    models = 'subsystem1 subsystem2'
    additional_outputs = 'elastic_strain slip_hardening'
  []
[]
