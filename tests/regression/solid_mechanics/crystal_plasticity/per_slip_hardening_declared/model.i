# neml2
# per_slip_hardening, with the per-slip axis DECLARED rather than established
# by the shape of an initial condition.
#
# Same physics as the sibling `per_slip_hardening` scenario: dynamic batch axis
# = (20,) (varying end-time / orientation), 12 slip systems for the cubic
# crystal. The difference is where the 12 comes from. There, the initial
# condition is a hand-built (20, 12) tensor carrying sub_batch_ndim=1, and it
# is the only thing in the file that knows the per-slip extent -- the IC is
# doing the declaring, implicitly. Here it is a plain value and
# `[Settings]/example_batch_shape` states the extent once, for every route.
[Tensors]
  # end_time = LinspaceScalar(1, 10, 20) -> shape (20,)
  [end_time]
    type = Python
    expr = 'linspace(Scalar(1.0).dynamic_batch, Scalar(10.0).dynamic_batch, 20)'
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, 20)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  # deformation_rate single = FillSR2(0.1, -0.05, -0.05) batched (20,)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, 20, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(100, 20, 6).contiguous())'
  []
  # vorticity single = FillWR2(0.1, -0.05, -0.05) batched (20,)
  [vorticity_single]
    type = Python
    expr = 'WR2(torch.tensor([0.1, -0.05, -0.05], dtype=torch.float64).unsqueeze(0).expand(20, 3).contiguous())'
  []
  # vorticity = LinspaceWR2(w_single, w_single, 100) -> shape (100, 20, 3)
  [vorticity]
    type = Python
    expr = 'WR2(vorticity_single.data.unsqueeze(0).expand(100, 20, 3).contiguous())'
  []

  # Crystal geometry inputs: lattice parameter + slip direction + slip plane
  [a]
    type = Python
    expr = 'Scalar(1.0)'
  []
  [sdirs]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 0], dtype=torch.int64))'
  []
  [splanes]
    type = Python
    expr = 'MillerIndex(torch.tensor([1, 1, 1], dtype=torch.int64))'
  []

  # Initial orientation = FillRot(R1, R2, R3, method='standard'):
  # convert standard Rodrigues r_std to modified-Rodrigues parameters via
  # r = r_std / (sqrt(|r_std|^2 + 1) + 1). Shape (20, 3).
  [initial_orientation]
    type = Python
    expr = 'MRP((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 20, dtype=torch.float64), torch.linspace(0.0, -0.25, 20, dtype=torch.float64), torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)], dim=-1)))'
  []
  # Initial dislocation density: a value, not a shape. The per-slip axis it
  # broadcasts into is declared in [Settings] at the bottom of this file.
  [initial_dislocation_density]
    type = Python
    expr = 'Scalar(1.0e1)'
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
    ic_Scalar_names = 'dislocation_density'
    ic_Scalar_values = 'initial_dislocation_density'
    ic_MRP_names = 'orientation'
    ic_MRP_values = 'initial_orientation'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
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
    coefficients = '1e5 0.25'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    stress = 'cauchy_stress'
  []
  [resolved_shear]
    type = ResolvedShear
    stress = 'cauchy_stress'
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
  [slip_rule]
    type = PowerLawSlipRule
    n = 8.0
    gamma0 = 2.0e-1
  []
  [slip_strength]
    type = DislocationObstacleStrengthMap
    dislocation_density = 'dislocation_density'
    alpha = 0.3
    mu = 1.0e5
    b = 1.0e-4
    constant_strength = 50.0
  []
  [dislocation_density_rate]
    type = PerSlipForestDislocationEvolution
    dislocation_density = 'dislocation_density'
    k1 = 1e2
    k2 = 40.0
  []
  [integrate_dislocation_density]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'dislocation_density'
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
    models = 'euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              slip_rule slip_strength dislocation_density_rate
              integrate_dislocation_density integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [es]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain orientation dislocation_density'
    residuals = 'elastic_strain_residual orientation_residual dislocation_density_residual'
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
    unknowns_MRP = 'orientation'
    unknowns_Scalar = 'dislocation_density'
  []
  [predictor]
    type = ComposedModel
    models = 'cp_warmup_1 cp_warmup_2'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'es'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update elasticity'
    additional_outputs = 'elastic_strain'
  []
[]

[Settings]
  # The per-slip extent of the dislocation-density unknown. Read by the eager
  # driver (which has to size the values it invents for inputs no initial
  # condition supplies) and by `neml2-compile` (which has to trace them). The
  # dynamic region `2` is a nominal trace hint and is ignored off the compiled
  # route; the `; 12` is the part that matters everywhere. Declaring the
  # variable also covers its history lag `dislocation_density~1`.
  [example_batch_shape]
    dislocation_density = '(2; 12)'
  []
[]
