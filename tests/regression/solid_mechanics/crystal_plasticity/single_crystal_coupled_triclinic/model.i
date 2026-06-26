# neml2
# Single-crystal coupled triclinic scenario, native port of
# tests/regression/solid_mechanics/crystal_plasticity/single_crystal_coupled_triclinic/model.i.
# Dynamic batch axis = (20,) for varying end-time / orientation; uses a fully
# populated triclinic SSR4 fed through GeneralElasticity.
[Tensors]
  # end_time = LinspaceScalar(1, 10, 20) -> shape (20,)
  [end_time]
    type = Python
    expr = 'Scalar.linspace(1.0, 10.0, 20)'
  []
  # times = LinspaceScalar(0, end_time, 100) -> shape (100, 20)
  [times]
    type = Python
    expr = 'Scalar(end_time.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  # deformation_rate_single = FillSR2(0.1, -0.05, -0.05) batched (20,) -> (20, 6)
  [deformation_rate_single]
    type = Python
    expr = 'SR2.fill(0.1, -0.05, -0.05, 0.0, 0.0, 0.0).dynamic_batch.expand(20)'
  []
  # deformation_rate = LinspaceSR2(d_single, d_single, 100) -> shape (100, 20, 6)
  [deformation_rate]
    type = Python
    expr = 'SR2(deformation_rate_single.data.unsqueeze(0).expand(100, 20, 6).contiguous())'
  []
  # vorticity_single = FillWR2(w1=0.1, w2=-0.05, w3=-0.05) batched (20,) -> (20, 3)
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
    expr = 'Rot((lambda r: r / (torch.sqrt((r * r).sum(-1, keepdim=True) + 1.0) + 1.0))(torch.stack([torch.linspace(0.0, 0.75, 20, dtype=torch.float64), torch.linspace(0.0, -0.25, 20, dtype=torch.float64), torch.linspace(-0.1, 0.1, 20, dtype=torch.float64)], dim=-1)))'
  []

  # Fully populated triclinic elastic stiffness tensor (unbatched, shape (6, 6)).
  [C]
    type = Python
    expr = 'SSR4(torch.tensor([[134615.3846153846, 57692.30769230767, 57692.30769230767, 10000.0, 5000.0, 15000.0], [57692.30769230767, 134615.3846153846, 57692.30769230767, 4000.0, 20000.0, 2000.0], [57692.30769230767, 57692.30769230767, 134615.3846153846, 8000.0, 1000.0, 25000.0], [10000.0, 4000.0, 8000.0, 76923.07692307692, 2000.0, 1000.0], [5000.0, 20000.0, 1000.0, 2000.0, 76923.07692307692, 1500.0], [15000.0, 2000.0, 25000.0, 1000.0, 1500.0, 76923.07692307692]], dtype=torch.float64))'
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
    ic_Rot_names = 'orientation'
    ic_Rot_values = 'initial_orientation'
    save_as = 'result.pt'
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
    type = GeneralElasticity
    elastic_stiffness_tensor = 'C'
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
  [integrate_orientation]
    type = WR2ImplicitExponentialTimeIntegration
    variable = 'orientation'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'euler_rodrigues elasticity orientation_rate resolved_shear
              elastic_stretch plastic_deformation_rate plastic_spin
              sum_slip_rates slip_rule slip_strength voce_hardening
              integrate_slip_hardening integrate_elastic_strain integrate_orientation'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'elastic_strain slip_hardening orientation'
    residuals = 'elastic_strain_residual slip_hardening_residual orientation_residual'
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
    additional_outputs = 'elastic_strain orientation'
  []
[]
