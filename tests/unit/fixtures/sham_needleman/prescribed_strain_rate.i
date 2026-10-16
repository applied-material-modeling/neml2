# neml2
# Sham-Needleman cavitation with the nucleation-driving strain rate supplied
# from outside instead of computed from the local grain-boundary sliding.
#
# Physically this is the case where the surrounding grains creep and that bulk
# strain rate, not the boundary sliding, sets how fast new cavities appear.
# The only structural change from complete.i is that SlidingDrivenStrainRate
# is dropped and strain_rate becomes a prescribed force; GrainBoundarySliding-
# Rate stays, because it still carries the shear response.

[Tensors]
  [times]
    type = Python
    expr = 'linspace(Scalar(0.0).dynamic_batch, Scalar(200.0).dynamic_batch, 100)'
  []
  [normal_jumps]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 0.08, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [tangential_jumps]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 0.04, 100, dtype=torch.float64).unsqueeze(-1))'
  []
  [strain_rates]
    type = Python
    expr = 'Scalar(torch.full((100, 1), 1.0e-2, dtype=torch.float64))'
  []
  [f0]
    type = Python
    expr = 'Scalar(torch.tensor(1.0e-2, dtype=torch.float64))'
  []
  [N0]
    type = Python
    expr = 'Scalar(torch.tensor(2.0, dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Scalar_names = 'normal_jump tangential_jump strain_rate'
    prescribed_Scalar_values = 'normal_jumps tangential_jumps strain_rates'
    ic_Scalar_names = 'cavity_fraction cavity_density'
    ic_Scalar_values = 'f0 N0'
  []
[]

[Models]
  [effective_fraction]
    type = EffectiveCavityFraction
    coalescence_fraction = 0.5
    failure_fraction = 0.8
    ultimate_fraction = 0.99
  []
  [opening]
    type = CavityVolumeOpening
    cavity_shape_factor = 0.61
    initial_cavity_fraction = 1.0e-2
    initial_cavity_density = 2.0
  []
  [traction]
    type = ShamNeedlemanTraction
    # Separation is detected from the converged state at the start of the
    # step. Switching on the trial unknown instead puts a jump in the
    # residual at f_F and leaves Newton with no root to find.
    cavity_fraction = 'cavity_fraction~1'
    normal_stiffness = 1000.0
    shear_stiffness = 800.0
    penalty_stiffness = 2000.0
    failure_fraction = 0.8
  []
  [volume_rate]
    type = DiffusiveCavityGrowthRate
    diffusivity = 1.0e-6
  []
  [growth_rate]
    type = CavityGrowthDamageRate
    cavity_shape_factor = 0.61
  []
  [sliding_rate]
    type = GrainBoundarySlidingRate
    sliding_viscosity = 4.0
  []
  [nucleation_rate]
    type = CavityNucleationRate
    nucleation_coefficient = 3.0
    reference_stress = 50.0
    exponent = 2.0
  []
  [nucleation_damage_rate]
    type = CavityNucleationDamageRate
  []
  [total_damage_rate]
    type = ScalarLinearCombination
    from = 'cavity_fraction_growth_rate cavity_fraction_nucleation_rate'
    to = 'cavity_fraction_rate'
    weights = '1 1'
  []

  [integrate_f]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'cavity_fraction'
  []
  [integrate_N]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'cavity_density'
  []
  [integrate_slip]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'viscous_sliding'
  []

  [implicit_rate]
    type = ComposedModel
    models = 'effective_fraction opening traction volume_rate growth_rate sliding_rate nucleation_rate nucleation_damage_rate total_damage_rate integrate_f integrate_N integrate_slip'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'cavity_fraction cavity_density viscous_sliding'
    residuals = 'cavity_fraction_residual cavity_density_residual viscous_sliding_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'cavity_fraction cavity_density viscous_sliding'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
