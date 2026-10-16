# neml2
# Simplified Sham-Needleman grain-boundary cavitation model, complete form.
#
# Three internal variables are integrated implicitly: the cavitated area
# fraction f, the cavity number density N, and the viscous sliding jump.
# The normal viscous opening is *not* integrated -- it is the algebraic
# function NV(f, N) - N0 V0, so the kinematic identity holds exactly at every
# converged step rather than only to integration order.
#
# Driven by a prescribed normal and tangential jump history.

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
    prescribed_Scalar_names = 'normal_jump tangential_jump'
    prescribed_Scalar_values = 'normal_jumps tangential_jumps'
    ic_Scalar_names = 'cavity_fraction cavity_density'
    ic_Scalar_values = 'f0 N0'
  []
[]

[Models]
  # --- Softening: the GTN-style coalescence map accelerates f once it passes
  # f_c. Only the softening functions see f*; the geometry keeps the physical f.
  [effective_fraction]
    type = EffectiveCavityFraction
    coalescence_fraction = 0.5
    failure_fraction = 0.8
    ultimate_fraction = 0.99
  []

  # --- Kinematics: the viscous normal opening is the smeared cavity volume.
  [opening]
    type = CavityVolumeOpening
    cavity_shape_factor = 0.61
    initial_cavity_fraction = 1.0e-2
    initial_cavity_density = 2.0
  []

  # --- Tractions from springs in series with the cavitation and sliding.
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

  # --- Cavity growth: Hull-Rimmer diffusive volume rate, then the geometric
  # conversion to an area-fraction rate.
  [volume_rate]
    type = DiffusiveCavityGrowthRate
    diffusivity = 1.0e-6
  []
  [growth_rate]
    type = CavityGrowthDamageRate
    cavity_shape_factor = 0.61
  []

  # --- Sliding and the local strain rate it drives.
  [sliding_rate]
    type = GrainBoundarySlidingRate
    sliding_viscosity = 4.0
  []
  [strain_rate]
    type = SlidingDrivenStrainRate
    length_scale = 0.5
  []

  # --- Nucleation and its contribution to the area fraction.
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

  # --- Implicit integration of the three internal variables.
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
    models = 'effective_fraction opening traction volume_rate growth_rate sliding_rate strain_rate nucleation_rate nucleation_damage_rate total_damage_rate integrate_f integrate_N integrate_slip'
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
