# neml2
# Sham-Needleman cavitation without nucleation: the cavity population is
# fixed at N_0 and damage accumulates only by diffusive growth of the
# cavities that are already there.
#
# This variant is assembled by *omitting* leaves, not by setting an option:
# CavityNucleationRate, CavityNucleationDamageRate and the
# ScalarLinearCombination that summed the two contributions are simply absent,
# and CavityGrowthDamageRate renames its output straight onto
# cavity_fraction_rate. The cavity density is a prescribed constant force
# instead of an unknown, so the implicit system drops from three equations to
# two.

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
  [densities]
    type = Python
    expr = 'Scalar(torch.full((100, 1), 2.0, dtype=torch.float64))'
  []
  [f0]
    type = Python
    expr = 'Scalar(torch.tensor(1.0e-2, dtype=torch.float64))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_Scalar_names = 'normal_jump tangential_jump cavity_density'
    prescribed_Scalar_values = 'normal_jumps tangential_jumps densities'
    ic_Scalar_names = 'cavity_fraction'
    ic_Scalar_values = 'f0'
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
    cavity_fraction_growth_rate = 'cavity_fraction_rate'
  []
  [sliding_rate]
    type = GrainBoundarySlidingRate
    sliding_viscosity = 4.0
  []

  [integrate_f]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'cavity_fraction'
  []
  [integrate_slip]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'viscous_sliding'
  []

  [implicit_rate]
    type = ComposedModel
    models = 'effective_fraction opening traction volume_rate growth_rate sliding_rate integrate_f integrate_slip'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'cavity_fraction viscous_sliding'
    residuals = 'cavity_fraction_residual viscous_sliding_residual'
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
    unknowns_Scalar = 'cavity_fraction viscous_sliding'
  []
  [model]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
[]
