# Units: length = microns, time = hours

# Initial setup: Al with 4 wt% Cu
# That gives 0.01738 mole fraction of Cu
x0_Cu = 0.01738

# Let's assume a temperature of 300 K, I can run a pycalphad simulation to predict the equilbrium Cu concentration in the matrix, precipitate, and the difference
T = 300.0
xp_Cu = 3.333291e-01
diff_FCC_Cu = 3.333270e-01

# Diffusivity ~ 0.150 exp[-30200/(RT)] cm^2/s which gives 297794 microns^2/hour
D = 297794

# The interpolated chemical potentials are given in J/mol

# The surface energy is around 0.1 J/m^2 = 0.1e-12 J/micron^2
gamma = 0.1e-12

# Molar volume of AlCu2
Vm = 21.55e12 # microns^3/mol

# Nucleation site density, let's assume 1e16 per micron^3 (too high for realistic nucleation, but it makes the test run faster)
N0 = 1e16

[Tensors]
  [edges]
    type = LinspaceScalar
    start = 0.0
    end = 1.0
    nstep = 201
    dim = 0
    group = 'intermediate'
  []
  [centers]
    type = CenterScalar
    points = 'edges'
  []
  [dx_centers]
    type = DifferenceScalar
    points = 'centers'
  []
  [dx]
    type = DifferenceScalar
    points = 'edges'
  []

  [scale_factor]
    type = Scalar
    values = 100.0
  []

  [true_centers]
    type = SemiInfiniteScalingScalar
    x = 'centers'
    s = 'scale_factor'
  []

  [center_jacobian]
    type = SemiInfiniteScalingJacobianScalar
    x = 'centers'
    s = 'scale_factor'
  []
  [center_inverse_jacobian]
    type = InverseSemiInfiniteScalingJacobianScalar
    x = 'centers'
    s = 'scale_factor'
  []

  [unscaled_ic]
    type = LinspaceScalar
    start = 1e-20
    end = 1e-20
    nstep = 200
    dim = 0
    group = 'intermediate'
  []

  [ic]
    type = ProductUserTensorScalar
    a = 'unscaled_ic'
    b = 'center_jacobian'
  []

  [time]
    type = LogspaceScalar
    start = -4
    end = 1
    nstep = 500
  []

  [x0_Cu]
    type = Scalar
    values = ${x0_Cu}
  []
  [xp_Cu]
    type = Scalar
    values = ${xp_Cu}
  []

  [X_Cu_vary]
    type = Scalar
    batch_shape = '(101)'
    values = '1e-10 1e-05 2e-05 3.0000000000000004e-05 4e-05 5e-05 6.000000000000001e-05 7.000000000000001e-05 8e-05 9e-05 0.0001 0.00011 0.00012000000000000002 0.00013000000000000002 0.00014000000000000001 0.00015000000000000001 0.00016 0.00017 0.00018 0.00019 0.0002 0.00021 0.00022 0.00023 0.00024000000000000003 0.00025 0.00026000000000000003 0.00027 0.00028000000000000003 0.00029 0.00030000000000000003 0.00031 0.00032 0.00033000000000000005 0.00034 0.00035000000000000005 0.00036 0.00037000000000000005 0.00038 0.00039000000000000005 0.0004 0.00041000000000000005 0.00042 0.00043000000000000004 0.00044 0.00045000000000000004 0.00046 0.00047000000000000004 0.00048000000000000007 0.0004900000000000001 0.0005 0.00051 0.0005200000000000001 0.0005300000000000001 0.00054 0.00055 0.0005600000000000001 0.0005700000000000001 0.00058 0.00059 0.0006000000000000001 0.0006100000000000001 0.00062 0.00063 0.00064 0.0006500000000000001 0.0006600000000000001 0.00067 0.00068 0.0006900000000000001 0.0007000000000000001 0.00071 0.00072 0.0007300000000000001 0.0007400000000000001 0.00075 0.00076 0.0007700000000000001 0.0007800000000000001 0.00079 0.0008 0.0008100000000000001 0.0008200000000000001 0.0008300000000000001 0.00084 0.0008500000000000001 0.0008600000000000001 0.0008700000000000001 0.00088 0.0008900000000000001 0.0009000000000000001 0.0009100000000000001 0.00092 0.00093 0.0009400000000000001 0.0009500000000000001 0.0009600000000000001 0.00097 0.0009800000000000002 0.00099 0.5'
    intermediate_dimension = 1
  []
  [chem_diff]
    type = Scalar
    batch_shape = '(101)'
    values = '-8296.144765949808 1275.6405840439052 1851.493398837157 2188.157551944702 2426.892895070991 2611.9687331249597 2763.1037948714666 2890.816650824378 3001.3859501538177 3098.861550337134 3186.0086656359817 3264.7996108296243 3336.6906701912367 3402.787762067668 3463.950534279424 3520.860459174736 3574.06690710042 3624.0192274577585 3671.0896461225925 3715.5899636607687 3757.7839655850107 3797.896801488294 3836.122180274753 3872.6279647321244 3907.5605744773407 3941.0484895747345 3973.2050665270513 4004.1308225497573 4033.9153042274847 4062.638628095031 4090.372759944698 4117.182584297448 4143.126804011203 4168.258701427768 4192.626785838281 4216.2753471157375 4239.244931277653 4261.572750922953 4283.293040909344 4304.437367824326 4325.034900251407 4345.112645682267 4364.695658871695 4383.807225664438 4402.469025685459 4420.701276766168 4438.522863467354 4455.951451796337 4473.003591846102 4489.694809832974 4506.03969086917 4522.051953504489 4537.744517043832 4553.129562457173 4568.218587585765 4583.022457289127 4597.551449070112 4611.815294670955 4625.823218032477 4639.5839700275255 4653.105860264134 4666.396786268032 4679.464260279453 4692.315433929402 4704.957120936679 4717.3958180881755 4729.637724573316 4741.688759903975 4753.554580482876 4765.240594962868 4776.751978525028 4788.093686101789 4799.270464718762 4810.286864961673 4821.147251665957 4829.582140453546 4829.5821404535445 4829.582140453546 4829.582140453552 4829.582140453546 4829.582140453554 4829.5821404535445 4829.5821404535445 4829.582140453542 4829.582140453552 4829.582140453546 4829.582140453559 4829.582140453556 4829.582140453561 4829.582140453539 4829.582140453539 4829.582140453549 4829.582140453546 4829.582140453546 4829.582140453552 4829.582140453546 4829.582140453556 4829.582140453549 4829.582140453546 4829.582140453546 4829.582140453546'
    intermediate_dimension = 1
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'time'
    ic_Scalar_names = 'number_density'
    ic_Scalar_values = 'ic'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'number_density'
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
  [input_temperature]
    type = ScalarConstantParameter
    value = ${T}
    parameter = 'forces/T'
  []

  [volume_fraction]
    type = PrecipitateVolumeFraction
    radius = 'true_centers'
    number_density = 'true_number_density'
    volume_fraction = 'vf'
  []
  [x_Cu]
    type = CurrentConcentration
    initial_concentration = 'x0_Cu'
    precipitate_volume_fractions = 'vf'
    precipitate_concentrations = 'xp_Cu'
    current_concentration = 'x_Cu'
  []

  [chemical_potential_difference]
    type = ScalarLinearInterpolation
    argument = 'x_Cu'
    abscissa = 'X_Cu_vary'
    ordinate = 'chem_diff'
  []

  [diffusivity_sum]
    type = ProjectedDiffusivitySum
    concentration_differences = ${diff_FCC_Cu}
    diffusivities = ${D}
    far_field_concentrations = 'x_Cu'
    projected_diffusivity_sum = 'diff_sum'
  []
  [growth_rate]
    type = SFFKGPrecipitationGrowthRate
    radius = 'true_centers'
    projected_diffusivity_sum = 'diff_sum'
    gibbs_free_energy_difference = chemical_potential_difference
    temperature = 'forces/T'
    gas_constant = 8.314
    growth_rate = 'growth_rate'
  []

  [scaled_cell_velocity]
    type = ScalarMultiplication
    from = 'growth_rate'
    scaling = 'center_inverse_jacobian'
    to = 'internal/scaled_cell_velocity'
  []
  [advection_velocity]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'internal/scaled_cell_velocity'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'v_edge'
  []
  [advective_flux]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'number_density'
    v_edge = 'v_edge'
    flux = 'J'
  []
  [left_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'J'
    bc_value = 0.0
    side = 'left'
  []
  [right_bc]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'J_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [flux_divergence]
    type = FiniteVolumeGradient
    u = 'J_with_bc_left_with_bc_right'
    dx = 'dx'
    grad_u = 'flux_div'
  []
  [rate_of_change]
    type = ScalarLinearCombination
    from = 'nucleation_flux flux_div'
    to = 'number_density_rate'
    weights = '1 1'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'number_density'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'input_temperature growth_rate scaled_cell_velocity advection_velocity advective_flux left_bc right_bc flux_divergence integrate_u unscale volume_fraction x_Cu diffusivity_sum zeldovich_factor kinetic_factor nucleation_barrier_and_critical_radius nucleation_flux_magnitude nucleation_flux rate_of_change'
  []
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'number_density'
  []
  [model_scaled]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [unscale]
    type = ScalarMultiplication
    from = 'number_density'
    scaling = 'center_inverse_jacobian'
    to = 'true_number_density'
  []


  [nucleation_barrier_and_critical_radius]
    type = NucleationBarrierandCriticalRadius
    surface_energy = ${gamma}
    total_gibbs_free_energy_difference = chemical_potential_difference
    molar_volume = ${Vm}
    nucleation_barrier = 'barrier'
    critical_radius = 'R_crit'
  []
  [zeldovich_factor]
    type = ZeldovichFactor
    critical_radius = 'R_crit'
    surface_energy = ${gamma}
    temperature = 'forces/T'
    molar_volume = ${Vm}
    avogadro_number = 6.02214076e23
    boltzmann_constant = 1.380649e-23
    zeldovich_factor = 'Z'
  []
  [kinetic_factor]
    type = KineticFactor
    critical_radius = 'R_crit'
    projected_diffusivity_sum = 'diff_sum'
    molar_volume = ${Vm}
    avogadro_number = 6.02214076e23
    kinetic_factor = 'beta'
  []
  [nucleation_flux_magnitude]
    type = NucleationFluxMagnitude
    zeldovich_factor = 'Z'
    kinetic_factor = 'beta'
    nucleation_barrier = 'barrier'
    temperature = 'forces/T'
    nucleation_site_density = ${N0}
    boltzmann_constant = 1.380649e-23
    nucleation_flux_magnitude = 'nucleation_magnitude'
  []
  [nucleation_flux]
    type = DumpInSmallestBin
    magnitude = 'nucleation_magnitude'
    cell_centers = 'true_centers'
    dumped_source = 'nucleation_flux'
  []

  [model]
    type = ComposedModel
    models = 'model_scaled unscale volume_fraction x_Cu'
    additional_outputs = 'number_density true_number_density vf x_Cu'
   []
[]