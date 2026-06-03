# neml2
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

[Tensors]
  [edges]
    type = Python
    expr = 'Scalar.linspace(0.0, 1.0, 101).sub_batch.retag(1)'
  []
  [centers]
    type = Python
    expr = 'Scalar(0.5 * (edges.data[..., :-1] + edges.data[..., 1:]), sub_batch_ndim=1)'
  []
  [dx_centers]
    type = Python
    expr = 'Scalar(centers.data[..., 1:] - centers.data[..., :-1], sub_batch_ndim=1)'
  []
  [dx]
    type = Python
    expr = 'Scalar(edges.data[..., 1:] - edges.data[..., :-1], sub_batch_ndim=1)'
  []

  [scale_factor]
    type = Python
    expr = 'Scalar(5.0)'
  []

  [true_centers]
    type = Python
    expr = 'Scalar(scale_factor.data * centers.data / (1.0 - centers.data), sub_batch_ndim=1)'
  []

  [center_jacobian]
    type = Python
    expr = 'Scalar(scale_factor.data / (1.0 - centers.data) ** 2, sub_batch_ndim=1)'
  []
  [center_inverse_jacobian]
    type = Python
    expr = 'Scalar((1.0 - centers.data) ** 2 / scale_factor.data, sub_batch_ndim=1)'
  []

  [unscaled_ic]
    type = Python
    expr = 'Scalar(1e-12 * torch.ones(100, dtype=torch.float64), sub_batch_ndim=1)'
  []

  [ic]
    type = Python
    expr = 'Scalar(unscaled_ic.data * center_jacobian.data, sub_batch_ndim=1)'
  []

  [time]
    type = Python
    expr = 'Scalar.linspace(0.0, 100.0, 500)'
  []

  [x0_Cu]
    type = Python
    expr = 'Scalar(torch.tensor(${x0_Cu}, dtype=torch.float64))'
  []
  [xp_Cu]
    type = Python
    expr = 'Scalar(torch.tensor(${xp_Cu}, dtype=torch.float64))'
  []

  [X_Cu_vary]
    type = Python
    expr = 'Scalar(torch.tensor([1e-10, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [chem_diff]
    type = Python
    expr = 'Scalar(torch.tensor([-8296.144765949808, 4829.582140453559, 4829.582140453546, 4829.582140453556, 4829.582140453537, 4829.582140453549, 4829.5821404535445, 4829.5821404535445, 4829.582140453554, 4829.582140453549, 4829.582140453542, 4829.582140453542, 4829.582140453532, 4829.582140453542, 4829.582140453554, 4829.5821404535445, 4829.582140453552, 4829.582140453537, 4829.5821404535445, 4829.582140453542], dtype=torch.float64), sub_batch_ndim=1)'
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
    type = SFFKPrecipitationGrowthRate
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
    grad_u = 'number_density_rate'
  []
  [integrate_u]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'number_density'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'input_temperature growth_rate scaled_cell_velocity advection_velocity advective_flux left_bc right_bc flux_divergence integrate_u unscale volume_fraction x_Cu diffusivity_sum'
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
  [model]
    type = ComposedModel
    models = 'model_scaled unscale volume_fraction x_Cu'
    additional_outputs = 'number_density true_number_density vf x_Cu'
  []
[]
