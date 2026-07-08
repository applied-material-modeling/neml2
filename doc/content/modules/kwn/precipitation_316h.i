# neml2
# Two-phase KWN precipitation model for 316H stainless steel.
x0_Cr = 0.1625
x0_C = 0.000375
x0_Mo = 0.0233

D0_Cr = 5.4e11
Q_Cr = 240000
D0_Mo = 2.664e12
Q_Mo = 283000

gamma_C = 3e-13
gamma_L = 2.5e-13

Vm_C = 6.0e12
Vm_L = 2.0e13

N0_C = 1.0e-3
N0_L = 5.0e-4

R_g = 8.314
k_B = 1.380649e-23
N_a = 6.02214076e23

[Tensors]
  [T_table]
    type = Python
    expr = 'Scalar(torch.tensor([773.0, 773.15, 823.15, 873.15, 923.15], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [Cr_eq_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.1564, 0.1564, 0.1569, 0.1575, 0.1583], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [C_eq_y]
    type = Python
    expr = 'Scalar(torch.tensor([7.25e-8, 7.25e-8, 2.92e-7, 9.48e-7, 2.97e-6], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [Mo_eq_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.0025, 0.0025, 0.0046, 0.0076, 0.0116], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [Cr_p_C_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.6985, 0.6985, 0.6905, 0.6832, 0.6752], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [C_p_C_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.0513, 0.0513, 0.0513, 0.0513, 0.0513], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [Mo_p_L_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [dx_C_Cr_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.5421, 0.5421, 0.5336, 0.5257, 0.5169], dtype=torch.float64), sub_batch_ndim=1)'
  []
  [dx_L_Mo_y]
    type = Python
    expr = 'Scalar(torch.tensor([0.4975, 0.4975, 0.4954, 0.4924, 0.4884], dtype=torch.float64), sub_batch_ndim=1)'
  []

  [x0_Cr]
    type = Python
    expr = 'Scalar(torch.tensor(${x0_Cr}, dtype=torch.float64))'
  []
  [x0_C]
    type = Python
    expr = 'Scalar(torch.tensor(${x0_C}, dtype=torch.float64))'
  []
  [x0_Mo]
    type = Python
    expr = 'Scalar(torch.tensor(${x0_Mo}, dtype=torch.float64))'
  []

  [edges]
    type = Python
    expr = 'logspace(Scalar(-4.0).sub_batch, Scalar(0.0).sub_batch, 201)'
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
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'number_density_C number_density_L'
  []
[]

[Solvers]
  [newton]
    type = NewtonWithLineSearch
    linear_solver = 'lu'
    rel_tol = 1e-8
    abs_tol = 1e-10
    max_its = 25
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [D_Cr]
    type = ArrheniusParameter
    temperature = 'T'
    reference_value = ${D0_Cr}
    activation_energy = ${Q_Cr}
    ideal_gas_constant = ${R_g}
  []
  [D_Mo]
    type = ArrheniusParameter
    temperature = 'T'
    reference_value = ${D0_Mo}
    activation_energy = ${Q_Mo}
    ideal_gas_constant = ${R_g}
  []

  [Cr_eq]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'Cr_eq_y'
  []
  [C_eq]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'C_eq_y'
  []
  [Mo_eq]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'Mo_eq_y'
  []
  [Cr_p_C]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'Cr_p_C_y'
  []
  [C_p_C]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'C_p_C_y'
  []
  [Mo_p_L]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'Mo_p_L_y'
  []
  [dx_C_Cr]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'dx_C_Cr_y'
  []
  [dx_L_Mo]
    type = ScalarLinearInterpolation
    argument = 'T'
    abscissa = 'T_table'
    ordinate = 'dx_L_Mo_y'
  []

  [vol_fraction_C]
    type = PrecipitateVolumeFraction
    radius = 'centers'
    number_density = 'number_density_C'
    volume_fraction = 'vf_C'
  []
  [vol_fraction_L]
    type = PrecipitateVolumeFraction
    radius = 'centers'
    number_density = 'number_density_L'
    volume_fraction = 'vf_L'
  []

  [x_Cr]
    type = CurrentConcentration
    initial_concentration = 'x0_Cr'
    precipitate_volume_fractions = 'vf_C vf_L'
    precipitate_concentrations = 'Cr_p_C 0'
    current_concentration = 'x_Cr'
  []
  [x_C]
    type = CurrentConcentration
    initial_concentration = 'x0_C'
    precipitate_volume_fractions = 'vf_C vf_L'
    precipitate_concentrations = 'C_p_C 0'
    current_concentration = 'x_C'
  []
  [x_Mo]
    type = CurrentConcentration
    initial_concentration = 'x0_Mo'
    precipitate_volume_fractions = 'vf_C vf_L'
    precipitate_concentrations = '0 Mo_p_L'
    current_concentration = 'x_Mo'
  []

  [S_C]
    type = ProjectedDiffusivitySum
    concentration_differences = 'dx_C_Cr'
    diffusivities = 'D_Cr'
    far_field_concentrations = 'x_Cr'
    projected_diffusivity_sum = 'S_C'
  []
  [S_L]
    type = ProjectedDiffusivitySum
    concentration_differences = 'dx_L_Mo'
    diffusivities = 'D_Mo'
    far_field_concentrations = 'x_Mo'
    projected_diffusivity_sum = 'S_L'
  []

  [dg_v_C]
    type = IdealSolutionVolumetricDrivingForce
    temperature = 'T'
    current_concentrations = 'x_Cr x_C'
    equilibrium_concentrations = 'Cr_eq C_eq'
    gas_constant = ${R_g}
    driving_force = 'dg_v_C'
  []
  [dg_v_L]
    type = IdealSolutionVolumetricDrivingForce
    temperature = 'T'
    current_concentrations = 'x_Mo'
    equilibrium_concentrations = 'Mo_eq'
    gas_constant = ${R_g}
    driving_force = 'dg_v_L'
  []

  [barrier_C]
    type = NucleationBarrierAndCriticalRadius
    surface_energy = ${gamma_C}
    total_gibbs_free_energy_difference = 'dg_v_C'
    molar_volume = ${Vm_C}
    nucleation_barrier = 'barrier_C'
    critical_radius = 'R_crit_C'
  []
  [barrier_L]
    type = NucleationBarrierAndCriticalRadius
    surface_energy = ${gamma_L}
    total_gibbs_free_energy_difference = 'dg_v_L'
    molar_volume = ${Vm_L}
    nucleation_barrier = 'barrier_L'
    critical_radius = 'R_crit_L'
  []

  [Z_C]
    type = ZeldovichFactor
    critical_radius = 'R_crit_C'
    surface_energy = ${gamma_C}
    temperature = 'T'
    molar_volume = ${Vm_C}
    avogadro_number = ${N_a}
    boltzmann_constant = ${k_B}
    zeldovich_factor = 'Z_C'
  []
  [Z_L]
    type = ZeldovichFactor
    critical_radius = 'R_crit_L'
    surface_energy = ${gamma_L}
    temperature = 'T'
    molar_volume = ${Vm_L}
    avogadro_number = ${N_a}
    boltzmann_constant = ${k_B}
    zeldovich_factor = 'Z_L'
  []

  [beta_C]
    type = KineticFactor
    critical_radius = 'R_crit_C'
    projected_diffusivity_sum = 'S_C'
    molar_volume = ${Vm_C}
    avogadro_number = ${N_a}
    kinetic_factor = 'beta_C'
  []
  [beta_L]
    type = KineticFactor
    critical_radius = 'R_crit_L'
    projected_diffusivity_sum = 'S_L'
    molar_volume = ${Vm_L}
    avogadro_number = ${N_a}
    kinetic_factor = 'beta_L'
  []

  [J_C]
    type = NucleationFluxMagnitude
    zeldovich_factor = 'Z_C'
    kinetic_factor = 'beta_C'
    nucleation_barrier = 'barrier_C'
    temperature = 'T'
    nucleation_site_density = ${N0_C}
    boltzmann_constant = ${k_B}
    nucleation_flux_magnitude = 'J_C'
  []
  [J_L]
    type = NucleationFluxMagnitude
    zeldovich_factor = 'Z_L'
    kinetic_factor = 'beta_L'
    nucleation_barrier = 'barrier_L'
    temperature = 'T'
    nucleation_site_density = ${N0_L}
    boltzmann_constant = ${k_B}
    nucleation_flux_magnitude = 'J_L'
  []

  [nuc_flux_C]
    type = DumpInSmallestBin
    magnitude = 'J_C'
    cell_centers = 'centers'
    dumped_source = 'nuc_flux_C'
  []
  [nuc_flux_L]
    type = DumpInSmallestBin
    magnitude = 'J_L'
    cell_centers = 'centers'
    dumped_source = 'nuc_flux_L'
  []

  [growth_C]
    type = RateLimitedPrecipitateGrowthRate
    radius = 'centers'
    current_concentration = 'x_Cr'
    equilibrium_concentration = 'Cr_eq'
    concentration_difference = 'dx_C_Cr'
    diffusivity = 'D_Cr'
    growth_rate = 'growth_C'
  []
  [growth_L]
    type = RateLimitedPrecipitateGrowthRate
    radius = 'centers'
    current_concentration = 'x_Mo'
    equilibrium_concentration = 'Mo_eq'
    concentration_difference = 'dx_L_Mo'
    diffusivity = 'D_Mo'
    growth_rate = 'growth_L'
  []

  [v_edge_C]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'growth_C'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'v_edge_C'
  []
  [v_edge_L]
    type = LinearlyInterpolateToCellEdges
    cell_values = 'growth_L'
    cell_centers = 'centers'
    cell_edges = 'edges'
    edge_values = 'v_edge_L'
  []

  [flux_C]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'number_density_C'
    v_edge = 'v_edge_C'
    flux = 'flux_C'
  []
  [flux_L]
    type = FiniteVolumeUpwindedAdvectiveFlux
    u = 'number_density_L'
    v_edge = 'v_edge_L'
    flux = 'flux_L'
  []

  [flux_C_left]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'flux_C'
    bc_value = 0.0
    side = 'left'
  []
  [flux_C_right]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'flux_C_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []
  [flux_L_left]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'flux_L'
    bc_value = 0.0
    side = 'left'
  []
  [flux_L_right]
    type = FiniteVolumeAppendBoundaryCondition
    input = 'flux_L_with_bc_left'
    bc_value = 0.0
    side = 'right'
  []

  [flux_div_C]
    type = FiniteVolumeGradient
    u = 'flux_C_with_bc_left_with_bc_right'
    dx = 'dx'
    grad_u = 'flux_div_C'
  []
  [flux_div_L]
    type = FiniteVolumeGradient
    u = 'flux_L_with_bc_left_with_bc_right'
    dx = 'dx'
    grad_u = 'flux_div_L'
  []

  [n_dot_C]
    type = ScalarLinearCombination
    from = 'nuc_flux_C flux_div_C'
    to = 'number_density_C_rate'
    weights = '1 1'
  []
  [n_dot_L]
    type = ScalarLinearCombination
    from = 'nuc_flux_L flux_div_L'
    to = 'number_density_L_rate'
    weights = '1 1'
  []

  [integrate_C]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'number_density_C'
  []
  [integrate_L]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'number_density_L'
  []

  [implicit_rate]
    type = ComposedModel
    models = 'D_Cr D_Mo Cr_eq C_eq Mo_eq Cr_p_C C_p_C Mo_p_L dx_C_Cr dx_L_Mo
              vol_fraction_C vol_fraction_L x_Cr x_C x_Mo
              S_C S_L dg_v_C dg_v_L
              barrier_C barrier_L Z_C Z_L beta_C beta_L J_C J_L
              nuc_flux_C nuc_flux_L
              growth_C growth_L
              v_edge_C v_edge_L flux_C flux_L
              flux_C_left flux_C_right flux_L_left flux_L_right
              flux_div_C flux_div_L
              n_dot_C n_dot_L integrate_C integrate_L'
  []

  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_Scalar = 'number_density_C number_density_L'
  []

  [model_step]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []

  [model]
    type = ComposedModel
    models = 'model_step vol_fraction_C vol_fraction_L x_Cr x_C x_Mo'
    additional_outputs = 'number_density_C number_density_L vf_C vf_L x_Cr x_C x_Mo'
  []
[]
