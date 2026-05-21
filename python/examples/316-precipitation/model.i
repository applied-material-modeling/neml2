# neml2
# Two-phase KWN precipitation model for 316H stainless steel,
# following the precipitation sub-model of
#   Hu et al., "Effect of microstructure evolution on the creep properties of
#   a polycrystalline 316H austenitic stainless steel",
#   Mater. Sci. Eng. A 772 (2020) 138787.
#
# Two precipitate populations compete for the same matrix solute reservoir:
#   - Cr23C6 (intragranular carbide) — rate-limited by Cr
#   - Fe2Mo  (Laves phase)           — rate-limited by Mo
#
# What's included from the paper:
#   * Multi-phase mass conservation (Cr, C, Mo across both phases)
#   * Classical nucleation theory (per phase) using the Hu-Cocks
#     ideal-solution driving force (Eq. 9, this object: IdealSolutionVolumetricDrivingForce)
#   * Zener diffusion-controlled growth (Eq. 10 first term)
#   * Coarsening — emerges naturally from the KWN population balance
#
# What's left out (deliberately):
#   * Dislocation coupling (heterogeneous nucleation factor w, pipe diffusion)
#   * Empirical coarsening-reduction factor C_f
#
# Units throughout: length = µm, time = hour, concentration = mass fraction (kg/kg),
# energy = J, temperature = K, amount = mol.

# Initial matrix mass fractions (paper Section 4.1.1: deduct initial 0.8 vol% Cr23C6)
x0_Cr = 0.1625
x0_C = 0.000375
x0_Mo = 0.0233

# Diffusivity pre-exponentials in µm²/hr (= m²/s × 1e12 × 3600)
# Paper Table 2: D_Cr = 1.5e-4 exp(-240 kJ/mol / RT) m²/s
#                D_Mo = 7.4e-4 exp(-283 kJ/mol / RT) m²/s
D0_Cr = 5.4e11
Q_Cr = 240000
D0_Mo = 2.664e12
Q_Mo = 283000

# Surface energies in J/µm² (= J/m² × 1e-12)
gamma_C = 3e-13
# 0.3 J/m² for carbide
gamma_L = 2.5e-13
# 0.25 J/m² for Laves

# Molar volumes in µm³/mol (= m³/mol × 1e18)
Vm_C = 6.0e12
# 6e-6 m³/mol for carbide
Vm_L = 2.0e13
# 2e-5 m³/mol for Laves

# Nucleation site densities in /µm³ (= /m³ × 1e-18)
N0_C = 1.0e-3
# 1e15 /m³ for carbide
N0_L = 5.0e-4
# 5e14 /m³ for Laves

# Universal constants
R_g = 8.314
k_B = 1.380649e-23
N_a = 6.02214076e23

# Temperatures (K) at which Table 2 reports c_eq, c_p
# 500, 550, 600, 650 °C. The table is padded with one point at 773.0 K
# (just below the lowest simulated temperature) because ScalarLinearInterpolation
# requires a strict lower bound, lower < x <= upper. Padding values match the
# 773.15 K entries so the pad is a constant extrapolation only.
[Tensors]
  [T_table]
    type = Scalar
    batch_shape = '(5)'
    values = '773.0 773.15 823.15 873.15 923.15'
    intermediate_dimension = 1
  []

  # Equilibrium matrix mass fractions
  [Cr_eq_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.1564 0.1564 0.1569 0.1575 0.1583'
    intermediate_dimension = 1
  []
  [C_eq_y]
    type = Scalar
    batch_shape = '(5)'
    values = '7.25e-8 7.25e-8 2.92e-7 9.48e-7 2.97e-6'
    intermediate_dimension = 1
  []
  [Mo_eq_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.0025 0.0025 0.0046 0.0076 0.0116'
    intermediate_dimension = 1
  []

  # Precipitate mass fractions (Cr, C in carbide; Mo in Laves)
  [Cr_p_C_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.6985 0.6985 0.6905 0.6832 0.6752'
    intermediate_dimension = 1
  []
  [C_p_C_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.0513 0.0513 0.0513 0.0513 0.0513'
    intermediate_dimension = 1
  []
  [Mo_p_L_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.5 0.5 0.5 0.5 0.5'
    intermediate_dimension = 1
  []

  # Pre-computed concentration differences (c_p - c_eq) for the growth-rate
  # denominator. The rate-limiting species for each phase: Cr for carbide,
  # Mo for Laves.
  [dx_C_Cr_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.5421 0.5421 0.5336 0.5257 0.5169'
    intermediate_dimension = 1
  []
  [dx_L_Mo_y]
    type = Scalar
    batch_shape = '(5)'
    values = '0.4975 0.4975 0.4954 0.4924 0.4884'
    intermediate_dimension = 1
  []

  # Initial matrix concentration scalars
  [x0_Cr]
    type = Scalar
    values = ${x0_Cr}
  []
  [x0_C]
    type = Scalar
    values = ${x0_C}
  []
  [x0_Mo]
    type = Scalar
    values = ${x0_Mo}
  []

  # Shared radius grid (both phases live in roughly the same size range,
  # 1 nm to 1 µm — paper Fig. 6 mean radius reaches ~200 nm).
  [edges]
    type = LogspaceScalar
    start = -4
    end = 0
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
  # ------------------------------------------------------------------
  # Temperature-dependent thermodynamic and kinetic parameters
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Mass balance — one CurrentConcentration per matrix solute. Each
  # passes BOTH phases' volume fractions so the denominator (1 - Σ f)
  # is correct, even when the species is only consumed by one phase.
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Per-phase projected diffusivity sum S — single rate-limiting
  # species per phase (Cr for carbide, Mo for Laves).
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Per-phase Hu-Cocks ideal-solution nucleation driving force
  # Δg = R T Σ_k ln(c_k / c_k_eq), unit weights ("product of all components")
  # Carbide: Cr × C; Laves: Mo
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Per-phase classical nucleation theory chain
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Per-phase Zener growth rate (rate-limited form)
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Per-phase finite-volume advection of n(R,t) in radius space
  # ------------------------------------------------------------------

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

  # ------------------------------------------------------------------
  # Rate of change = nucleation source + advection divergence,
  # implicit-Euler integration per phase
  # ------------------------------------------------------------------

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

  # Outer wrapper that surfaces the diagnostic outputs the notebook needs.
  [model]
    type = ComposedModel
    models = 'model_step vol_fraction_C vol_fraction_L x_Cr x_C x_Mo'
    additional_outputs = 'number_density_C number_density_L vf_C vf_L x_Cr x_C x_Mo'
  []
[]
