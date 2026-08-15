# Regression scenario C (viscous): Simo-Ju + Weibull + viscous relaxation
# under uniaxial load / unload / reload. Exercises the viscous leaf's
# unloading branch (target < omega_prev -> omega frozen) alongside the
# loading-branch relaxation. Most sensitive test of the viscous path
# under non-monotonic strain.
#
# T_total = 1 s => mu_visc * T = 20. The unloading gate freezes omega at its
# previous value for any mu_visc, so the no-healing behavior here matches the
# rate-independent monotone cap. The loading branches do NOT: at this mu*T the
# trace is tens of percent of peak away from the rate-independent one.

[Tensors]
  # Time axis: 30 (load) + 29 (unload, first pt dropped) + 30 (reload, first dropped) = 89
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 89, dtype=torch.float64))'
  []
  # Piecewise linear eps_xx: 0 -> +3e-3 -> +0.5e-3 -> +5e-3
  [eps_xx_history]
    type = Python
    expr = '''Scalar(torch.cat([
        torch.linspace(0.0, 3.0e-3, 30, dtype=torch.float64),
        torch.linspace(3.0e-3, 0.5e-3, 30, dtype=torch.float64)[1:],
        torch.linspace(0.5e-3, 5.0e-3, 31, dtype=torch.float64)[1:],
    ]))'''
  []
  [strains]
    type = Python
    expr = '''SR2(torch.stack([
        eps_xx_history.data,
        -0.3 * eps_xx_history.data,
        -0.3 * eps_xx_history.data,
        torch.zeros_like(eps_xx_history.data),
        torch.zeros_like(eps_xx_history.data),
        torch.zeros_like(eps_xx_history.data),
    ], dim=-1))'''
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'E'
    prescribed_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [effective_stress]
    type              = LinearIsotropicElasticity
    coefficients      = '2.5e9 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain            = 'E'
    stress            = 'sigma_tilde'
  []
  [strain_energy]
    type              = LinearIsotropicStrainEnergyDensity
    coefficients      = '2.5e9 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    decomposition     = 'NONE'
    strain            = 'E'
    active_strain_energy_density   = 'psi0'
    inactive_strain_energy_density = 'psi0_unused'
  []
  [weibull_target]
    type = WeibullDamage
    r    = 'psi0'
    D    = 'D_target'
    Y_in = 300.0
    p1   = 5.0
    p2   = 2.0
  []
  [viscous_relax]
    type    = ViscousDamageRelaxation
    target  = 'D_target'
    omega   = 'D'
    time    = 't'
    mu_visc = 20.0
  []
  [damaged_stress]
    type             = DamagedStress
    damage           = 'D'
    effective_stress = 'sigma_tilde'
    stress           = 'sigma'
  []
  [model]
    type               = ComposedModel
    models             = 'effective_stress strain_energy weibull_target viscous_relax damaged_stress'
    additional_outputs = 'D D_target psi0 sigma_tilde'
  []
[]
