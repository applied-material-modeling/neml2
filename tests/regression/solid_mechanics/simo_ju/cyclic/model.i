# Regression scenario C: Simo-Ju + Weibull under uniaxial load / unload
# / reload. Verifies the IrreversibleScalar damage cap freezes D during
# unloading and picks it back up during reloading.
#
# Path: eps_xx: 0 -> +3e-3 (load past damage onset) -> +0.5e-3 (partial
# unload -- stress should reduce along the damaged secant) -> +5e-3
# (reload; D should stay flat during unload, then grow again once the
# previous peak strain is exceeded).

[Tensors]
  # Time axis: 0, 1, 2 correspond to (start, peak-1, valley), then a
  # linear reload from valley to peak-2.
  [times]
    type = Python
    # 30 (load) + 29 (unload, first point dropped) + 30 (reload, first dropped) = 89
    expr = 'Scalar(torch.linspace(0.0, 1.0, 89, dtype=torch.float64))'
  []
  # Piecewise linear eps_xx: rises to 3e-3 over t in [0, 1/3], falls to
  # 0.5e-3 over t in [1/3, 2/3], rises to 5e-3 over t in [2/3, 1].
  # eps_yy = eps_zz = -nu * eps_xx (uniaxial-stress condition).
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
  [damage_history]
    type = IrreversibleScalar
    from = 'psi0'
    to   = 'r'
  []
  [weibull]
    type = WeibullDamage
    r    = 'r'
    D    = 'D_step'
    Y_in = 300.0
    p1   = 5.0
    p2   = 2.0
  []
  [damage_monotone]
    type = IrreversibleScalar
    from = 'D_step'
    to   = 'D'
  []
  [damaged_stress]
    type             = DamagedStress
    damage           = 'D'
    effective_stress = 'sigma_tilde'
    stress           = 'sigma'
  []
  [model]
    type               = ComposedModel
    models             = 'effective_stress strain_energy damage_history weibull damage_monotone damaged_stress'
    additional_outputs = 'D r psi0 sigma_tilde'
  []
[]
