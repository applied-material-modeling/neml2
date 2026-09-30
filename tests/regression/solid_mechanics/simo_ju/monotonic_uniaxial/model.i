# Regression scenario A: Simo-Ju + Weibull under monotonic uniaxial-stress
# tension. Nominal parameter set (E=2.5 GPa, Y_in=300, p1=5, p2=2). Pins
# the composed model's output across elastic + damage-onset + softening
# regimes. Detects any regression in WeibullDamage, the strain-energy leaf,
# or the composed-model dependency resolution.

[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 50, dtype=torch.float64))'
  []
  # Uniaxial-stress state: eps_yy = eps_zz = -nu*eps_xx (nu = 0.3).
  # Peak: eps_xx = 5e-3 (5 permille, well past damage saturation).
  [max_strain]
    type = Python
    expr = 'SR2(torch.tensor([5e-3, -1.5e-3, -1.5e-3, 0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  # 50-step linear ramp from 0 -> max_strain
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 50, dtype=torch.float64).unsqueeze(-1))'
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
