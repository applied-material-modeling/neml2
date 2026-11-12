# Regression scenario B: Simo-Ju + Weibull under equibiaxial in-plane tension.
# Loading: eps_xx = eps_yy = t * 5e-3, eps_zz free (plane stress not
# enforced -- we prescribe only in-plane components and let the third
# principal come from the eps_zz value chosen below).
#
# This exercises the isotropic energy norm psi_0 = 0.5*e:C:e in a
# non-uniaxial state. A bug where the model implicitly assumes uniaxial
# strain (e.g., a hardcoded axial component) would trip this test.

[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 50, dtype=torch.float64))'
  []
  # Equibiaxial in-plane tension with zero out-of-plane strain
  # (mirrors a "plane strain, biaxial" state -- distinct from uniaxial).
  [max_strain]
    type = Python
    expr = 'SR2(torch.tensor([5e-3, 5e-3, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64))'
  []
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
