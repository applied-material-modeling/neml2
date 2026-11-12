# Regression scenario A (viscous): Simo-Ju + Weibull + viscous relaxation
# under monotonic uniaxial-stress tension. Nominal (E=2.5 GPa, Y_in=300,
# p1=5, p2=2, mu_visc=20). Pins the composed viscous pipeline's output
# across elastic + damage-onset + softening regimes.
#
# T_total = 1 s => mu_visc * T = 20. That is NOT the rate-independent limit:
# against the rate-independent monotonic_uniaxial regression this trace
# departs by up to 46.7% of peak stress (16.1% RMS). The viscous lag is the
# whole point of the scenario -- if a change makes this look like the
# rate-independent trace, the viscous leaf has stopped contributing.
# Recovering rate-independence needs mu_visc * T in the hundreds.

[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 50, dtype=torch.float64))'
  []
  [max_strain]
    type = Python
    expr = 'SR2(torch.tensor([5e-3, -1.5e-3, -1.5e-3, 0.0, 0.0, 0.0], dtype=torch.float64))'
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
