# Simo-Ju CDM with viscous relaxation — AOTI export/parity scenario.
#
# CLAUDE.md makes cross-route parity an invariant, and this is the Simo-Ju
# composition most likely to break it: ViscousDamageRelaxation carries derived
# (old-state) inputs `omega~1` / `t~1` and a `where` branch on the
# loading/unloading gate. Both are exactly the constructs that can diverge
# between the eager and compiled routes, so they need export coverage.
#
# Data flow:
#
#   E (strain) --+--> LinearIsotropicElasticity --------------> sigma_tilde --+
#                |                                                            |
#                +--> LinearIsotropicStrainEnergyDensity --> psi0             |
#                                                             |               |
#                                                  WeibullDamage --> D_target |
#                                                             |               |
#                            (omega~1, t~1, t) --> ViscousDamageRelaxation    |
#                                                             |               |
#                                                             D --------------+
#                                                                             |
#                                                            DamagedStress --> sigma
#
# mu_visc is deliberately small here. The generic AOTI harness drives every
# scenario with randn inputs, so `t` and `t~1` are unordered and dt can be
# negative or large; a physical mu_visc = 20 would then put (1 + mu*dt) near
# zero and blow the parity tolerance up on conditioning alone. Route parity is
# what is under test, not the constitutive response — the physics is covered by
# tests/verification/.../brandyberry_fig11a_Yin100 and the simo_ju regressions.

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
    mu_visc = 0.1
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
