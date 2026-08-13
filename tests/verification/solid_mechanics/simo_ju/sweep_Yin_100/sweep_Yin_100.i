# Cross-implementation check: rate-independent Simo-Ju + Weibull.
# Case: sweep_Yin_100
# Parameters: Y_in = 100,  p1 = 1,  p2 = 1
#
# The reference CSV is produced by make_sweep_csvs.py from an independent
# numpy implementation of the same equations, so the composed NEML2 model
# must reproduce it to machine precision. This checks the NEML2 wiring --
# composition, chain rule, driver -- and pins behaviour across the sweep.
#
# It does NOT verify the constitutive law against a publication: a
# transcription error would appear identically on both sides. The case that
# can falsify the model is ../brandyberry_fig11a_Yin100/, whose reference is
# digitized from a published figure.
#
# Note this chain is rate-independent. Brandyberry's published curves include
# viscous relaxation (mu_visc = 20), which is why these directories are not
# named after their figures.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'sweep_Yin_100.csv'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'sweep_Yin_100.csv'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'sweep_Yin_100.csv'
    variable = 'stress'
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
  [verification]
    type = Verification
    driver = 'driver'
    SR2_names = 'output.sigma'
    SR2_values = 'stresses'
    rtol = 1e-6
    atol = 1e-3
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
    Y_in = 100
    p1   = 1
    p2   = 1
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
    models             = 'effective_stress strain_energy damage_history
                          weibull damage_monotone damaged_stress'
    additional_outputs = 'D r psi0 sigma_tilde'
  []
[]
