# Verification against published data -- Brandyberry, Zhang & Geubelle (2022),
# Comput. Methods Appl. Mech. Engrg. 399, 115388, Fig. 11(a), Y_in = 100 curve.
#
# Reference values are digitized from the published plot (25 points; the raw
# digitizer export is committed next to this file as
# raw_digitized_fig11a_Yin100.csv, converted by ../make_digitized_csvs.py).
# This is the only case in this directory whose reference was NOT produced by
# our own implementation, so it is the only one that can falsify the model.
#
# The published curves are the VISCOUS model, not the rate-independent one.
# Composing WeibullDamage -> DamagedStress without ViscousDamageRelaxation
# lands 13.29% of peak below this data throughout the softening branch. Do not
# drop viscous_relax from this composition.
#
# Three different agreement numbers appear in discussions of this case; they
# are not interchangeable:
#
#   13.29% of peak  rate-independent chain vs this data (the defect found)
#    0.28% of peak  viscous chain vs this data on a CONVERGED time grid --
#                   the physics claim, i.e. how well the model reproduces the
#                   published curve when time discretization is negligible
#    1.61% of peak  what THIS test actually achieves (max |err| 3.7e4 Pa)
#
# The gap between the last two is backward-Euler error at the digitizer's own
# sampling: the largest step is dt = 0.102 s around eps = 6e-4, where
# mu*dt ~ 2. That is a property of integrating on a 25-point grid, not of the
# model. Trimming the low-stress tail does not help -- the sparse gap is
# mid-curve, not at the end.
#
# Loading schedule: mu_visc = 20 1/s and a constant strain rate of 1e-3 1/s,
# i.e. eps = 1e-3 at t = 1 s. Fitting the single unknown time scale to the
# digitized curve gives mu*T = 19.92, which recovers mu = 20 at T = 1 s and
# confirms the reading of the figure.
#
# Tolerance budget (worst case over the 25 points, stress in Pa):
#   digitization scatter, measured on the 11 pre-damage points where the
#   response is exactly E*eps ................................ < 0.6% of peak
#   backward-Euler error on this grid (see above) ............. up to 3.7e4 Pa
# The second term dominates, so atol is sized by it and rtol carries the
# large-stress points; both leave ~1.35x margin. Note this makes atol 6.8% of
# peak stress, larger than several of the low-stress tail points themselves --
# those points are effectively carried by atol alone. What the test still
# discriminates sharply is the defect it was written for: dropping
# viscous_relax fails at max |err| = 2.166e5 Pa, 4.3x the atol.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'brandyberry_fig11a_Yin100.csv'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'brandyberry_fig11a_Yin100.csv'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'brandyberry_fig11a_Yin100.csv'
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
    rtol = 0.05
    atol = 5.0e4
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
    Y_in = 100.0
    p1   = 1.0
    p2   = 1.0
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
