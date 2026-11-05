# Verification against published data -- Brandyberry, Zhang & Geubelle (2022),
# Comput. Methods Appl. Mech. Engrg. 399, 115388, Fig. 11(b), p2 = 10 curve.
# Parameters: Y_in = 300, p1 = 1, p2 = 10, mu_visc = 20.
#
# Reference values are digitized from the published plot (28 points; the raw
# digitizer export is committed next to this file as
# raw_digitized_fig11b_p2_10.csv, converted by ../make_digitized_csvs.py).
#
# WHY THIS CASE EXISTS ALONGSIDE brandyberry_fig11a_Yin100
#
# That case sits at p1 = 1, p2 = 1, where the damage law degenerates:
#
#     D = 1 - exp[-( <r-Y_in> / (p1*Y_in) )^p2 ]   ->   1 - exp[-(r-Y_in)/Y_in]
#
# Both shape parameters drop out, so no error in how p1 or p2 enter can be
# detected there. Two plausible transcription errors -- coding arg^p2 as
# arg*p2, or p1*Y_in as p1+Y_in-1 -- change that curve by EXACTLY zero, while
# changing this one by 0.41 and 0.83 in damage respectively. p2 = 10 is the
# strongest discriminator in the published p2 sweep.
#
# It also independently confirms the loading time. The figures plot stress
# against strain, so the time scale is not given and had to be inferred. On
# Fig. 11(a) a one-parameter fit gave mu*T = 19.92 (T = 0.996 s). Fitting this
# curve independently gives mu*T = 20.27 (T = 1.013 s) -- 1.7% apart. More to
# the point, PREDICTING this curve with T = 1 s carried over from the other
# one, with nothing fitted here at all, lands at 0.85% of peak versus 0.81%
# when fitted. The time scale transfers, so T ~= 1 s is a property of the
# figures rather than an artifact of one fit.
#
# Agreement numbers, same convention as the Fig. 11(a) case:
#
#   27.17% of peak  rate-independent chain vs this data (would-be defect)
#    0.85% of peak  viscous chain, converged grid, T = 1 s NOT fitted here
#    1.43% of peak  what THIS test achieves (max |err| 5.05e4 Pa)
#
# The last gap is backward-Euler error at the digitizer's sampling (largest
# step dt = 0.074 s, so mu*dt ~ 1.5), not a model error.
#
# Tolerances: atol is sized by that integration artifact, rtol carries the
# large-stress points, both with ~1.3x margin. The case still discriminates
# sharply against the error it guards -- the rate-independent chain misses by
# 27% of peak.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'brandyberry_fig11b_p2_10.csv'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'brandyberry_fig11b_p2_10.csv'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'brandyberry_fig11b_p2_10.csv'
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
    atol = 6.5e4
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
    p1   = 1.0
    p2   = 10.0
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
