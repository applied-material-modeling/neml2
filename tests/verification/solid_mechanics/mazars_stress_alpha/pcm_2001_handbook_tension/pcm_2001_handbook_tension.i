# Verification scenario: MazarsDamageStressAlpha vs Pijaudier-Cabot/Mazars
# 2001 handbook Figure 1 -- uniaxial-stress tension branch.
#
# Drives the Mazars composed pipeline through 10 digitized tension strain
# points (loaded from modified_ds.csv and repackaged into
# pcm_2001_handbook_tension.csv by make_pcm_reference_csvs.py) and
# compares the per-step nominal stress to the digitized reference.
#
# Loading is monotonic uniaxial-stress tension (Poisson laterals ON
# strain, becoming compressive laterally) from virgin state to the
# most-positive digitized point. As in the compression sibling, strain
# shears are zero and the elastic Poisson ratio matches between the
# elasticity block (0.2) and the lateral-strain factor used in the
# generator, so off-diagonal stress components are exactly zero
# throughout.
#
# Tolerance: rtol=1e-1, atol=5e-1 MPa. Tension is digitization-noisier
# than compression (only 10 digitized points; tension RMSE measured at
# 3.6% of peak in Phase G). 10% rtol absorbs that; atol=0.5 MPa covers
# early-loading small-stress points.

[Tensors]
  [times]
    type = CSVScalar
    csv_file = 'pcm_2001_handbook_tension.csv'
    variable = 'time'
  []
  [strains]
    type = CSVSR2
    csv_file = 'pcm_2001_handbook_tension.csv'
    variable = 'strain'
  []
  [stresses]
    type = CSVSR2
    csv_file = 'pcm_2001_handbook_tension.csv'
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
    SR2_names = 'output.stress'
    SR2_values = 'stresses'
    rtol = 1e-1
    atol = 5e-1
  []
[]

[Models]
  [effective_stress]
    type              = LinearIsotropicElasticity
    coefficients      = '30000.0 0.2'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain            = 'E'
    stress            = 'effective_stress'
  []
  [eq_strain]
    type              = MazarsEquivalentStrain
    strain            = 'E'
    equivalent_strain = 'eps_tilde'
  []
  [eps_max_history]
    type = IrreversibleScalar
    from = 'eps_tilde'
    to   = 'eps_tilde_max'
  []
  [damage]
    type              = MazarsDamageStressAlpha
    equivalent_strain = 'eps_tilde_max'
    strain            = 'E'
    effective_stress  = 'effective_stress'
    damage            = 'D'
    eps_d0            = 1.0e-4
    A_t               = 1.0
    B_t               = 15000.0
    A_c               = 1.2
    B_c               = 1500.0
    E                 = 30000.0
    nu                = 0.2
  []
  [damage_monotone]
    type = IrreversibleScalar
    from = 'D'
    to   = 'D_mono'
  []
  [damaged_stress]
    type              = DamagedStress
    damage            = 'D_mono'
    effective_stress  = 'effective_stress'
    stress            = 'stress'
  []
  [model]
    type               = ComposedModel
    models             = 'effective_stress eq_strain eps_max_history damage damage_monotone damaged_stress'
    additional_outputs = 'D D_mono eps_tilde eps_tilde_max'
  []
[]
