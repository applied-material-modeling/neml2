# Regression scenario: MazarsDamageStressAlpha — uniaxial-stress compression
# at Pijaudier-Cabot/Mazars 2001 §1.6 handbook parameters.
#
# Pins compression-side behavior: D_c activation via Poisson-induced lateral
# tension. Sister of tension_pcm; together they cover both regimes of the
# stress-based alpha implementation.
#
# Loading: monotonic ramp 0 -> -2e-3 in eps_xx, with positive Poisson laterals
# (uniaxial-stress condition: eps_yy = eps_zz = -nu*eps_xx = +0.2*|eps_xx|).
# Steps: 30.
#
# v3 port: same v2 [Tensors] -> v3 Python expression migration as tension_pcm.

[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 30, dtype=torch.float64))'
  []
  # Peak strain at t = 1 (uniaxial-stress compression with Poisson laterals)
  [max_strain]
    type = Python
    expr = 'SR2(torch.tensor([-2e-3, 4e-4, 4e-4, 0.0, 0.0, 0.0], dtype=torch.float64))'
  []
  # 30-step linear ramp from 0 to max_strain -> shape (30, 6)
  [strains]
    type = Python
    expr = 'SR2(max_strain.data.unsqueeze(0) * torch.linspace(0.0, 1.0, 30, dtype=torch.float64).unsqueeze(-1))'
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
