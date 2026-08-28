# Regression scenario: MazarsDamageStressAlpha — uniaxial-stress tension
# at Pijaudier-Cabot/Mazars 2001 §1.6 handbook parameters.
#
# Pins the trusted output across the elastic, peak, and softening regimes.
# A future code change that breaks the stress-based alpha implementation
# will trip this test on the next regression run.
#
# Loading: monotonic ramp 0 -> +5e-4 in eps_xx, with Poisson laterals
# (uniaxial-stress condition: eps_yy = eps_zz = -nu*eps_xx).
# Steps: 30 (sufficient to span elastic + softening regions).
#
# v3 port: the v2 [Tensors] block used LinspaceScalar/FullScalar/FillSR2/
# LinspaceSR2 — none of which exist in v3. Replaced with `type = Python`
# expressions that build the same tensors via torch primitives. Loading
# history is mathematically identical to the v2 scenario.

[Tensors]
  [times]
    type = Python
    expr = 'Scalar(torch.linspace(0.0, 1.0, 30, dtype=torch.float64))'
  []
  # Peak strain at t = 1 (uniaxial-stress tension with Poisson laterals)
  [max_strain]
    type = Python
    expr = 'SR2(torch.tensor([5e-4, -1e-4, -1e-4, 0.0, 0.0, 0.0], dtype=torch.float64))'
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
