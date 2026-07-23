# Full Mazars (1986) CDM in NEML2 — STRESS-BASED α weighting variant.
#
# Identical to mazars.i except the damage block uses MazarsDamageStressAlpha
# (Mazars 1986 Section 3.5.1 α formula) instead of MazarsDamage (simplified
# strain-magnitude α). All other blocks (effective stress, equivalent strain,
# irreversibility ratchets, damaged stress, top-level ComposedModel) are
# unchanged.
#
# The single new wiring is the [damage] block taking the upstream
# effective_stress output as an additional input.
#
# Material parameters: typical concrete from Mazars 1986 §5.1 + the defaults
# used in mazars.i for Phase 5 verification.
#
# Data flow:
#
#    E (strain) --+--> effective_stress  (LinearIsotropicElasticity) --+--+
#                 |                                                    |  |
#                 +--> MazarsEquivalentStrain --> eps_tilde             |  |
#                                                    |                   |  |
#                                                    +--> IrreversibleScalar --> eps_tilde_max
#                                                                                     |
#                 +-------------------------------+--+----------------+-> MazarsDamageStressAlpha --> D
#                                                 ^                                     |
#                                       NEW: needs effective_stress                     |
#                                                                                       |
#                                                                                       +-> IrreversibleScalar --> D_mono
#                                                                                                                      |
#                                                       effective_stress -----------+----+--> DamagedStress --> stress
#

[Models]

  # 1. Effective stress: sigma_tilde = C : E
  [effective_stress]
    type              = LinearIsotropicElasticity
    coefficients      = '32000 0.2'                  # E_modulus [MPa], Poisson ratio
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain            = 'E'
    stress            = 'effective_stress'
  []

  # 2. Mazars equivalent strain: eps_tilde = sqrt(sum_i <eps_i>^2)
  [eq_strain]
    type              = MazarsEquivalentStrain
    strain            = 'E'
    equivalent_strain = 'eps_tilde'
  []

  # 3. Irreversible history max: eps_tilde_max = max over time of eps_tilde
  [eps_max_history]
    type = IrreversibleScalar
    from = 'eps_tilde'
    to   = 'eps_tilde_max'
  []

  # 4. STRESS-BASED Mazars damage law (Mazars 1986 §3.5.1):
  #    D = alpha_t * D_t(eps_max) + alpha_c * D_c(eps_max)
  #    where alpha_t, alpha_c are computed from the principal decomposition
  #    of the effective stress mapped back through the isotropic compliance.
  #
  #    DUPLICATION NOTICE: E and nu below MUST match the [effective_stress]
  #    block's coefficients above. The HIT file is responsible for this
  #    consistency — there is no cross-block check. If you change E or nu
  #    in one place, change it in the other.
  [damage]
    type              = MazarsDamageStressAlpha
    equivalent_strain = 'eps_tilde_max'
    strain            = 'E'
    effective_stress  = 'effective_stress'    # NEW WIRING vs mazars.i
    damage            = 'D'
    # Damage-law parameters (same defaults as mazars.i — typical concrete)
    eps_d0            = 1.0e-4
    A_t               = 0.7
    B_t               = 1.0e4
    A_c               = 1.4
    B_c               = 1850.0
    # Elastic constants — MUST equal [effective_stress] coefficients (32000, 0.2)
    E                 = 32000.0
    nu                = 0.2
  []

  # 5. Enforce damage monotonicity (never decreases).
  #    Without this, a single call with zero current strain but previous
  #    damage history would lose the damage (α weighting hits 0/0 → default 0).
  [damage_monotone]
    type = IrreversibleScalar
    from = 'D'
    to   = 'D_mono'
  []

  # 6. Nominal stress: stress = (1 - D_mono) * effective_stress
  [damaged_stress]
    type              = DamagedStress
    damage            = 'D_mono'
    effective_stress  = 'effective_stress'
    stress            = 'stress'
  []

  # Top-level model — same children list as mazars.i
  [model]
    type               = ComposedModel
    models             = 'effective_stress eq_strain eps_max_history damage damage_monotone damaged_stress'
    additional_outputs = 'D D_mono eps_tilde eps_tilde_max'
  []

[]
