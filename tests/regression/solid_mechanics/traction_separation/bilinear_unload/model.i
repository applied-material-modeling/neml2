# Bilinear mixed-mode TSL under a load-unload-reload schedule.
# Mixed-mode jump ramps up into softening, partially unloads, then reloads
# past the previous peak. Damage must be irreversible (non-decreasing) so
# the unloading limb returns elastically along the secant of the current
# damaged stiffness, and reloading resumes the softening branch only after
# the effective separation passes its previous maximum.
[Tensors]
  [times]
    type = LinspaceScalar
    start = 0
    end = 1
    nstep = 30
  []
  [jumps]
    type = Vec
    batch_shape = '(30)'
    values = '
      0.00333 0.00333 0.00333   0.00667 0.00667 0.00667   0.01 0.01 0.01
      0.01333 0.01333 0.01333   0.01667 0.01667 0.01667   0.02 0.02 0.02
      0.02333 0.02333 0.02333   0.02667 0.02667 0.02667   0.03 0.03 0.03
      0.03333 0.03333 0.03333   0.03667 0.03667 0.03667   0.04 0.04 0.04
      0.04333 0.04333 0.04333   0.04667 0.04667 0.04667   0.05 0.05 0.05
      0.04625 0.04625 0.04625   0.0425 0.0425 0.0425   0.03875 0.03875 0.03875
      0.035 0.035 0.035   0.03125 0.03125 0.03125   0.0275 0.0275 0.0275
      0.02375 0.02375 0.02375   0.02 0.02 0.02
      0.03 0.03 0.03   0.04 0.04 0.04   0.05 0.05 0.05
      0.06 0.06 0.06   0.07 0.07 0.07   0.08 0.08 0.08   0.09 0.09 0.09
    '
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_Vec_names = 'displacement_jump'
    force_Vec_values = 'jumps'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [model]
    type = BiLinearMixedModeTractionSeparation
    penalty_stiffness = 1000.0
    normal_fracture_energy = 1.0
    shear_fracture_energy = 1.0
    normal_strength = 10.0
    shear_strength = 10.0
    eta = 2.0
    alpha = 1e-6
  []
[]
