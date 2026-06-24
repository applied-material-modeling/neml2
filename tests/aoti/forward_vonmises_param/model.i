# Von Mises effective stress = sqrt((3/2) dev(S):dev(S)), with S scaled from a
# structural input by a promotable weight. Exercises the AOTI-safe `sqrt` (inside
# SR2Invariant / the Frobenius `norm`) on the parameter-derivative path: sqrt's
# reverse-mode backward saves its output -- the saved-output AOTI lowering hazard
# (pytorch/pytorch#187907) the recompute-Function workaround resolves. The
# effective stress is a magnitude (not a normalized direction), so
# d(effective_stress)/d(weight) is non-zero.
[Models]
  [scale_stress]
    type = SR2LinearCombination
    from = 'stress_in'
    to = 'stress'
    weights = '1.0'
  []
  [mandel_stress]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'mandel_stress'
    invariant = 'effective_stress'
  []
  [model]
    type = ComposedModel
    models = 'scale_stress mandel_stress vonmises'
  []
[]
