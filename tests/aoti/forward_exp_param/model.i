# Kocks-Mecking yield stress tau = mu * exp(C), combined with a structural input
# x so the parameter-Jacobian carrier has a batch source. Exercises the AOTI-safe
# `exp` wrapper: d(out)/d(C) = d(tau)/d(C) = mu*exp(C) backprops through exp, whose
# reverse-mode backward saves its output -- the saved-output AOTI lowering hazard
# (pytorch/pytorch#187907) the recompute-Function workaround resolves.
[Models]
  [ys]
    type = KocksMeckingYieldStress
    C = 0.1
    shear_modulus = 50000.0
    yield_stress = 'tau'
  []
  [combine]
    type = ScalarLinearCombination
    from = 'tau x'
    to = 'out'
    weights = '1.0 1.0'
  []
  [model]
    type = ComposedModel
    models = 'ys combine'
  []
[]
