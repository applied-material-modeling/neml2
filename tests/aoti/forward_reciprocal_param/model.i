# Kocks-Mecking flow viscosity eta = exp(B) * mu * eps0^(k*T*A / (mu*b^3)).
# The shear modulus `mu` sits in a DENOMINATOR (inside the exponent's `c =
# k*T*A/(mu*b^3)`), so a parameter derivative d(eta)/d(mu) backprops through both
# a reciprocal/division (mu in the denominator) AND exp -- the saved-output AOTI
# lowering hazard (pytorch/pytorch#187907) the recompute-Function workaround +
# division-dunder routing resolve. Constants are non-physical but numerically
# tame so the exponent stays O(1).
[Models]
  [visc]
    type = KocksMeckingFlowViscosity
    temperature = 'T'
    A = -0.1
    B = -1.0
    shear_modulus = 2.0
    eps0 = 2.718281828459045
    k = 1.0
    b = 1.0
    viscosity = 'eta'
  []
  [model]
    type = ComposedModel
    models = 'visc'
  []
[]
