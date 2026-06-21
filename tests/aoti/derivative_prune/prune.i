# Two INDEPENDENT pipelines composed into one model, used to exercise the
# `-d` reachability prune at runtime (the off-path forward / implicit segment
# branches in csrc/aoti/ops.cpp):
#
#   * a pure-forward pipeline   strain -> stress -> mandel_stress   (forward seg)
#   * a forward+implicit pipeline   x -> y_rate -> y                (implicit seg)
#
# The two share no variables, so requesting a derivative of one pipeline's
# output prunes the *other* pipeline's whole segment:
#   -d mandel_stress:strain  keeps the forward seg, prunes the implicit seg
#   -d y:x                   keeps the implicit seg, prunes the forward seg

[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    strain = 'strain'
    stress = 'stress'
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
  []
  [mandel]
    type = IsotropicMandelStress
    cauchy_stress = 'stress'
    mandel_stress = 'mandel_stress'
  []
  [rate]
    type = MacaulaySplit
    from = 'x'
    to_positive = 'y_rate'
    to_negative = 'y_neg_unused'
  []
  [integrate]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'y'
    time = 't'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'rate integrate'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'y'
    residuals = 'y_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    abs_tol = 1e-10
    rel_tol = 1e-08
    max_its = 25
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [implicit_solve]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
  []
  [model]
    type = ComposedModel
    models = 'elasticity mandel implicit_solve'
  []
[]
