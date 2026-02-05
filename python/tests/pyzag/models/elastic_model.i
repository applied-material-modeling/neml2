[Models]
  [Erate]
    type = SR2VariableRate
    variable = 'forces/E'
    rate = 'forces/E_rate'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    rate_form = true
    strain = 'forces/E'
  []
  [integrate_stress]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/S'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'Erate elasticity integrate_stress'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
  []
[]
