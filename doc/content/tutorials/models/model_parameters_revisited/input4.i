# Same composition, but the CTE is now promoted to an *input variable*
# named ``alpha`` — no provider model, no literal, no tensor. NEML2 sees
# a bare name that does not match anything else in the file and adds an
# input slot for it. The caller must supply ``alpha`` at evaluation time.
[Models]
  [eq1]
    type = ThermalEigenstrain
    reference_temperature = '300'
    CTE                   = 'alpha'   # ← bare name → promoted to input variable
  []
  [eq2]
    type    = SR2LinearCombination
    from    = 'strain eigenstrain'
    to      = 'elastic_strain'
    weights = '1 -1'
  []
  [eq3]
    type              = LinearIsotropicElasticity
    strain            = 'elastic_strain'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients      = '2e5            0.3'
  []
  [eq]
    type   = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
