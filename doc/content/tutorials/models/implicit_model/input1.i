[Models]
  [eq1]
    type = SR2LinearCombination
    from = 'strain plastic_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [eq2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
  []
  [eq3]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'stress'
    invariant = 'effective_stress'
  []
  [eq4]
    type = YieldFunction
    yield_stress = 5
  []
  [surface]
    type = ComposedModel
    models = 'eq3 eq4'
  []
  [eq5]
    type = Normality
    model = 'surface'
    function = 'yield_function'
    from = 'stress'
    to = 'flow_direction'
  []
  [eq6]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
    yield_function = 'yield_function'
  []
  [eq7]
    type = AssociativePlasticFlow
  []
  [eq8]
    type = SR2BackwardEulerTimeIntegration
    variable = 'plastic_strain'
  []
  [system]
    type = ComposedModel
    models = 'eq1 eq2 surface eq5 eq6 eq7 eq8'
  []
[]
