[Models]
  [eq1]
    type = SR2LinearCombination
    from_var = 'forces/E state/Ep'
    to_var = 'state/Ee'
    coefficients = '1 -1'
  []
  [eq2]
    type = LinearIsotropicElasticity
    coefficients = '1e5 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'state/Ee'
    stress = 'state/S'
  []
  [eq3]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/S'
    invariant = 'state/s'
  []
  [eq4]
    type = YieldFunction
    yield_stress = 5
    yield_function = 'state/fp'
    effective_stress = 'state/s'
  []
  [surface]
    type = ComposedModel
    models = 'eq3 eq4'
  []
  [eq5]
    type = Normality
    model = 'surface'
    function = 'state/fp'
    from = 'state/S'
    to = 'state/N'
  []
  [eq6]
    type = PerzynaPlasticFlowRate
    reference_stress = 100
    exponent = 2
    yield_function = 'state/fp'
    flow_rate = 'state/gamma_rate'
  []
  [eq7]
    type = AssociativePlasticFlow
    flow_rate = 'state/gamma_rate'
    flow_direction = 'state/N'
    plastic_strain_rate = 'state/Ep_rate'
  []
  [eq8]
    type = SR2BackwardEulerTimeIntegration
    variable = 'state/Ep'
  []
  [system]
    type = ComposedModel
    models = 'eq1 eq2 surface eq5 eq6 eq7 eq8'
  []
[]
