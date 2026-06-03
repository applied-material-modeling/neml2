# A composition of two models bound by shared variable names:
#
#   elasticity:  stress    = 3K vol(elastic_strain) + 2G dev(elastic_strain)
#   vonmises:    vm_stress = sqrt(3/2 * dev(stress) : dev(stress))
#   chain:       the two glued together by ComposedModel
#
# Substitute your custom Model from the previous tutorials in place of
# `elasticity` (or `vonmises`) — the wiring works exactly the same way.
[Models]
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '100 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    stress = 'stress'
  []
  [vonmises]
    type = SR2Invariant
    tensor = 'stress'
    invariant = 'vm_stress'
    invariant_type = VONMISES
  []
  [chain]
    type = ComposedModel
    models = 'elasticity vonmises'
    # Surface the intermediate stress so callers can see both quantities.
    additional_outputs = 'stress'
  []
[]
