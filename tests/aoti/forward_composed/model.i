# forward_composed: two forward-only leaves wired in a ComposedModel.
# Tests that the segment partitioner produces a single forward segment and
# that the trace handles inter-leaf data flow correctly.

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
  [model]
    type = ComposedModel
    models = 'elasticity mandel'
  []
[]
