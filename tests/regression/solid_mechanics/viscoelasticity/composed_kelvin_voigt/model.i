# Kelvin-Voigt response built from primitives — shows that the named `KelvinVoigtElement` is
# equivalent to a parallel combination of a scalar-modulus spring and a Newtonian dashpot. Same
# load history and same parameters as `kelvin_voigt/model.i`, so the gold reference matches that
# scenario's gold within Newton-solve tolerance. Composition pattern (parallel = shared strain,
# stresses sum):
#   - SR2VariableRate produces strain_rate from strain
#   - SR2LinearCombination with weight = E gives the spring stress, weight = eta on strain_rate
#     gives the dashpot stress, then weights = '1 1' sums them into the total stress
# No internal state and no implicit solve are needed.
[Tensors]
  [end_time]
    type = LogspaceScalar
    start = -1
    end = 3
    nstep = 10
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = end_time
    nstep = 100
  []
  [exx]
    type = FullScalar
    batch_shape = '(10)'
    value = 0.01
  []
  [eyy]
    type = FullScalar
    batch_shape = '(10)'
    value = -0.005
  []
  [ezz]
    type = FullScalar
    batch_shape = '(10)'
    value = -0.005
  []
  [max_strain]
    type = FillSR2
    values = 'exx eyy ezz'
  []
  [strains]
    type = LinspaceSR2
    start = 0
    end = max_strain
    nstep = 100
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    force_SR2_names = 'strain'
    force_SR2_values = 'strains'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [strain_rate]
    type = SR2VariableRate
    variable = 'strain'
  []
  [spring_stress]
    type = SR2LinearCombination
    from = 'strain'
    weights = '1000'
    to = 'spring_stress'
  []
  [dashpot_stress]
    type = SR2LinearCombination
    from = 'strain_rate'
    weights = '100'
    to = 'dashpot_stress'
  []
  [stress_sum]
    type = SR2LinearCombination
    from = 'spring_stress dashpot_stress'
    weights = '1 1'
    to = 'stress'
  []
  [model]
    type = ComposedModel
    models = 'strain_rate spring_stress dashpot_stress stress_sum'
  []
[]
