# neml2
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
    type = Python
    expr = 'Scalar(torch.logspace(-1, 3, 10, dtype=torch.float64))'
  []
  [times]
    type = Python
    expr = 'Scalar(torch.stack([torch.linspace(0, t.item(), 100, dtype=torch.float64) for t in end_time.data], dim=-1))'
  []
  [strains]
    # Mandel-fill (10, 6) max_strain = (exx, eyy, ezz, 0, 0, 0) = (0.01, -0.005, -0.005, 0, 0, 0),
    # then linspace ramp from 0 to max across 100 timesteps -> shape (100, 10, 6).
    type = Python
    expr = 'SR2(torch.tensor([0.01, -0.005, -0.005, 0.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 1, 6).expand(100, 10, 6) * torch.linspace(0.0, 1.0, 100, dtype=torch.float64).reshape(100, 1, 1))'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'
    prescribed_SR2_names = 'strain'
    prescribed_SR2_values = 'strains'
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
