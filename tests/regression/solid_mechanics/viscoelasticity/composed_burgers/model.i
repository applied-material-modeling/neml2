# Burgers response built from primitives — Maxwell branch in series with a Kelvin-Voigt block,
# both internal-state dashpots solved together. Same load history and parameters as
# `burgers/model.i`. The trick for any series chain: a single shared stress flows through every
# element, so the Maxwell spring's `stress` output is consumed both by the Maxwell dashpot AND
# (after subtracting the Kelvin spring's stress) by the Kelvin dashpot. Two viscous strains, two
# residuals, one Newton solve.
#
# Math, cross-checked against the closed-form BurgersElement:
#   stress = E_M * (strain - mvs - kvs)
#   maxwell_viscous_strain_rate = stress / eta_M
#   kelvin_dashpot_stress = stress - E_K * kvs
#   kelvin_voigt_strain_rate = kelvin_dashpot_stress / eta_K
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
  # Maxwell-branch elastic strain: strain absorbed by the series chain that isn't already in the
  # Maxwell dashpot or in the Kelvin-Voigt block.
  [elastic_strain]
    type = SR2LinearCombination
    from = 'strain maxwell_viscous_strain kelvin_voigt_strain'
    weights = '1 -1 -1'
    to = 'elastic_strain'
  []
  [maxwell_spring]
    type = SR2LinearCombination
    from = 'elastic_strain'
    weights = '5000'
    to = 'stress'
  []
  [maxwell_dashpot]
    type = LinearDashpot
    stress = 'stress'
    viscous_strain_rate = 'maxwell_viscous_strain_rate'
    viscosity = 200
  []

  # Kelvin-Voigt block (parallel of spring and dashpot, both under stress = chain stress)
  [kelvin_spring_stress]
    type = SR2LinearCombination
    from = 'kelvin_voigt_strain'
    weights = '2000'
    to = 'kelvin_spring_stress'
  []
  [kelvin_dashpot_stress]
    type = SR2LinearCombination
    from = 'stress kelvin_spring_stress'
    weights = '1 -1'
    to = 'kelvin_dashpot_stress'
  []
  [kelvin_dashpot]
    type = LinearDashpot
    stress = 'kelvin_dashpot_stress'
    viscous_strain_rate = 'kelvin_voigt_strain_rate'
    viscosity = 100
  []

  [integrate_EvM]
    type = SR2BackwardEulerTimeIntegration
    variable = 'maxwell_viscous_strain'
  []
  [integrate_EK]
    type = SR2BackwardEulerTimeIntegration
    variable = 'kelvin_voigt_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'elastic_strain maxwell_spring maxwell_dashpot
              kelvin_spring_stress kelvin_dashpot_stress kelvin_dashpot
              integrate_EvM integrate_EK'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'maxwell_viscous_strain kelvin_voigt_strain'
    residuals = 'maxwell_viscous_strain_residual kelvin_voigt_strain_residual'
  []
[]

[Solvers]
  [newton]
    type = Newton
    linear_solver = 'lu'
  []
  [lu]
    type = DenseLU
  []
[]

[Models]
  [predictor]
    type = ConstantExtrapolationPredictor
    unknowns_SR2 = 'maxwell_viscous_strain kelvin_voigt_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update elastic_strain maxwell_spring maxwell_dashpot
              kelvin_spring_stress kelvin_dashpot_stress kelvin_dashpot'
    additional_outputs = 'maxwell_viscous_strain kelvin_voigt_strain stress'
  []
[]
