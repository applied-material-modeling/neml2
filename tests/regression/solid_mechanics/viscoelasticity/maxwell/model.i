# Maxwell stress relaxation: a single Maxwell element (spring + dashpot in series) under a ramped
# strain. The Maxwell dashpot rate equation is composed with linear isotropic elasticity by
# decomposing the total strain into elastic and viscous parts and applying the elasticity to the
# elastic strain. Under sustained strain the stress relaxes asymptotically to zero.
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
  [elastic_strain]
    type = SR2LinearCombination
    from = 'strain viscous_strain'
    to = 'elastic_strain'
    weights = '1 -1'
  []
  [elasticity]
    type = LinearIsotropicElasticity
    coefficients = '1e4 0.3'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    strain = 'elastic_strain'
    stress = 'stress'
  []
  [maxwell]
    type = MaxwellViscoelasticity
    viscosity = 100
  []
  [integrate_Ev]
    type = SR2BackwardEulerTimeIntegration
    variable = 'viscous_strain'
  []
  [implicit_rate]
    type = ComposedModel
    models = 'elastic_strain elasticity maxwell integrate_Ev'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'implicit_rate'
    unknowns = 'viscous_strain'
    residuals = 'viscous_strain_residual'
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
    unknowns_SR2 = 'viscous_strain'
  []
  [update]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [model]
    type = ComposedModel
    models = 'update elastic_strain elasticity'
    additional_outputs = 'viscous_strain'
  []
[]
