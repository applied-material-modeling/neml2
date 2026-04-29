nbatch = '(1)'
nstep = 100

# reaction mechanism
Y = 0.5835777126099713 # yield
n = 1.0 # reaction order
k0 = 0.04210147513030456 # reaction rate coefficient
Q = 21191.61425138572 # J/mol
R = 8.31446261815324 # J/K/mol

# initial mass fraction
wc0 = 0 # char (residue)
wb0 = 1 # binder (precursor)

# control volume
mu = 0.2
zeta = 0.05

[Tensors]
  [yield]
    type = Scalar
    values = '${Y}'
  []
  [endtime]
    type = Scalar
    values = '2700'
    batch_shape = '${nbatch}'
  []
  [times]
    type = LinspaceScalar
    start = 0
    end = endtime
    nstep = '${nstep}'
  []
  [T]
    type = LinspaceScalar
    start = '300'
    end = '1500'
    nstep = '${nstep}'
  []
[]

[Drivers]
  [driver]
    type = TransientDriver
    model = 'model'
    prescribed_time = 'times'

    force_Scalar_names = 'T'
    force_Scalar_values = 'T'

    ic_Scalar_names = 'wb wc'
    ic_Scalar_values = '${wb0} ${wc0}'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
  []
[]

[Models]
  [reaction_coef]
    type = ArrheniusParameter
    reference_value = '${k0}'
    activation_energy = '${Q}'
    ideal_gas_constant = '${R}'
    temperature = 'T'
  []
  [reaction_rate]
    type = ContractingGeometry
    coef = 'reaction_coef'
    order = '${n}'
    conversion_degree = 'alpha'
    reaction_rate = 'alpha_rate'
  []
  [reaction_ode]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'alpha'
  []
  [reaction]
    type = ComposedModel
    models = 'reaction_coef reaction_rate reaction_ode'
  []
[]

[EquationSystems]
  [eq_sys]
    type = NonlinearSystem
    model = 'reaction'
    unknowns = 'alpha'
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
    unknowns_Scalar = 'alpha'
  []
  [solve_reaction]
    type = ImplicitUpdate
    equation_system = 'eq_sys'
    solver = 'newton'
    predictor = 'predictor'
  []
  [binder_rate]
    type = ScalarLinearCombination
    from = 'alpha_rate'
    weights = '-1'
    to = 'wb_rate'
  []
  [char_rate]
    type = ScalarLinearCombination
    from = 'alpha_rate'
    weights = '${Y}'
    to = 'wc_rate'
  []
  [gas_rate]
    type = ScalarLinearCombination
    from = 'wb_rate wc_rate'
    weights = '-0.2 -0.2'
    to = 'wg_rate'
  []
  [open_pore_rate]
    type = ScalarLinearCombination
    from = 'alpha_rate'
    weights = '${zeta}'
    to = 'phio_rate'
  []
  [binder]
    type = ScalarForwardEulerTimeIntegration
    variable = 'wb'
  []
  [char]
    type = ScalarForwardEulerTimeIntegration
    variable = 'wc'
  []
  [gas]
    type = ScalarForwardEulerTimeIntegration
    variable = 'wg'
  []
  [open_pore]
    type = ScalarForwardEulerTimeIntegration
    variable = 'phio'
  []
  [model]
    type = ComposedModel
    models = "solve_reaction reaction_rate
              binder_rate char_rate gas_rate open_pore_rate
              binder char gas open_pore"
    additional_outputs = 'alpha'
  []
[]
