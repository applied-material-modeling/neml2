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
    time = 'forces/t'

    force_Scalar_names = 'forces/T'
    force_Scalar_values = 'T'

    ic_Scalar_names = 'state/wb state/wc'
    ic_Scalar_values = '${wb0} ${wc0}'
    save_as = 'result.pt'
  []
  [regression]
    type = TransientRegression
    driver = 'driver'
    reference = 'gold/result.pt'
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
  [reaction_coef]
    type = ArrheniusParameter
    reference_value = '${k0}'
    activation_energy = '${Q}'
    ideal_gas_constant = '${R}'
    temperature = 'forces/T'
    parameter = 'state/k'
  []
  [reaction_rate]
    type = ContractingGeometry
    reaction_coef = 'reaction_coef'
    reaction_order = '${n}'
    conversion_degree = 'state/alpha'
    reaction_rate = 'state/alpha_rate'
  []
  [reaction_ode]
    type = ScalarBackwardEulerTimeIntegration
    variable = 'state/alpha'
  []
  [reaction]
    type = ComposedModel
    models = 'reaction_rate reaction_ode'
  []
  [solve_reaction]
    type = ImplicitUpdate
    implicit_model = 'reaction'
    solver = 'newton'
  []
  [binder_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '-1'
    to_var = 'state/wb_rate'
  []
  [char_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '${Y}'
    to_var = 'state/wc_rate'
  []
  [gas_rate]
    type = ScalarLinearCombination
    from_var = 'state/wb_rate state/wc_rate'
    coefficients = '-${mu} -${mu}'
    to_var = 'state/wg_rate'
  []
  [open_pore_rate]
    type = ScalarLinearCombination
    from_var = 'state/alpha_rate'
    coefficients = '${zeta}'
    to_var = 'state/phio_rate'
  []
  [binder]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wb'
  []
  [char]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wc'
  []
  [gas]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/wg'
  []
  [open_pore]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/phio'
  []
  [model]
    type = ComposedModel
    models = "solve_reaction reaction_rate
              binder_rate char_rate gas_rate open_pore_rate
              binder char gas open_pore"
    additional_outputs = 'state/alpha'
  []
[]
