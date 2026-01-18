nbatch = '(1)'
nstep = '100'

mL = -14832
Ts = 1600
Tf = 1700

[Tensors]
  [endtime]
    type = Scalar
    values = '${nstep}'
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
    start = '1800'
    end = '1400'
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

    save_as = 'result.pt'

    ic_Scalar_names = 'state/phif'
    ic_Scalar_values = '1.0'
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
  # this portion is to make sure the regression test runs by adding a dummy model
  [phifrate]
    type = ScalarParameterToState
    from = 0.0
    to = 'state/phif_rate'
  []
  [phif]
    type = ScalarForwardEulerTimeIntegration
    variable = 'state/phif'
  []
  ## end dummy portion
  [liquid_phase_portion]
    type = HermiteSmoothStep
    argument = 'forces/T'
    value = 'state/cliquid'
    lower_bound = '${Ts}'
    upper_bound = '${Tf}'
    complement = false
  []
  [solid_phase_portion]
    type = ScalarLinearCombination
    from_var = 'state/cliquid'
    to_var = 'state/omcliquid'
    coefficients = -1.0
    constant_coefficient = 1.0
  []
  [phase_regularization]
    type = SymmetricHermiteInterpolation
    argument = 'forces/T'
    value = 'state/eta'
    lower_bound = '${Ts}'
    upper_bound = '${Tf}'
  []
  [Tdot]
    type = ScalarVariableRate
    variable = 'forces/T'
    rate = 'state/Tdot'
    time = 'forces/t'
  []
  [heatrelease]
    type = ScalarMultiplication
    from_var = 'state/eta state/Tdot'
    to_var = 'state/q'
    coefficient = '${mL}'
  []
  [liquid_phase_fluid]
    type = ScalarMultiplication
    from_var = 'state/cliquid state/phif'
    to_var = 'state/phif_l'
  []
  [solid_phase_fluid]
    type = ScalarMultiplication
    from_var = 'state/omcliquid state/phif'
    to_var = 'state/phif_s'
  []
  [model]
    type = ComposedModel
    models = 'phifrate phif liquid_phase_portion solid_phase_portion
              liquid_phase_fluid solid_phase_fluid
              phase_regularization Tdot heatrelease'
    additional_outputs = 'state/phif state/eta'
  []
[]
