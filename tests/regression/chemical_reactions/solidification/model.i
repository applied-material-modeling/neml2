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

    force_Scalar_names = 'T'
    force_Scalar_values = 'T'

    save_as = 'result.pt'

    ic_Scalar_names = 'phif'
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
  # dummy model: phif stays constant (rate = 0)
  [phifrate]
    type = ScalarConstantParameter
    value = 0.0
    parameter = 'phif_rate'
  []
  [phif]
    type = ScalarForwardEulerTimeIntegration
    variable = 'phif'
  []
  ## end dummy portion
  [liquid_phase_portion]
    type = HermiteSmoothStep
    argument = 'T'
    value = 'cliquid'
    lower_bound = '${Ts}'
    upper_bound = '${Tf}'
    complement = false
  []
  [solid_phase_portion]
    type = ScalarLinearCombination
    from = 'cliquid'
    to = 'omcliquid'
    weights = -1.0
    offset = 1.0
  []
  [phase_regularization]
    type = SymmetricHermiteInterpolation
    argument = 'T'
    output = 'eta'
    lower_bound = '${Ts}'
    upper_bound = '${Tf}'
  []
  [Tdot]
    type = ScalarVariableRate
    variable = 'T'
  []
  [heatrelease]
    type = ScalarMultiplication
    from = 'eta T_rate'
    to = 'q'
    scaling = '${mL}'
  []
  [liquid_phase_fluid]
    type = ScalarMultiplication
    from = 'cliquid phif'
    to = 'phif_l'
  []
  [solid_phase_fluid]
    type = ScalarMultiplication
    from = 'omcliquid phif'
    to = 'phif_s'
  []
  [model]
    type = ComposedModel
    models = 'phifrate phif liquid_phase_portion solid_phase_portion
              liquid_phase_fluid solid_phase_fluid
              phase_regularization Tdot heatrelease'
    additional_outputs = 'phif eta'
  []
[]
