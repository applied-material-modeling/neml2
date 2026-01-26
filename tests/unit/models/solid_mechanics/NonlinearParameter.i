[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_SR2_names = 'state/internal/M'
    input_SR2_values = 'M'
    input_Scalar_names = 'forces/T'
    input_Scalar_values = '550'
    output_Scalar_names = 'state/internal/fp'
    output_Scalar_values = '102.5057'
    derivative_abs_tol = 1e-06
    check_second_derivatives = true
    check_AD_parameter_derivatives = false
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '40 120 80 10 10 90'
  []
  [T_data]
    type = LinspaceScalar
    start = 273.15
    end = 2000
    nstep = 10
    dim = 0
    group = 'intermediate'
  []
  [s0_data]
    type = LinspaceScalar
    start = 50
    end = 30
    nstep = 10
    dim = 0
    group = 'intermediate'
  []
[]

[Models]
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/M'
    invariant = 'state/internal/s'
  []
  [s0]
    type = ScalarLinearInterpolation
    argument = 'forces/T'
    abscissa = 'T_data'
    ordinate = 's0_data'
  []
  [yield]
    type = YieldFunction
    yield_stress = 's0'
  []
  [model]
    type = ComposedModel
    models = 'vonmises yield'
  []
[]
