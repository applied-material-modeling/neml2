[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_scalar_names = 'state/internal/k'
    input_scalar_values = '20'
    input_symr2_names = 'state/internal/M state/internal/X'
    input_symr2_values = 'M X'
    output_scalar_names = 'state/internal/Nk'
    output_scalar_values = '-0.8165'
    output_symr2_names = 'state/internal/NM state/internal/NX'
    output_symr2_values = 'NM NX'
    value_abs_tol = 1e-4
  []
[]

[Tensors]
  [M]
    type = FillSR2
    values = '100 110 100 50 40 30'
  []
  [X]
    type = FillSR2
    values = '60 -10 20 40 30 -60'
  []
  [NM]
    type = FillSR2
    values = '-0.2843 0.2843 0 0.071064 0.071064 0.639578'
  []
  [NX]
    type = FillSR2
    values = '0.2843 -0.2843 0 -0.071064 -0.071064 -0.639578'
  []
[]

[Models]
  [overstress]
    type = SR2LinearCombination
    to_var = 'state/internal/O'
    from_var = 'state/internal/M state/internal/X'
    coefficients = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'state/internal/O'
    invariant = 'state/internal/s'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    isotropic_hardening = 'state/internal/k'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [model]
    type = Normality
    model = 'flow'
    function = 'state/internal/fp'
    from = 'state/internal/k state/internal/X state/internal/M'
    to = 'state/internal/Nk state/internal/NX state/internal/NM'
  []
[]
