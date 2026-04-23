[Drivers]
  [unit]
    type = ModelUnitTest
    model = 'model'
    input_Scalar_names = 'k'
    input_Scalar_values = '20'
    input_SR2_names = 'M X'
    input_SR2_values = 'M X'
    output_Scalar_names = 'Nk'
    output_Scalar_values = '-0.8165'
    output_SR2_names = 'NM NX'
    output_SR2_values = 'NM NX'
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
    to = 'O'
    from = 'M X'
    weights = '1 -1'
  []
  [vonmises]
    type = SR2Invariant
    invariant_type = 'VONMISES'
    tensor = 'O'
    invariant = 's'
  []
  [yield]
    type = YieldFunction
    yield_stress = 50
    effective_stress = 's'
    isotropic_hardening = 'k'
    yield_function = 'fp'
  []
  [flow]
    type = ComposedModel
    models = 'overstress vonmises yield'
  []
  [model]
    type = Normality
    model = 'flow'
    function = 'fp'
    from = 'k X M'
    to = 'Nk NX NM'
  []
[]
