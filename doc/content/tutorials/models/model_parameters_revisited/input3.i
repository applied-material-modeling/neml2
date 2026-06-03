# Same composition again, but now alpha(T) and E(T) are temperature-
# dependent. We declare two ScalarLinearInterpolation sub-models and
# point eq1/eq3 at them by name. The interpolations become children of
# the ComposedModel, and the original "scalar" parameters eq1.alpha and
# eq3.E disappear — they are replaced by the abscissa/ordinate
# parameters of the interpolants.
[Tensors]
  [alpha_x]
    type = Python
    expr = 'Scalar(torch.tensor([300., 400., 500.], dtype=torch.float64)).with_sub_batch(1)'
  []
  [alpha_y]
    type = Python
    expr = 'Scalar(torch.tensor([1e-5, 1.5e-5, 1.8e-5], dtype=torch.float64)).with_sub_batch(1)'
  []
  [E_x]
    type = Python
    expr = 'Scalar(torch.tensor([300., 350., 400., 450.], dtype=torch.float64)).with_sub_batch(1)'
  []
  [E_y]
    type = Python
    expr = 'Scalar(torch.tensor([2.0e5, 1.9e5, 1.8e5, 1.7e5], dtype=torch.float64)).with_sub_batch(1)'
  []
[]

[Models]
  [alpha]
    type     = ScalarLinearInterpolation
    argument = 'temperature'
    abscissa = 'alpha_x'
    ordinate = 'alpha_y'
  []
  [E]
    type     = ScalarLinearInterpolation
    argument = 'temperature'
    abscissa = 'E_x'
    ordinate = 'E_y'
  []
  [eq1]
    type                  = ThermalEigenstrain
    reference_temperature = '300'
    CTE                   = 'alpha'   # ← name of the [Models/alpha] sub-model
  []
  [eq2]
    type    = SR2LinearCombination
    from    = 'strain eigenstrain'
    to      = 'elastic_strain'
    weights = '1 -1'
  []
  [eq3]
    type              = LinearIsotropicElasticity
    strain            = 'elastic_strain'
    coefficient_types = 'YOUNGS_MODULUS POISSONS_RATIO'
    coefficients      = 'E              0.3'           # ← E references [Models/E]
  []
  [eq]
    type   = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
