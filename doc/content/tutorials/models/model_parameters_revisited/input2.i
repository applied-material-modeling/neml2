# Same composition as input1.i, but the CTE is read from a [Tensors]
# entry. The named tensor is a (2, 2)-shape batched Scalar, so the whole
# model evaluates on a (2, 2) batch.
[Tensors]
  [alpha]
    type = Python
    expr = 'Scalar(torch.tensor([[1e-6, 2e-6], [1e-5, 5e-7]], dtype=torch.float64))'
  []
[]

[Models]
  [eq1]
    type = ThermalEigenstrain
    reference_temperature = '300'
    CTE                   = 'alpha'   # ← name of the [Tensors] entry above
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
    coefficients      = '2e5            0.3'
  []
  [eq]
    type   = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
