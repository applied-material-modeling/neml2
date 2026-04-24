[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'a'
    invariant = 'a_bar'
    invariant_type = I1
  []
  [eq2]
    type = SR2Invariant
    tensor = 'b'
    invariant = 'b_bar'
    invariant_type = VONMISES
  []
  [eq3]
    type = SR2LinearCombination
    from = 'a b'
    to = 'b_rate'
    weights = 'eq2 eq1'
    weight_as_parameter = true
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
