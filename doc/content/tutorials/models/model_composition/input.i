[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'forces/a'
    invariant = 'state/a_bar'
    invariant_type = I1
  []
  [eq2]
    type = SR2Invariant
    tensor = 'state/b'
    invariant = 'state/b_bar'
    invariant_type = VONMISES
  []
  [eq3]
    type = SR2LinearCombination
    from_var = 'forces/a state/b'
    to_var = 'state/b_rate'
    coefficients = '1 1'
    coefficient_as_parameter = true
  []
[]
