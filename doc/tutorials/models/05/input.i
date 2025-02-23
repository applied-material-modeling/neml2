c_aa = 1
c_ab = 2
c_ba = 3
c_bb = 4

[Models]
  [eq1]
    type = SR2Invariant
    tensor = 'state/a'
    invariant = 'state/a_bar'
    invariant_type = I1
  []
  [eq2]
    type = ScalarLinearCombination
    from_var = 'state/a_bar state/b'
    to_var = 'state/a_rate'
    coefficients = '${c_aa} ${c_ab}'
    coefficient_as_parameter = true
  []
  [eq3]
    type = ScalarLinearCombination
    from_var = 'state/a_bar state/b'
    to_var = 'state/b_rate'
    coefficients = '${c_ba} ${c_bb}'
    coefficient_as_parameter = true
  []
[]
