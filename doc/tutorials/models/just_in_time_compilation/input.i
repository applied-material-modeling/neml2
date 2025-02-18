[Models]
  [eq1]
    type = ScalarVariableRate
    variable = 'state/a'
  []
  [eq2]
    type = ScalarVariableRate
    variable = 'state/b'
  []
  [eq3]
    type = ScalarVariableRate
    variable = 'state/c'
  []
  [eq]
    type = ComposedModel
    models = 'eq1 eq2 eq3'
  []
[]
